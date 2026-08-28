# Chapter 15 — KV store and the ring
> **status:** polished  ·  **path:** Muse Glimmer, pinned Muser tree
>
> *Prerequisites: [Ch 9](09-muse-glimmer-architecture.md) (the 39/13 layer
> split), [Ch 14](14-qk-norm-and-rope.md) (rotated K, unrotated V, and why
> NoPE bytes are position-free), [Ch 13](13-the-qkv-gate-matvec-family.md)
> (the K/V projections that produce this chapter's inputs). This is an
> infrastructure chapter: the data structure every attention dispatch of
> [Ch 16](16-attention-decode-kernels.md) reads.*

---

## 15.1 What the KV store computes

Chapter 14 ended with Q rotated (on the sliding layers), K likewise, V
untouched — and the current token's K and V waiting to be written into the
store attention will read. This chapter is that write.

Before any code, the question this chapter answers: where does a long
session's memory actually live, and what does keeping it cost? The answer
starts one step earlier, with why a store has to exist at all.

To produce a token, attention must score the current query against the
**Key** of every visible past token and take a weighted sum of their
**Values** ([Ch 16](16-attention-decode-kernels.md) does the math from
zero). The Key and Value of past token `i` depend only on token `i` — they
are projections of that token's hidden state ([Ch 13](13-the-qkv-gate-matvec-family.md))
— and once computed they never change. So every engine faces the same
choice: recompute all past K/V every step (O(n²) work over a generation —
token 100,000 recompute 99,999 keys it already computed) or compute each
once, **store** it, reuse it forever. The store is the
[KV cache](../glossary.md#kv-cache).

This chapter is about the *write* side: the per-layer GPU buffers, the two
storage regimes Muse Glimmer's architecture forces, the kernels that copy
this token's K and V into them, and the ring arithmetic that decides
*where* the write lands. The read side is [Ch 16](16-attention-decode-kernels.md);
the portable-asset side is [Part V](22-the-price-of-context.md).

**The recurrence that motivates it all is per layer class.** For the 13
NoPE layers, "every past token" grows without bound — the cache genuinely
grows with context. For the 39 sliding layers, attention only ever sees
the last `sliding_window = 2,048` tokens (`config.rs:16, 131`), so the
recompute-if-you-don't-cache body is bounded at 2,048 keys — small enough
that a *fixed-size ring* whose capacity never grows is the natural store.
One model, two memory regimes, decided by `layer % 4 == 3`
(`config.rs:84-93`).

## 15.2 The planes in code — `MetalKvPlane`

Start with the object, because everything the ring does later is
bookkeeping held in its fields — and it is worth knowing up front what a
wrong field costs. A plane that loses track of which row is its oldest
does not crash; it hands attention some other token's key and keeps
going. The per-layer cache object is seven fields:

```rust
// crates/muser-engine/src/decode.rs:182
struct MetalKvPlane {
    key: GpuHalfBuffer,
    value: GpuHalfBuffer,
    capacity: usize,
    len: usize,
    origin_logical: usize,
    origin_physical: usize,
    head_major: bool,
}
```

- **`key` / `value`** — two [f16](../glossary.md#f16) buffers; *f16 on every
  Metal lane* (the `GpuHalfBuffer` type is the constraint). F32 planes
  exist only in the kvpack interchange, never in live decode.
- **`capacity` / `len`** — the plane holds at most `capacity` token rows;
  `len` are currently live.
- **`origin_logical` / `origin_physical`** — the ring's explicit
  bookkeeping: which *logical* token position sits at the front of the
  live window, and which *physical* slot it occupies. Said the other way
  round, the plane never asks "where does token *n* belong?"; it asks
  "how far has my window slid since I last knew where its front was?"
- **`head_major`** — which of the two layouts this plane uses (§15.3).

Those two origin fields are the whole difference between this ring and
the obvious alternative, addressing straight by "position mod capacity",
and we did not choose the long way round for elegance. The ancestor
engine took the obvious route and it bit: the extraction manifest records
the naive form as a named out-of-bounds hazard — "Ferrite indexed by
absolute position; the ring modulus was unwired/stubbed" — which is why
the SWA ring-address translation is Muser-owned rather than inherited
(`docs/extraction-manifest.md`). Carrying the origins explicitly costs
two integers per plane and deletes that whole class of bug.

Allocation is per layer kind, and it is where the two regimes become
concrete:

```rust
// crates/muser-engine/src/decode.rs:1344
let mut cache = Vec::with_capacity(cfg.n_layers);
for layer in 0..cfg.n_layers {
    let capacity = match cfg.layer_kinds[layer] {
        MuseLayerKind::SlidingRope => max_context.min(cfg.sliding_window).max(32),
        MuseLayerKind::FullNoPe => max_context.max(32),
    };
    // Zero-filled on purpose: wrapped SWA rows must never expose
    // uninitialized storage during a sequence boundary transition.
    cache.push(MetalKvPlane::new(
        &shared.context,
        capacity,
        cfg.kv_dim(),
        matches!(cfg.layer_kinds[layer], MuseLayerKind::FullNoPe),
    )?);
}
```

A sliding layer gets `min(max_context, 2,048)` rows, **token-major**; a
NoPE layer gets `max_context` rows, **head-major**.

The zero-fill comment sitting in that loop is worth a detour, because the
reason it matters for anyone reading Muser's docs beside this book is
that we had the fact backwards in writing before we had it right. An
engineering doc stated that Metal KV buffers were allocated without a CPU
memset. It is a plausible claim — a buffer that is about to be overwritten
row by row looks like a pure waste of a memset — and it stood long enough
to be quoted. The audit that went checking found the opposite for live
planes: they zero-fill on purpose, so that a wrapped SWA row can never
expose uninitialized storage while a sequence boundary is in flight. Only
detached remote-install generations take the uninitialized path. The
correction is retained, and it is the corrected fact that should be cited,
never the stale line:
`[docs/kvpack-merge-handoff §3 D2, per the 2026-08-20 audit]`.

## 15.3 The two layouts

Why should one model carry two layouts at all? Because a layout is never
chosen for the writer's convenience. It is chosen for whoever reads it
most often — and the two layer classes have different readers, so they
get different shapes.

`kv_dim = n_kv_heads × head_dim = 2 × 128 = 256` halves per token row.
The two planes lay those halves out differently:

```
  TOKEN-MAJOR (SWA ring), capacity C = 2,048:

    [token 0   ][token 1   ]…[token C−1 ]   each row = 256 halves
     k0h0 k0h1  k1h0 k1h1                      (h0 = KV head 0's 128,
                                                  h1 = KV head 1's 128)
    index: physical × kv_dim + element

  HEAD-MAJOR (NoPE growing), capacity C = max_context:

    KV head 0: [tok 0][tok 1]…[tok C−1]   128 halves per row
    KV head 1: [tok 0][tok 1]…[tok C−1]
    index: (kv_head × capacity + physical) × head_dim + dim
```

*Figure 15.1: the two KV layouts. Token-major keeps one token's two head
rows adjacent — a ring rotation moves whole tokens. Head-major keeps each
KV head's sequence contiguous — a head's whole history is one linear span,
which is exactly the shape llama.cpp's flash-attn vec kernel wants for its
head-major `ns10 = 128` addressing (`metal/encode/attn.rs:503-511`).* The
index formulas are the store kernels' own (`muse_reference.metal:1224`,
`:1228`); why each layout pairs with its layer class is §15.6's tradeoff.

## 15.4 The store kernels

With the shapes settled, the write itself is an anticlimax, and that is
the design. The useful question here is not "how does the store work?"
but "how little is the store allowed to know?" — because every fact the
GPU kernel is told about token positions is a fact that can be wrong on
the GPU, where nothing checks it.

Two kernels write K/V; both are pure quantize-free copies — the values are
already f32 in the activation buffers and f16 in the planes.

**Single-token, token-major** — the SWA ring's steady-state write:

```metal
// crates/muser-engine/src/shaders/muse_reference.metal:979
kernel void muser_kv_store_f16(
    device const float *key [[buffer(0)]],
    device const float *value [[buffer(1)]],
    device half *key_cache [[buffer(2)]],
    device half *value_cache [[buffer(3)]],
    constant uint &kv_dim [[buffer(4)]],
    constant uint &write_index [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (index < kv_dim) {
        uint destination = write_index * kv_dim + index;
        key_cache[destination] = key[index];
        value_cache[destination] = value[index];
    }
}
```

One thread per element; 256 threads; the row's position (`write_index`)
arrives pre-computed from the ring arithmetic of §15.5 — the kernel never
sees an absolute token position. The f32→half conversion is implicit in
the assignment. The wrapper binds buffers and dispatches
`dispatch_1d(key.len())` (`metal/encode/attn.rs:635-655`).

**Batched, both layouts** — the NoPE planes and prefill chunks:

```metal
// crates/muser-engine/src/shaders/muse_reference.metal:1203
kernel void muser_kv_store_batch_f16(
    device const float *key [[buffer(0)]],
    device const float *value [[buffer(1)]],
    device half *key_cache [[buffer(2)]],
    device half *value_cache [[buffer(3)]],
    constant uint &kv_dim [[buffer(4)]],
    constant uint &source_first [[buffer(5)]],
    constant uint &source_count [[buffer(6)]],
    constant uint &start_position [[buffer(7)]],
    constant uint &capacity [[buffer(8)]],
    constant uint &origin_logical [[buffer(9)]],
    constant uint &origin_physical [[buffer(10)]],
    constant uint &head_dim [[buffer(11)]],
    constant uint &head_major [[buffer(12)]],
    uint index [[thread_position_in_grid]]) {
    uint total = source_count * kv_dim;
    if (index < total) {
        uint source_token = source_first + index / kv_dim;
        uint element = index % kv_dim;
        uint logical = start_position + source_token;
        uint physical = (origin_physical + logical - origin_logical) % capacity;
        uint destination = physical * kv_dim + element;
        if (head_major != 0u) {
            uint kv_head = element / head_dim;
            uint dim = element % head_dim;
            destination = (kv_head * capacity + physical) * head_dim + dim;
        }
        uint source = source_token * kv_dim + element;
        key_cache[destination] = key[source];
        value_cache[destination] = value[source];
    }
}
```

The layout switch is the `if (head_major)` block: token-major lands at
`physical × kv_dim + element`; head-major re-interleaves to
`(kv_head × capacity + physical) × head_dim + dim` — Figure 15.1's two
index formulas, one kernel. Note that this kernel *does* see absolute
positions (`start_position`, `origin_logical`) but only inside the
explicit-origin translation `physical = (origin_physical + logical −
origin_logical) % capacity` — never as `logical % capacity`.

## 15.5 Ring write-position arithmetic — `append`

Everything so far has deferred one question: which row does this token's
K and V land in? The stake is unusually quiet. Get the row wrong and
nothing crashes — attention simply scores the query against some other
token's key, the logits shift, and the only symptom is generated text
that is subtly worse than it should be, at a rate no test that checks for
crashes will ever catch. So the arithmetic that answers the question is
small, explicit, and refuses to guess. The CPU-side reservation that
decides `write_index` is eleven lines and fail-closed:

```rust
// crates/muser-engine/src/decode.rs:263
/// Reserve the physical row for `position` and advance explicit ring
/// metadata. No physical placement is derived from the absolute token ID.
fn append(&mut self, layer: usize, position: usize) -> Result<usize, MetalModelError> {
    let expected = self.origin_logical + self.len;
    if position != expected {
        return Err(MetalModelError::CacheDiscontinuity {
            layer,
            expected,
            got: position,
        });
    }
    if self.len < self.capacity {
        let write = (self.origin_physical + self.len) % self.capacity;
        self.len += 1;
        Ok(write)
    } else {
        let write = self.origin_physical;
        self.origin_logical += 1;
        self.origin_physical = (self.origin_physical + 1) % self.capacity;
        Ok(write)
    }
}
```

Walk it at the ring boundary, capacity 2,048:

```
  positions 0..2047 (fill):  write = origin_physical + len, len → 2048
                             origin_logical = 0, origin_physical = 0

  position 2048 (first wrap):
      len == capacity → write = origin_physical = 0     (overwrite slot 0)
      origin_logical  → 1        (the window now starts at token 1)
      origin_physical → 1        (slot 1 is the oldest live row)

  position 2049:  write = 1; origins → (2, 2).   And so on, forever.
```

*Figure 15.2: the ring wrap. The write pointer, the logical origin, and
the physical origin advance together; the plane's physical layout is a
rotation of the logical tail.*

Three properties fall out of the wrap arithmetic of Figure 15.2:

1. **Fail-closed continuity.** `position != origin_logical + len` is an
   error, not a modulo — a skipped or replayed position can never silently
   alias a live row (`decode.rs:117-121` defines the error; prefill's
   module doc states the invariant: "physical placement is never derived
   from absolute position", `prefill.rs:15-17`).
2. **NoPE degenerates to append-only.** A NoPE plane's capacity is
   `max_context`; `len < capacity` always holds in a valid session, so the
   `else` branch never runs, `origin_logical` stays 0, and the modulo
   vanishes — the "ring" is a growing array.
3. **`append_batch`** (`decode.rs:286-314`) is the chunked form: it
   advances the origins by the overflow when a chunk crosses capacity and
   hands back which source rows are still live — the arithmetic behind
   prefill's ring wrap.

The `write_physical` this returns is exactly the `write_index` the store
kernel of §15.4 receives (`decode.rs:5643, 5661-5667`).

## 15.6 Why restore must preserve rotation

The plane's rotation (where `origin_physical` points) looks like
implementation detail. It is not — it is *numerics*. Attention scans rows
in physical order and floating-point accumulation is order-sensitive, so
two planes holding identical rows in different rotations produce
different last-bit logits.

Put it plainly, because this is the idea in the chapter most worth
re-reading: the same keys, stored starting at a different slot, are
summed in a different order, and a different order of floating-point
additions is a different number. The rotation is not where the data
happens to sit. The rotation is part of the data. The restore path
documents exactly this, with a test name in the comment:

```rust
// crates/muser-engine/src/decode.rs:376
// Install at the rotation a sequentially-built live ring holds at this
// logical origin. Attention scans rows in physical order and float
// accumulation is order-sensitive, so a restore packed at origin 0
// can never replay a wrapped live session's logits bitwise (caught by
// real_model_wrap_boundaries_and_detached_restore_replay_exactly).
// NoPE planes never wrap (origin_logical is always 0), so their
// rotation is 0 and this reduces to the previous layout.
let rotation = snapshot.origin_logical % snapshot.capacity;
```

`detached_from` then scatters the snapshot's *logical* rows back at
`(rotation + logical_offset) % capacity` (`decode.rs:384-407`) — head-major
per KV head, token-major as a split head/tail copy. Without this, a saved
session restored on another process would produce bit-different logits
from the same bytes, and the exactness gates of
[Ch 38](38-measuring-against-llama-cpp.md) (and kvpack's bit-identical
warm hits, [Ch 25](25-warm-reuse.md)) would fail for a reason no amount of
weight-side care could fix. Bitwise replay requires the *rotation* to
travel with the cache.

## 15.7 Footprint per token, derived by hand

Now the bill. Two questions a reader actually has — what does a session
cost in memory, and where does that cost concentrate — are answerable
with nothing but multiplication, so we would rather you re-derive them
than take our word.

One K row per layer = `n_kv_heads × head_dim × 2 B = 2 × 128 × 2 = 512 B`;
K and V together = **1,024 B per layer per token** — topology-derived
arithmetic, not a measured RSS `[docs/memory-footprint.md §KV formula]`.
Derive the three numbers that matter:

```
  per token, whole model:  52 layers × 1,024 B = 53,248 B ≈ 52 KiB

  SWA steady state (bounded):  39 × 2,048 × 1,024 = 81,788,928 B ≈ 78 MiB
      — constant, whatever the context depth

  NoPE at max context:  13 × 131,072 × 1,024 = 1,744,830,464 B ≈ 1.66 GiB
      — grows linearly, unbounded to the 131,072 model limit

  one slot at 131,072:  81,788,928 + 1,744,830,464 = 1,826,619,392 B
                      ≈ 1.827 GB (decimal)
```

which is the memory-footprint table's one-slot figure (1.827 GB; four
slots 7.306 GB) `[docs/memory-footprint.md, via the measured-numbers
ledger §1k]`. Note the asymmetry: at full depth, ~96 % of the KV footprint
is the 13 NoPE layers. The same split shows up on the wire: the deep
130,815-token handoff's 1,823,184,896 B payload decomposes as
`130,814 × 13,312 B` of NoPE (13 layers × 1,024 B per token) plus three
pipe-safe SWA groups of `2,048 × 13,312 B` — i.e. ≈95.5 % NoPE / ≈4.5 %
SWA by these terms `[receipt phase4-disagg-20260820/130815-g900091/;
docs/kvpack-merge-handoff §6]`. [Ch 22](22-the-price-of-context.md) owns
the full footprint treatment (2k/32k/131k tables, slot arithmetic); keep
this chapter's derivation as the per-layer-class split it is.

## 15.8 Tradeoffs

The store could have been built differently at four points — how the
cells are compared, how the planes are laid out, what precision they
hold, and whether the store is its own dispatch at all. Each alternative
deserves a price rather than a dismissal. Start with the frame the rest
of them are argued in.

**A 2×2 that splits the confounds.** The ancestor book's sharpest KV
device was a 2×2 layout × precision matrix whose ratios multiplied out to
the combined effect, exposing that a flag named "addressing" was mostly a
precision swap `[ferrite-book Ch 14]`. Muser's KV matrix has the same
shape with different axes — **layout (token-major vs head-major) × layer
class (SWA-RoPE vs NoPE)** — and the decomposition that matters is the
payload split of §15.7: at depth, the NoPE cells carry ≈95.5 % of the
bytes and the SWA rings ≈4.5 %.

That split, not any kernel choice, is what shaped the disaggregated
lane's send schedule: stream the "SWA groups (~82 MB) early" and hold
"the NoPE bulk (95.7 % of payload)" back until prefill finishes layer 51.
The reasoning felt free — the small planes are ready first, so send them
first and buy overlap for nothing. On the wire it was not free. A
burst-then-wait shape is exactly the traffic pattern that lets an
EEE-capable link drop into an idle state between bursts, and the pacing
section of the merge handoff is where that collision is written down
(`[docs/kvpack-merge-handoff §6 "Pacing reality"]`);
[Ch 31](31-the-wire-discipline.md) tells the story properly. The lesson
outlives the wire: the payload split is an input to schedules made three
layers of abstraction away, so when you change one cell of this 2×2,
re-derive the split before predicting anything downstream of it.

**Ring + growing plane vs one layout, vs paging.** Why not one layout for
all 52 layers? A token-major *growing* NoPE plane would make each head's
history strided, defeating the head-major `ns10=128` addressing of the
pinned attention kernel; a head-major *ring* would rotate each head
independently — same rotation, but the store/restore walk gets no
contiguous tail and the llama vec path cannot read a wrapped ring as one
span. The hybrid pairs each class with the layout its reader wants. Why
not *paging* (16-token blocks behind a block table), the ancestor's
design? Because Muse Glimmer's SWA layers bound themselves — a full ring
*is* the whole live set, nothing to evict — and the NoPE layers' prefix
sharing is handled one level up, by kvpack's content-addressed chunks
([Ch 24](24-kvpack-the-format.md)), not by an in-GPU page table. The
OS-paging analogy survives only as contrast `[ferrite-book Ch 14]`.

**f16, not quantized, not f32.** The planes are f16 on every Metal lane
(the `GpuHalfBuffer` field type, `decode.rs:183-184`). The ancestor
quantized its cache to Q8_0 and measured an 80/15 precision/addressing
split when it stopped `[ferrite-book Ch 14]` — Ferrite-lineage context,
not a Muser decision point: Muser's parity anchor (pinned llama.cpp) runs
F16 KV, so Muser does too, and the lane table's decode rows all say
"FP16 KV" `[docs/muser-architecture.md]`. The kvpack *interchange* still
carries both encodings (`PlaneEncoding::{F16Le, F32Le}`,
`crates/muser-engine/src/cache.rs:11-16`) because the wire format serves
producers and archives beyond the live planes. A quantized live cache
would halve §15.7's numbers and is exactly the kind of change the
exactness contract makes a research lane, not a default.

**Store-then-barrier vs fused store-in-attention.** Muser's decode stores
K/V in its own dispatch and puts an explicit
`memory_barrier_with_resources` between store and attention
(`decode.rs:5669-5670`) when the pinned vec kernel follows. The ancestor
fused the quantizing store *into* its attention kernel to save a dispatch
`[ferrite-book Ch 15]`. Muser's store is a 512-byte-per-plane copy with
nothing to amortize, and the split keeps the pinned llama kernels
untouched — the same pin-the-boundary reasoning as
[Ch 13](13-the-qkv-gate-matvec-family.md). The ferrite interleaved
fallback kernel *does* fuse its store (`flash_attn_decode_vec_contiguous_
f16.metal:534-543`) — on that route only, one simdgroup writes the current
K/V as a side effect; see [Ch 16](16-attention-decode-kernels.md).

## 15.9 Where the gap lives

The book's running question about the decode gap is which dispatches are
waste and which are structure. This chapter is where that question gets
its least comfortable answer, twice.

Two of the four +196 families live here. **52 KV-publication splits**
(production's separate `kv_store` + `attention` closures vs legacy's
combined `kv_store_attention`) are classed "session/publication structure
— Keep; combining closures alone does not remove their kernel math"
`[docs/decode-dispatch-gap-20260815.md §Corrected closure-count diff]`.
And **39 SWA wrapped-ring staging groups** are the batch graph's
multi-row ring feature — the staging shadow of
[Ch 36](36-prefill-vs-decode-paths.md) — kept "until a bit-exact
ring-aware replacement exists." Neither is waste in the ordinary sense.
Both are dispatches that exist because removing them would change a
number the exactness contract does not allow to change — which is a
harder thing to argue away than a merely inefficient loop. The
note's own ranked list still hopes for "a one-row ring-aware attention
path that avoids SWA staging, gated by bitwise KV and full-logit equality
at positions 1, 31, 32, 33, 2,047, 2,048, and 2,049." This chapter is
where those two rows of the table become concrete: the splits are the
store dispatches you just read; the staging groups are what wrapping a
2,048-ring under a 512-token prefill chunk costs.

## 15.10 What comes next

The planes hold every visible token's K and V — a ring for the sliding
layers, a growing span for the full ones — and the current token's Q is
rotated and QK-normalized. Everything attention needs is in place. There
is no single "attention kernel" to read next: there is a **route ladder**,
selected per layer per token by alignment predicates. That ladder — and
the Q·Kᵀ softmax V math it implements — is
[Ch 16](16-attention-decode-kernels.md).

## References

- `crates/muser-engine/src/decode.rs:182-190` — `MetalKvPlane`.
- `crates/muser-engine/src/decode.rs:228-261` — zero-filled and
  uninitialized constructors.
- `crates/muser-engine/src/decode.rs:263-314` — `append` / `append_batch`
  (the ring arithmetic; quoted).
- `crates/muser-engine/src/decode.rs:374-417` — `detached_from` and the
  rotation-preserving restore (quoted).
- `crates/muser-engine/src/decode.rs:1344-1358` — per-layer-kind capacity
  and layout allocation (quoted).
- `crates/muser-engine/src/decode.rs:5643-5745` — the store + route call
  sites in `encode_token`; `:117-121` `CacheDiscontinuity`.
- `crates/muser-engine/src/shaders/muse_reference.metal:979-992` —
  `muser_kv_store_f16` (quoted).
- `crates/muser-engine/src/shaders/muse_reference.metal:1203-1234` —
  `muser_kv_store_batch_f16`, both layouts (quoted).
- `crates/muser-engine/src/metal/encode/attn.rs:635-655, 786-827` — the
  two store wrappers.
- `crates/muser-engine/src/prefill.rs:15-17` — the
  no-placement-from-absolute-position invariant.
- `crates/muser-engine/src/cache.rs:11-16` — `PlaneEncoding` (interchange
  encodings).
- `crates/muser-engine/src/config.rs:13-21, 84-93` — layer counts, window,
  the partition rule.
- `crates/muser-engine/src/lib.rs:7-15` — the position-free-NoPE summary.
- `[docs/memory-footprint.md]` — the 1,024 B/row formula and the
  1.827/7.306 GB slot table (§15.7's tags).
- `[receipt phase4-disagg-20260820/130815-g900091/]` — the
  1,823,184,896 B deep payload (`payload_bytes` verified).
- `[docs/kvpack-merge-handoff §6]` — the ~82 MB SWA / 95.7 % NoPE pacing
  decomposition.
- `[docs/decode-dispatch-gap-20260815.md]` — the 52 publication splits and
  39 SWA staging rows (§15.9).
- [Ch 16](16-attention-decode-kernels.md) — the read side.
- [Ch 22](22-the-price-of-context.md), [Ch 23](23-the-swa-ring-and-the-growing-cache.md),
  [Ch 24](24-kvpack-the-format.md) — footprint, server policy, the
  portable format.
- `[ferrite-book Ch 14]` — the ancestor's paged-Q8 cache (the 2×2 device
  and the paging contrast, ported as contrast only).
