# Chapter 23 — The SWA ring and the growing cache
> **status:** polished  ·  **path:** Muse Glimmer, pinned Muser tree

*Prerequisites: [Ch 22](22-the-price-of-context.md) (the two cost curves),
[Ch 15](15-kv-store-and-the-ring.md) (the planes, the layouts, the store
kernels — this chapter goes deeper, not around), [Ch 16](16-attention-decode-kernels.md)
(the route ladder that reads these planes), [Ch 8](08-the-dflash-draft.md)
(speculative blocks that must be able to roll back).*

---

## 23.1 What this chapter is about

[Ch 22](22-the-price-of-context.md) ended with two curves: one flat at
81,788,928 B (the 39 sliding layers, bounded at 2,048 rows), one climbing to
1,744,830,464 B (the 13 full layers, growing to 131,072). This chapter is
the machinery under the curves — not *whether* the memory exists, but *how a
token gets into it, how a chunk of tokens crosses the ring boundary without
destroying the attention inputs, how the same bytes come back out as a
snapshot, and what the server does when the context outgrows the model's
limit.*

The recurring lesson of this chapter is that the ring is not a storage
detail. Its rotation is numerically observable (bitwise replay depends on
it), its wrapped shape constrains which pinned kernels may read it, and its
overwrite-in-place semantics force a stage-then-commit protocol in prefill
and a row-retention protocol in speculative decoding. The growing NoPE plane
has none of these problems — which is exactly why it, and not the ring, is
the part of the cache that becomes a portable asset in
[Ch 24](24-kvpack-the-format.md).

One piece of background from [Ch 15 §15.2](15-kv-store-and-the-ring.md),
kept to one paragraph: each layer's cache is a `MetalKvPlane` — two f16
buffers plus five fields of explicit bookkeeping (`capacity`, `len`,
`origin_logical`, `origin_physical`, `head_major`,
`decode.rs:182-190`). Sliding layers get capacity
`min(max_context, 2,048)` **token-major**; full layers get `max_context`
**head-major** (`decode.rs:1346-1348`). Physical placement is never derived
from an absolute token position — prefill's module doc states the invariant
in one line ("physical placement is never derived from absolute position",
`prefill.rs:15-17`).

## 23.2 Reserving rows: `append` for one token, `append_batch` for a chunk

[Ch 15 §15.5](15-kv-store-and-the-ring.md) walked the single-token
`append` (`decode.rs:263-284`): fail-closed continuity against
`origin_logical + len`, then either write at
`(origin_physical + len) % capacity` (filling) or overwrite
`origin_physical` and advance both origins (wrapping). Decode calls it once
per layer per token (`decode.rs:5643`). Prefill chunks need the batched
form, and its arithmetic is where the ring's edge cases live:

```rust
// crates/muser-engine/src/decode.rs:286
fn append_batch(
    &mut self,
    layer: usize,
    start_position: usize,
    token_count: usize,
) -> Result<(usize, usize), MetalModelError> {
    let expected = self.origin_logical + self.len;
    if start_position != expected {
        return Err(MetalModelError::CacheDiscontinuity {
            layer,
            expected,
            got: start_position,
        });
    }
    let total = self
        .len
        .checked_add(token_count)
        .ok_or_else(|| MetalModelError::InvalidSnapshot("cache length overflow".into()))?;
    if total <= self.capacity {
        self.len = total;
    } else {
        let overflow = total - self.capacity;
        self.origin_logical += overflow;
        self.origin_physical = (self.origin_physical + overflow) % self.capacity;
        self.len = self.capacity;
    }
    let source_first = self.origin_logical.saturating_sub(start_position);
    Ok((source_first, token_count - source_first))
}
```

Walk the two regimes with numbers.

**Wrapping (SWA, from the first chunk that crosses 2,048).** `len = 1,900`,
chunk of 512, capacity 2,048: `total = 2,412 > 2,048` — this chunk *does*
overflow a SWA ring, so take it as the wrap case directly. `overflow = 364`; the window's
logical start advances 364 rows (`origin_logical: 0 → 364`), the physical
origin advances 364 rows, `len` saturates at 2,048. The 364 evicted rows
were the oldest *ring* rows, so every row of the new chunk is still live:
`source_first = 364 − 1,900` saturates to `0`, and the function returns
`(0, 512)` — all 512 source rows survive. In steady state (`len` already
`== capacity == 2,048`, chunk 512), `overflow = 512`, again all from old
rows, again `(0, 512)`. The returned pair exists for the asymmetric case
where the chunk itself is wider than the whole window (`token_count >
capacity`, possible only in tiny test geometries): then part of the chunk's
own head has already scrolled out before the chunk ends, and
`source_first > 0` names the first surviving source row. The
`saturating_sub` is the guard that keeps that case a number, not a panic.

**NoPE degenerates to pure append.** A growing plane's capacity is
`max_context`, `total <= capacity` always holds inside valid context bounds,
the `else` branch never runs, `origin_logical` stays 0, `origin_physical`
stays 0 — the modulo never fires and the "ring" is an array. That is not an
accident; it is the design's way of making one code path serve two regimes:
the ring machinery costs a NoPE plane nothing.

Both `append` forms fail closed before any GPU write happens:
`CacheDiscontinuity` ("Metal KV cache for layer {layer} expected logical
position {expected}, got {got}", `decode.rs:117-122`) means a skipped or
replayed position can never silently alias a live row. When it trips, the
operator sees the layer index and both positions — enough to find the caller
that broke continuity.

## 23.3 Crossing the wrap in prefill: the staging shadow

A wrapped ring has a property that a contiguous buffer never has: **the
live window is split across the physical array's seam.** Rows `[origin ..
capacity)` hold the older half, rows `[0 .. origin)` hold the newer half.
The attention kernels of [Ch 16](16-attention-decode-kernels.md) want one
linear span of rows. And a chunk that wraps will *overwrite* the oldest live
rows — the very rows this chunk's queries must still attend to. Writing
in place would be a [WAR](../glossary.md#war-hazard) hazard committed against
your own inputs.

So a wrapped SWA prefill does not write the ring at all until attention is
done. It stages:

```rust
// crates/muser-engine/src/metal/encode/attn.rs:103
pub fn encode_stage_swa_prefill_f16(
    &self,
    encoder: &ComputeCommandEncoderRef,
    current_key: &GpuBuffer,      // this chunk's fresh K (f32, from projections)
    current_value: &GpuBuffer,
    ring_key: &GpuHalfBuffer,     // the live (rotated) ring
    ring_value: &GpuHalfBuffer,
    staged_key: &GpuHalfBuffer,   // detached shadow: old rows ‖ new rows
    staged_value: &GpuHalfBuffer,
    kv_dim: usize,
    old_len: usize,
    old_origin_physical: usize,
    ring_capacity: usize,
    token_count: usize,
) {
```

The kernel `muser_stage_swa_prefill_f16` (`shaders/muse_reference.metal:1240`)
un-rotates the old ring rows into logical order in the shadow, appends the
chunk's rows after them, and attention runs over the shadow as one
contiguous span — the FA2 route (`encode_flash_attention_v2`,
`decode.rs:4329-4346`) or the llama vec route with its padded-index
materialization (§23.7). Only after attention does the CPU commit the
reservation:

```rust
// crates/muser-engine/src/decode.rs:4348
let (_source_first, _source_count) = self.cache[layer_index].append_batch(
    layer_index,
    start_position,
    token_count,
)?;
```

and the next chunk sees a consistently rotated ring. The non-wrapping
fallback branch keeps the ordering explicit with a comment — "Attend before
overwriting any still-visible old rows" (`decode.rs:4356`) — the same law,
stated for the route where the overwrite is partial.

This staging is not free, and the campaign counted the cost: the batch
graph's wrapped-ring work appears in the dispatch-gap accounting as **39 SWA
wrapped-ring staging groups**, one per sliding layer, kept "until a
bit-exact ring-aware replacement exists" `[docs/decode-dispatch-gap-20260815.md
§Corrected closure-count diff]`. It is structure the exactness contract
refuses to cheapen — same verdict as the 52 KV-publication splits of
[Ch 15 §15.9](15-kv-store-and-the-ring.md).

## 23.4 Snapshots: logical order going out, rotation preserved coming back

To hand a plane to anything outside the engine — a durable pack
([Ch 24](24-kvpack-the-format.md)), a migration ([Ch 26](26-delta-handoff-and-migration.md))
— the rotated physical layout must become *logical*: ascending token order,
no rotation, no head interleave assumptions about the consumer. The snapshot
walk does exactly that, per layout:

```rust
// crates/muser-engine/src/decode.rs:327
for logical_offset in 0..self.len {
    let physical = (self.origin_physical + logical_offset) % self.capacity;
    if self.head_major {
        for kv_head in 0..kv_dim / head_dim {
            let start = (kv_head * self.capacity + physical) * head_dim;
            key_logical.extend_from_slice(&key[start..start + head_dim]);
            value_logical.extend_from_slice(&value[start..start + head_dim]);
        }
    } else {
        let start = physical * kv_dim;
        key_logical.extend_from_slice(&key[start..start + kv_dim]);
        value_logical.extend_from_slice(&value[start..start + kv_dim]);
    }
}
```

The reverse — `detached_from` (`decode.rs:351-417`) — is where the ring
teaches its one numerics lesson, and [Ch 15 §15.6](15-kv-store-and-the-ring.md)
quoted the load-bearing comment: attention scans rows in physical order,
float accumulation is order-sensitive, so a restore "packed at origin 0 can
never replay a wrapped live session's logits bitwise"
(`decode.rs:376-380`). The fix is one line of arithmetic plus a
layout-aware scatter:

```text
rotation = origin_logical % capacity        (decode.rs:383)

token-major:  copy the logical rows as a head at [rotation .. capacity)
              and a tail at [0 .. rotation)         (decode.rs:397-406)
head-major:   for each logical row, for each KV head:
                  destination = (head * capacity + (rotation + logical) % capacity)
                                                       (decode.rs:384-395)
```

Note what the rotation formula says about the two regimes: a NoPE plane's
`origin_logical` is always 0, so its rotation is always 0 and the head-major
scatter reduces to a plain copy at physical = logical — the ring logic
switches itself off. A restored SWA ring comes back rotated exactly as a
sequentially-built live ring would sit at that logical origin, because
`origin_logical % capacity` is where sequential appends would have left the
physical origin. The property is test-enforced, not aspirational — the
comment names the test: `real_model_wrap_boundaries_and_detached_restore_
replay_exactly` (`decode.rs:380`).

The interchange contract sits one level up, in `SessionCacheSnapshot` —
"A complete restorable cut. The 39 SWA layers contain the complete logical
tail and the 13 NoPE layers contain `[0, position)`"
(`crates/muser-engine/src/cache.rs:39-41`). Its shape gate is fail-closed
per layer: an SWA plane must carry exactly `min(position, window)` rows
starting at `position − count`, a NoPE plane exactly `position` rows from 0,
with byte lengths checked to the element (`cache.rs:62-118`). Two
consequences worth naming. First, the interchange never carries rotation —
rotation is *reconstructed* on install by the formula above, so a pack's
bytes are layout-stable while replay stays bitwise. Second, the CPU oracle
is deliberately not a ring — it allocates f32 full-history planes and
applies the window as a mask — so CPU and Metal snapshots are mutually
uninstallable (`F32Le` vs `ProductionF16Required`,
`cache.rs:14-17`; `muser-kvpack/src/session.rs:164-166`)
`[docs/kvpack-merge-handoff §4]`. Exactness by incompatibility: the two
backends cannot accidentally share state that only one of them defined.

## 23.5 The growing plane: why head-major suits append and relocation

The NoPE plane's layout — `[kv_head][capacity][head_dim]` — pairs with its
job in three ways, each anchored in code you have already met:

1. **The reader wants per-head spans.** The pinned llama.cpp
   `flash_attn_ext` vec kernel addresses KV head-major with
   `ns10 = 128` strides (`metal/encode/attn.rs:503-511`,
   [Ch 15 Figure 15.1](15-kv-store-and-the-ring.md)). A head's whole
   history is one linear span; the kernel never crosses a seam because a
   growing plane has no seam.
2. **Append never wraps.** §23.2's NoPE degeneration means physical order
   *is* logical order, `origin_logical = 0` forever, and the batch store
   kernel's head-major index —
   `(kv_head * capacity + physical) * head_dim + dim`
   (`muse_reference.metal:1228`) — is a plain row append per head.
3. **Relocation is memcpy.** The 13 NoPE layers apply no rotation
   ([Ch 14](14-qk-norm-and-rope.md): no RoPE at all), so a row's bytes do
   not encode its absolute position. Moving row 5,000 to a different
   machine, or installing it at a different physical offset, changes
   nothing about what attention computes with it. The engine's own module
   doc names this "the whole kvpack free lunch" (`lib.rs:8-10`), and the
   interchange's install math is the proof by construction — the head-major
   tile scatter in `write_f16_tile` (`cache.rs:205-222`) is byte movement
   plus index arithmetic, no numeric transformation anywhere.

The ring cannot make claim 3: an SWA key row was *rotated by RoPE at store
time* into its absolute position, so its bytes are position-bound. This
single asymmetry — position-free growing planes versus position-bound
bounded rings — drives everything from the transfer schedule (NoPE tiles
stream during CUDA prefill; SWA groups ride along as window snapshots,
[Ch 22 §22.7](22-the-price-of-context.md)) to delta admission rules
([Ch 26](26-delta-handoff-and-migration.md)).

## 23.6 The third interaction: speculative blocks and the checkpoint

DFlash speculative decoding ([Ch 8](08-the-dflash-draft.md),
[Ch 33](33-speculation-and-the-distributed-verdict.md)) proposes a block of
up to 16 tokens, the target verifies them, and on rejection the cache must
roll back to the block's start. The two regimes pay differently, and the
checkpoint type says so in its own doc comment:

```rust
// crates/muser-engine/src/decode.rs:208
/// Lightweight transactional checkpoint for one speculative verification
/// block. Growing NoPE planes only need their logical metadata rewound. SWA
/// planes may overwrite live ring rows, so the small set of destinations
/// touched by the candidate block is retained here instead of copying the
/// complete multi-gigabyte cache on every DFlash round.
pub(crate) struct MetalSpeculativeCheckpoint {
    start_position: usize,
    token_count: usize,
    planes: Vec<MetalSpeculativePlaneCheckpoint>,
}
```

A NoPE plane's speculative writes land in *unused* rows past `len` —
rewinding is restoring three integers (`origin_logical`,
`origin_physical`, `len`; `decode.rs:419-421`: "A NoPE plane grows into
unused storage, so only its logical metadata is kept"). An SWA plane at
steady state has no unused rows: the block overwrites up to `token_count`
live ring rows, so the checkpoint retains exactly those destinations'
physical indices and key/value bits (`decode.rs:428-434`) — at most 16 rows
× 1,024 B per plane, ~16 KiB per sliding layer, never a copy of the
multi-gigabyte cache. Commit discards the checkpoint; rollback restores the
rows and the metadata. The two regimes of this chapter, priced as rollback
protocols: metadata-only versus row-retention. (The commit/rollback driver
is `decode.rs:1386-1496`.)

## 23.7 Context shift: the engine has no shift, the server has a policy

What happens when the conversation outgrows 131,072 positions? Not an
engine operation. There is no shift, truncate, or evict op anywhere in
`MetalKvPlane` or the encode paths — the map's audit is blunt: "there is no
engine-level 'shift' op; the server owns the policy" `[code-map §6, per
source]`. The policy is two variants:

```rust
// crates/muser-server/src/state.rs:215
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextPolicy {
    Shift,
    Error,
}
```

`shift` is the default; `error` refuses the request
`[docs/muser-architecture.md §Context and sessions]`. The architecture doc
states what a shift preserves — and it is semantic units, not token counts:

> Chat shifting preserves system content and whole newest turns/tool/image
> units. Raw shifting preserves the configured prefix plus the newest
> suffix. A request is rejected if the minimum retained unit plus output
> reserve cannot fit. `[docs/muser-architecture.md §Context and sessions]`

The chat rule is built by splitting the message list into a system prologue
and *turns*, where a turn begins at each user message and everything after
it — assistant replies, tool calls, tool results, images — stays attached:

```rust
// crates/muser-server/src/openai.rs:5356
/// A shift may remove only complete units beginning at a user boundary.
/// Assistant calls, tool results, and image-bearing messages remain attached
/// to their turn and therefore move or disappear as one replay unit.
fn complete_chat_turns(messages: &[Message]) -> Vec<Vec<Message>> {
```

`shift_chat_units` (`openai.rs:5374-5391`) enforces the precondition (all
system messages precede the turns) and `prepare_with_context_policy`
(`openai.rs:5292-5317`) drops oldest turns in a loop until the retained
conversation fits `max_context − output_reserve`, rejecting when "system
content, newest complete turn, and output reserve cannot fit"
(`openai.rs:5311-5315`). The raw path keeps a configured prefix plus the
newest suffix (`compact_raw_prompt`, `openai.rs:5338-5354`).

**The rebuild is a staging production with atomic publication.** The
retained context is re-prefilled into the runtime's one hidden
full-capacity session — `staging`, "deliberately outside `slots`, so it can
never admit or decode a fifth serving request" (`state.rs:240-243`) — and
only a successful prefill promotes it:

```rust
// crates/muser-server/src/openai.rs:1563  (abridged to the spine; DFlash
// pair-swap branch elided, see file)
if shifted {
    // Do not start a potentially long staging prefill for a client that
    // disconnected while waiting for its serving-slot lease.
    measured_emit("", None, None)?;
    let batch = prepared_prefill.materialize(runtime)?;
    let mut staging = match runtime.staging.try_lock() { ... };
    staging.reset();
    let prepared = staging
        .prefill(batch)
        .map(|_| ())
        .map_err(|_| accelerator_failure(runtime));
    swap_staging_on_success(session, &mut staging, prepared)?;
    // The old serving generation is now the hidden owner. Empty it only
    // after the infallible ownership swap; no failure path can have touched
    // the live session that was committed before this rebuild.
    staging.reset();
}
```

`swap_staging_on_success` is four lines — `prepared?; std::mem::swap(live,
staging)` (`openai.rs:2790-2798`) — and the pair variant swaps target and
DFlash states together (`:2800-2811`). Publication is a pointer swap, so it
cannot fail halfway; the comment above states the invariant. A busy staging
lock is `Overloaded`, a poisoned one latches the pool unhealthy and returns
`Unavailable` (`openai.rs:1569-1576`). Every shift advances a committed
`context_epoch` (`openai.rs:1547-1552`), and later continuation requests
must validate their lineage against the stored replay plan — the retained
turns must appear "as one exact ordered run under identical leading system
content" (`openai.rs:5408-5413`).

Why rebuild instead of truncating the live cache in place? The retained set
is a prefix (system) plus a suffix (newest turns) with a hole in the middle:
every token after the hole changes logical position, and on the 39 RoPE
layers a changed position means changed key bytes ([Ch 14](14-qk-norm-and-rope.md))
— the middle cannot simply be deleted. A fresh generation computes the
retained context at its true positions, and the atomic swap makes the
replacement all-or-nothing. The staging prefill is real work at real depth —
which is precisely the cost [Ch 25](25-warm-reuse.md)'s reuse ladder exists
to skip when the prefix is *not* holed.

## 23.8 Tradeoffs

**Explicit origins vs `position % capacity`.** The ancestor indexed KV by
absolute position with the ring modulus "unwired/stubbed — a named OOB
hazard muser fixed from day one" (`docs/extraction-manifest.md`, per
[Ch 15](15-kv-store-and-the-ring.md)). Muser's two origin fields make the
ring's rotation an explicit, checkpointable fact: restore can reproduce it
(§23.4), speculative rollback can retain it (§23.6), and the route ladder
can test it (below). The measured consequence of *not* having it is the
ancestor's hazard record; the measured consequence of having it is the
bitwise replay test named at `decode.rs:380`.

**The compact ring vs the pinned kernel's addressing.** The route predicate
for llama's pinned SWA vec kernel accepts only rings the kernel can read
safely:

```rust
// crates/muser-engine/src/decode.rs:5646
let llama_vec_rows = (strict_attention || self.kernels.has_llama_flash_attn_vec())
    && plane.len > 0
    // The pinned vec kernel rounds KV reads to a 32-row block.
    // A deliberately tiny raw session can have a smaller backing
    // allocation, so taking the vec path would read past it and
    // poison the full distribution with NaNs.
    && plane.capacity >= 32
    && (plane.origin_physical == 0 || plane.len == plane.capacity);
// Token-major SWA cannot use llama's pad kernel (nb11 is a full
// token row, not one head). Only take that path when the window
// is a multiple of 32 so the vec kernel never pads.
let llama_swa = llama_vec_rows && plane.len.is_multiple_of(32);
```

A ring is vec-eligible while filling (`origin_physical == 0`) and again at
steady state (`len == capacity`) — every wrapped steady-state decode token
qualifies. The odd states (a restored ring sitting mid-rotation below
capacity) fall back to the split-K or ferrite interleaved kernels of
[Ch 16](16-attention-decode-kernels.md). And when the one-row decode path
*does* use the llama kernel against compact ring rows, Muser stages the
ring into llama's absolute 256-row-padded indices first — "so the pinned
vec kernel sees the same reduction lanes rather than a mathematically
equivalent compact permutation" (`metal/encode/attn.rs:140-144`): the
staging copy buys the comparator's exact reduction order. Exactness beats
the compact layout, again at a counted cost (§23.3).

**Server-owned shift vs engine-owned eviction.** Pushing the policy up
means the engine's cache code has exactly one writer discipline (append
continuity) and the *semantic* decisions — what a turn is, what the system
prologue means, how much output reserve to protect — live where the request
model lives. The price is that a shift is a full re-prefill of the retained
context through staging rather than a surgical cache edit; the mitigation
is not cleverer eviction but the reuse ladder of
[Ch 25](25-warm-reuse.md), which makes the un-holed case free. No retained
measurement isolates the staging re-prefill's wall cost from the request it
serves **[unverified]** — it is bounded above by the cold prefill numbers of
[Ch 25](25-warm-reuse.md) at the retained depth.

**Where the gap lives.** Two rows of the +196 accounting live in this
chapter's machinery: the 39 SWA wrapped-ring staging groups (§23.3) and the
52 KV-publication splits ([Ch 15 §15.9](15-kv-store-and-the-ring.md)). Both
are classed "session/publication structure — Keep" by the gap note itself
`[docs/decode-dispatch-gap-20260815.md]`; this chapter is the concrete
machinery those labels were pasted onto.

## 23.9 What comes next

The ring and the growing plane are now complete stories: reserved
fail-closed, staged across wraps, snapshotted to logical order, restored at
the exact rotation, rolled back by metadata or by retained rows, and
rebuilt wholesale by a server policy that swaps generations atomically. But
everything so far keeps the cache *inside one process on one machine*. The
interchange snapshot of §23.4 already hinted at the next move — a
fail-closed, shape-checked, layout-stable byte contract that something
outside the engine can hold. That something is kvpack: a vendored,
provenance-pinned format that turns a prefill into a durable, portable,
authenticated asset. The format is
[Ch 24](24-kvpack-the-format.md).

## References

- `crates/muser-engine/src/decode.rs:117-122` — `CacheDiscontinuity`.
- `crates/muser-engine/src/decode.rs:263-314` — `append` / `append_batch`
  (`append_batch` quoted; §23.2's walk).
- `crates/muser-engine/src/decode.rs:208-226, 373-434, 1386-1496` — the
  speculative checkpoint: metadata rewind vs retained rows.
- `crates/muser-engine/src/decode.rs:322-349` — the snapshot walk (quoted);
  `:351-417` `detached_from` and the rotation-preserving install;
  `:376-383` the order-sensitivity comment and the named replay test.
- `crates/muser-engine/src/decode.rs:4290-4352` — the wrapped SWA staging
  route and its `append_batch` commit; `:4356` the attend-before-overwrite
  comment.
- `crates/muser-engine/src/decode.rs:5643-5657` — the route predicates that
  test ring state (quoted).
- `crates/muser-engine/src/metal/encode/attn.rs:103-138` —
  `encode_stage_swa_prefill_f16` (signature quoted); `:140-144` the
  llama-padded-index staging and its exactness rationale.
- `crates/muser-engine/src/shaders/muse_reference.metal:1224-1228, 1240` —
  the store/scatter indices and the staging kernel.
- `crates/muser-engine/src/cache.rs:13-17, 39-47, 62-118, 205-222` — the
  interchange: encodings, the SWA-tail/NoPE-full cut contract, the
  fail-closed shape gate, `write_f16_tile`'s head-major scatter.
- `crates/muser-engine/src/prefill.rs:15-17` — placement-never-from-position.
- `crates/muser-server/src/state.rs:215-219` — `ContextPolicy` (quoted);
  `:240-243` the out-of-pool staging generation.
- `crates/muser-server/src/openai.rs:5241-5317` —
  `prepare_with_context_policy` (the turn-dropping loop);
  `:5338-5354` `compact_raw_prompt`; `:5356-5391` `complete_chat_turns` /
  `shift_chat_units`; `:5408-5413` lineage validation; `:1547-1552` the
  context epoch; `:1563-1637` the staging rebuild (spine quoted);
  `:2790-2811` the swap helpers.
- `crates/muser-kvpack/src/session.rs:164-166` — `ProductionF16Required`
  (CPU/Metal snapshot mutual uninstallability, with
  `[docs/kvpack-merge-handoff §4]`).
- `[docs/muser-architecture.md]` — §Context and sessions (shift semantics
  quoted), §Slots and scheduling (staging is not a fifth slot).
- `[docs/decode-dispatch-gap-20260815.md]` — the 39 staging groups and 52
  publication splits rows.
- `[docs/extraction-manifest.md]` — the ancestor's ring-modulus hazard and
  Muser's fix (via [Ch 15](15-kv-store-and-the-ring.md)).
- [Ch 15](15-kv-store-and-the-ring.md) — planes, layouts, store kernels,
  single-token `append` (this chapter's ancestors).
- [Ch 16](16-attention-decode-kernels.md) — the read-side route ladder.
- [Ch 24](24-kvpack-the-format.md), [Ch 25](25-warm-reuse.md) — the
  portable format and the reuse ladder that staging re-prefill motivates.
- `[ferrite-book Ch 14]` — the ancestor's paged cache, kept as contrast
  ([Ch 22 §22.8](22-the-price-of-context.md)).
