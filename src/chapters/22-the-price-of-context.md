# Chapter 22 — The price of context
> **status:** polished  ·  **path:** Muse Glimmer, pinned Muser tree

*Prerequisites: [Ch 15](15-kv-store-and-the-ring.md) (the KV planes, the
ring, the two layouts — this chapter costed nothing there on purpose),
[Ch 9](09-muse-glimmer-architecture.md) (the 39/13 layer split, GQA 32:2),
[Ch 1](01-why-inference-is-a-memory-problem.md) (bytes-per-token as the
organizing number). This is a systems chapter: no new kernels, one bill.*

---

## 22.1 What this chapter computes

[Ch 21](21-sampling-argmax-and-grammar.md) closed the kernel walk and then
pointed at a debt: every attention chapter from [Ch 14](14-qk-norm-and-rope.md)
onward borrowed memory at "one thousand twenty-four bytes per token per
layer" ([Ch 15](15-kv-store-and-the-ring.md)) and we never once asked what
the loan costs. This part of the book is the repayment schedule. The next
five chapters cover what the [KV cache](../glossary.md#kv-cache) costs (this
chapter), how the two storage regimes actually work
([Ch 23](23-the-swa-ring-and-the-growing-cache.md)), how the cache becomes a
portable, sealed artifact ([Ch 24](24-kvpack-the-format.md)), what a hit is
worth ([Ch 25](25-warm-reuse.md)), and how to move only the part you don't
already have ([Ch 26](26-delta-handoff-and-migration.md)).

This chapter derives every number by hand, because the numbers are short
enough to derive and because trusting a memory table you cannot re-derive is
how a wrong figure survives into product copy. (The campaign has already
caught one: an early "~7 GB payload" estimate for the deep handoff was wrong
by ~4× — the measured wire payload is 1,823,184,896 B
`[receipt phase4-disagg-20260820/130815-g900091/]`. Derive, then measure,
then reconcile.)

## 22.2 The per-token bill, one layer at a time

Start from the model's geometry, which is not a convention but a pinned
constant of this engine:

```rust
// crates/muser-engine/src/config.rs:13
pub const MUSE_LAYER_COUNT: usize = 52;
pub const MUSE_SWA_LAYER_COUNT: usize = 39;
pub const MUSE_NOPE_LAYER_COUNT: usize = 13;
pub const MUSE_SWA_WINDOW: usize = 2_048;
pub const MUSE_MAX_CONTEXT: usize = 131_072;
pub const MUSE_HEAD_COUNT: usize = 32;
pub const MUSE_KV_HEAD_COUNT: usize = 2;
pub const MUSE_HEAD_DIM: usize = 128;
pub const MUSE_KV_ROW_ELEMENTS: usize = MUSE_KV_HEAD_COUNT * MUSE_HEAD_DIM;
```

[Ch 13](13-the-qkv-gate-matvec-family.md) projected 32 query heads but only
**2 KV heads** — that is [GQA](../glossary.md#gqa-grouped-query-attention), the grouping that makes the
cache cheap before any cleverness is applied. One cached row of one layer
holds, per token:

```
  K row, one layer, one token:
      n_kv_heads × head_dim × bytes_per_element
    = 2           × 128       × 2            (f16, every Metal lane)
    = 512 bytes

  K + V together (two separate f16 buffers — decode.rs:182-190):
    = 2 × 512 = 1,024 bytes per layer per token
```

That is the doc's own formula, `2 KV heads * 128 values * 2 bytes * (K + V)
= 1,024 bytes` `[docs/memory-footprint.md §KV formula]`, rebuilt from the
constants. The element type is [f16](../glossary.md#f16) on every Metal lane —
the `GpuHalfBuffer` field type is the constraint
(`crates/muser-engine/src/decode.rs:182-184`) — because the parity anchor,
pinned llama.cpp, runs F16 KV `[docs/muser-architecture.md]`. Whole model:

```
  per token, all 52 layers:  52 × 1,024 = 53,248 B  (≈ 52 KiB)
```

Here is the part that trips people up: **both layer classes pay the same
per-token price.** A sliding layer's row costs exactly what a full layer's
row costs — 1,024 B. The 39/13 split does not change the price of a token.
It changes *how many tokens keep paying.*

## 22.3 Two regimes: one curve flattens, one doesn't

Same price per token, different stopping points — so total footprint is a
function of how far each layer class keeps paying. For a slot configured
with context `C`, the footprint formula is
`[docs/memory-footprint.md §KV formula]`:

```text
swa_rows  = min(C, 2,048)
nope_rows = C
slot_kv_bytes = (39 * swa_rows + 13 * nope_rows) * 1,024
```

Why per class:

- The **39 SWA layers** attend only to the trailing 2,048 tokens
  (`layer % 4 != 3`, `config.rs:84-93`), so their planes are allocated once
  at `min(max_context, sliding_window)` capacity
  (`decode.rs:1346-1347`) and never grow. Every token past 2,048 *overwrites*
  a row ([Ch 15](15-kv-store-and-the-ring.md)'s ring) instead of adding one.
- The **13 NoPE layers** attend to the entire history, so their planes are
  allocated at `max_context` capacity (`decode.rs:1348`) and fill one row per
  token until the model limit.

Draw the two components as functions of context:

```
  KV bytes
  1.83 GB ┤                                          NoPE: 13×C×1,024
          │                                     ▄▄
          │                                 ▄▄▄▄
          │                             ▄▄▄▄          ← grows linearly,
          │                         ▄▄▄▄               never flattens
          │                    ▄▄▄▄
  81.8 MB ┤▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄  SWA: 39×2,048×1,024
          │  ↑ flat from C = 2,048 onward              (constant after wrap)
          └──┬──────┬──────────┬──────────┬──────→ context C
           2,048  32,768    65,536    131,072
```

*Figure 22.1: the two growth regimes. At full depth the NoPE term is
1,744,830,464 B and the SWA term is 81,788,928 B — 95.5 % of a slot's KV
bytes live in the 13 full layers.* The curve's shape has a direct
consequence for cost accounting, and it is the amortization insight that
makes long context survivable at all:

```
  marginal KV cost of one more token:
      below 2,048:   52 × 1,024 = 53,248 B/token
      above 2,048:   13 × 1,024 = 13,312 B/token   (exactly 13/52 = 25 %)

  kv(131,072) / kv(2,048)
    = 1,826,619,392 / 109,051,904
    = 16.75×   — not the naive 131,072/2,048 = 64×
```

Thirty-nine of fifty-two layers stop billing at token 2,048. A 64× longer
context costs 16.75× more KV — the architecture quietly discounted
three-quarters of the cache. Keep that 13,312 B/token figure; it returns as
the exact NoPE-per-token payload unit in [Ch 26](26-delta-handoff-and-migration.md).

One measurement-grade cross-check that the two derivations agree. Below the
window, the per-class formula and the whole-model formula must coincide:

```
  (39 + 13) × 2,048 × 1,024 = 109,051,904 B
  52 × 2,048 × 1,024         = 109,051,904 B   ✓
```

Two routes, one number — the habit the ancestor book taught with its
block-form versus element-form footprint derivations `[ferrite-book Ch 14]`,
kept here because it catches transcription errors no single formula can.

## 22.4 The footprint table, derived and cross-checked

Now produce the table the release contract actually uses — one slot and the
four-slot release configuration, at three depths `[docs/memory-footprint.md]`:

```
  C = 8,192:
      slot = (39×2,048 + 13×8,192) × 1,024
           = (79,872 + 106,496) × 1,024 = 190,840,832 B = 0.191 GB
      ×4 slots = 763,363,328 B = 0.763 GB

  C = 32,768:
      slot = (79,872 + 425,984) × 1,024 = 517,996,544 B = 0.518 GB
      ×4 slots = 2,071,986,176 B = 2.072 GB

  C = 131,072:
      slot = (79,872 + 1,703,936) × 1,024 = 1,826,619,392 B = 1.827 GB
      ×4 slots = 7,306,477,568 B = 7.306 GB
```

| Context per slot | One-slot KV | Four-slot KV |
|---:|---:|---:|
| 8,192 | 0.191 GB | 0.763 GB |
| 32,768 | 0.518 GB | 2.072 GB |
| 131,072 | 1.827 GB | 7.306 GB |

*Table 22.1: decimal GB, KV planes only — my arithmetic lands on the
document's values to the byte `[docs/memory-footprint.md]`.* These are
topology-derived allocation numbers, not a measured peak RSS, and the doc
says so on its first line: it is "a topology-derived allocation estimate,
not a measured peak-RSS result and not launch guidance for unqualified
Macs" `[docs/memory-footprint.md]`.

Two honesty notes the same document insists on, and this book inherits:

1. **KV planes are not the process.** The target GGUF is 16,756,681,056 B on
   disk (`crates/muser-engine/src/lib.rs:13-14`), the DFlash draft GGUF
   1,631,205,312 B and the vision projector 1,400,328,928 B — the latter two
   load only when configured `[docs/memory-footprint.md §Other material
   allocations]`. Add shared Metal pipelines and workspaces, per-slot logits
   and sampler state, prefill chunk buffers (~0.99 GB of f32 batch-activation
   widths, reused, chunked at 512 positions — "must not be labeled peak
   RSS"), network buffers, and temporary restore/migration material.
   "Summing artifact sizes with the KV formula is therefore only a lower
   bound, not a safe RAM recommendation" `[docs/memory-footprint.md]`.
2. **A staging generation exists.** Restore and context shift build a
   replacement state before swapping it in, so an operation can temporarily
   require additional state — but staging "is not a fifth concurrently
   serving slot" `[docs/muser-architecture.md §Slots and scheduling]`
   (`state.rs:240-243`; the full mechanism is
   [Ch 23](23-the-swa-ring-and-the-growing-cache.md)).

## 22.5 The 131,072-position limit

The horizontal axis of Figure 22.1 stops at 131,072 because the model stops
there: `MUSE_MAX_CONTEXT: usize = 131_072` (`config.rs:17`), and the
architecture doc states the serving rule — "The model limit is 131,072
positions per slot" `[docs/muser-architecture.md §Context and sessions]`.
The NoPE planes are allocated at exactly that capacity
(`decode.rs:1348`), which is why the one-slot figure above is a ceiling, not
a trajectory that keeps going.

Two practical footnotes. First, the deep campaign cells you will meet in
[Ch 25](25-warm-reuse.md) and [Ch 26](26-delta-handoff-and-migration.md) use
fixtures at 130,815 and 131,008 tokens — inside the limit, not at it; depth
labels on one claim are never the depths of another (the measured-numbers
ledger's rule 4). Second, the limit is per slot, so four concurrent slots
each get their own 131,072 positions — the table's 7.306 GB row is that
configuration's KV bill `[docs/memory-footprint.md]`.

## 22.6 Why KV — not weights — decides how many slots a 96 GB Mac serves

With the per-slot bill and the 131,072 ceiling on the table, the capacity
question can be asked. The decode host is one Apple Silicon Mac, an
**M3 Ultra with 96 GB** of
unified memory `[docs/memory-footprint.md §intro; docs/release-provenance.md]`.
Ask the capacity question: how many concurrent request slots can it serve?
The answer's shape surprises people who arrived from a weights-first
intuition, so build it in two steps.

**Weights are paid once and shared.** All slots run the same pinned model.
The 16,756,681,056 B GGUF is mmap'd once; immutable weights, Metal
pipelines, and the DFlash executor are explicitly shared across slots
`[docs/muser-architecture.md §Slots and scheduling]` (`state.rs:221-236`).
Going from one slot to four does not buy four copies of the model.

**KV is paid per slot, forever, at the depth the conversation reaches.**
Each slot owns "independent target KV, DFlash state, logits, RNG, sampler
and grammar state" `[docs/muser-architecture.md §Slots and scheduling]`.
Table 22.1 is that cost: the only term that scales with *context depth* is
the slot's KV. At full depth, four slots add 7.306 GB on top of the shared
artifacts — and the deeper each conversation runs, the larger that term
grows, with the NoPE planes doing essentially all of the growing (Figure
22.1). Serving capacity at depth is therefore KV-bound: the weights fit or
they don't (a question answered once, at load), while the number of
*long-context* conversations you can host is a question about 1,024 B per
layer per token per slot.

Two boundaries keep this honest rather than purely arithmetic:

- **The slot count is also a design constant, not just a memory outcome.**
  The packed decode graph accepts 1..=4 sequences
  (`forward_decode_group`, `decode.rs:4874`); the server enforces
  `--parallel` in 1..=4 (`state.rs:1054-1056`). Four is the release
  configuration, and the memory contract is stated as such: "The v0.1
  release contract is four full-context slots on the 96 GB M3 Ultra. No
  smaller-memory configuration may be advertised as supported until a
  retained hardware qualification measures it" `[docs/memory-footprint.md
  §Release requirement]`.
- **The final serving benchmark must retain process and system memory
  evidence** for the exact four-slot binary, artifacts, context cell, and
  concurrency before any of this becomes a product claim
  `[docs/memory-footprint.md §Release requirement]`. Until that matrix
  passes, this chapter supports engineering capacity checks only.

Put as a ratio, the two currencies this book tracks: one decode token reads
≈16.76 GB of weights (`[Ch 1](01-why-inference-is-a-memory-problem.md)`,
artifact size above) and writes 53,248 B of KV below the window, 13,312 B
above it — a factor of ~315,000 between the per-token weight read and the
per-token KV write. Weights dominate the *speed* of a token
([Ch 1](01-why-inference-is-a-memory-problem.md)); KV dominates the
*capacity* of the machine. The whole of Part V is about spending the second
currency well.

## 22.7 Where the bytes go at depth: the 95.5/4.5 split

The same per-class arithmetic, applied to the wire instead of the slot,
predicts what a deep handoff must carry. The measured 130,815-token
disaggregated payload is **1,823,184,896 B** (`payload_bytes` verified in
the client receipt `[receipt phase4-disagg-20260820/130815-g900091/]`), and
it reconciles to the byte against this chapter's formula:

```
  NoPE:  130,814 rows × 13 layers × 1,024 B = 130,814 × 13,312 = 1,741,395,968 B
  SWA:   3 groups × 2,048 rows × 13,312 B                         =    81,788,928 B
  total                                                      = 1,823,184,896 B  ✓
```

(Two details you will meet again: the NoPE row count is 130,814, not
130,815, because the receiver deliberately holds back the boundary token and
decodes it locally; and the SWA rings travel as three 13-layer groups — the
transfer schedule of `[Ch 24](24-kvpack-the-format.md)`.) By these terms,
**95.5 % of a deep payload is the 13 NoPE planes and 4.5 % is the 39 SWA
rings** `[docs/kvpack-merge-handoff §3 D1, §6]`. This is not incidental: it
is the load-bearing fact that the NoPE layers are *position-free*
([Ch 14](14-qk-norm-and-rope.md) — no RoPE, so their rows are relocatable
bytes: "relocate = memcpy — the whole kvpack free lunch",
`crates/muser-engine/src/lib.rs:8-10`), and it is why the entire Part VI
wire economy is really a NoPE-plane economy.

For the record, the wrong early figure — "~7 GB" for this same payload —
came from reading the producer's `--kv-cache-memory-bytes` *allocation*
(7–8 GB of vLLM cache memory) as if it were the payload. The correction is
recorded in the merge-handoff audit `[docs/kvpack-merge-handoff §3 D1]` and
in the campaign ledger's landmine list. Allocation is not traffic; traffic
is not allocation.

## 22.8 Tradeoffs

**Why f16 KV and not quantized, at 4 bits or 8.** Halving (or quartering)
Table 22.1 looks tempting at 7.306 GB per four slots. It is not taken,
and the reason is the exactness contract, not conservatism-as-a-virtue: the
parity anchor is pinned llama.cpp running F16 KV, so the live planes are
`GpuHalfBuffer` f16 on every Metal lane (`decode.rs:182-184`), and the lane
table's decode rows all say "FP16 KV" `[docs/muser-architecture.md]`. A
quantized live cache would change attention inputs and therefore logits —
the same class of change the dispatch-gap campaign rejected when a norm
fusion moved logprobs 3.197e-4 over a 1e-4 contract
`[docs/decode-dispatch-gap-20260815.md]`. The interchange format does carry
both encodings (`PlaneEncoding::{F16Le, F32Le}`,
`crates/muser-engine/src/cache.rs:13-16`) because the wire serves producers
and archives beyond the live planes; a KIVI-style 4-bit store exists inside
kvpack as a CPU reference codec with honest error bounds, "not the hot
path" `[docs/kvpack-merge-handoff §5]`. The measured consequence of staying
f16 is everything in [Ch 25](25-warm-reuse.md): bit-identical warm hits.

**Ring-plus-growing vs paging.** The ancestor's cache was paged — 16-token
blocks behind a block table, a whole allocator chapter
`[ferrite-book Ch 14]`. Muser has no page table, and the arithmetic of this
chapter explains why the design didn't need one: the 39 SWA layers bound
themselves (a full ring *is* the live set; nothing to evict — 81.8 MB,
Figure 22.1's flat line), and the 13 NoPE layers' prefix sharing is handled
one level *above* the GPU, by kvpack's content-addressed chunks
([Ch 24](24-kvpack-the-format.md)), not by in-GPU indirection. Paging bought
the ancestor fragmentation control it needed at 8 GB; at 96 GB with a
bounded 75 % of layers, the simpler structure wins on review surface —
which is a bet, not a measurement, and it is labeled as one. What *was*
measured is the per-class byte split this chapter derives
(`[receipt phase4-disagg-20260820/130815-g900091/]`), and any future format
change must re-derive §22.7's payload decomposition before predicting wire
behavior.

**The "grows with context" correction.** The ancestor chapter's motivating
sentence — the KV cache is *the* structure that grows with context — is
true for 13 of Muser's 52 layers and false for the other 39
`[ferrite-book Ch 14]` (the port audit's flagged hazard #1). Porting that
sentence uncorrected would misprice everything: capacity planning
(§22.6), wire payloads (§22.7), and the warm-up economics of
[Ch 25](25-warm-reuse.md) all turn on *which* layers grow. This chapter's
standing instruction: any KV statement must name its layer class.

**Where the gap lives.** This chapter is not the Metal gap — nothing here
is a dispatch. If anything, it is the gap's opposite: the KV-publication
splits and SWA staging groups that [Ch 15 §15.9](15-kv-store-and-the-ring.md)
counted inside the +196-closure accounting exist *because* the store is
structured for exactness-preserving publication, and this chapter is the
bill that explains why nobody cheapened them.

## 22.9 What comes next

The bill is derived, cross-checked, and split by layer class: 1,024 B per
layer per token; 39 layers stop at 2,048 rows, 13 grow to the model limit;
a full-depth slot costs 1.827 GB and four of them 7.306 GB on a 96 GB
machine whose weights are paid once. But the arithmetic treated the two
regimes as curves on a chart. The chart is implemented — as a ring whose
write pointer, logical origin, and physical origin advance together across
a wrap, next to a growing plane whose per-head spans must stay contiguous,
plus a server policy that decides what to *do* when context exceeds the
limit. That machinery — and the reason a restored ring must keep its
rotation to replay bitwise — is
[Ch 23](23-the-swa-ring-and-the-growing-cache.md).

## References

- `crates/muser-engine/src/config.rs:13-21` — the pinned geometry constants
  (quoted); `:84-93` the `layer % 4 == 3` partition rule.
- `crates/muser-engine/src/decode.rs:182-190` — `MetalKvPlane` (f16 fields);
  `:1338-1358` per-layer-kind capacity allocation (`min(max_context,
  sliding_window)` vs `max_context`).
- `crates/muser-engine/src/lib.rs:7-14` — model facts: 39/13 split, GQA
  32:2, head_dim 128, the 16,756,681,056 B artifact, NoPE
  relocate-as-memcpy.
- `crates/muser-engine/src/cache.rs:13-16` — `PlaneEncoding::{F16Le, F32Le}`
  (interchange encodings, not live planes).
- `crates/muser-server/src/state.rs:221-243` — slot independence vs shared
  weights; the out-of-pool `staging` generation.
- `crates/muser-server/src/state.rs:1054-1056`, `decode.rs:4874` — the
  1..=4 slot bound.
- `[docs/memory-footprint.md]` — the 1,024 B row formula, the one-slot /
  four-slot table (§22.4's cross-check), other material allocations, the
  96 GB M3 Ultra release contract, and the lower-bound honesty rule.
- `[docs/muser-architecture.md]` — §Slots and scheduling (shared weights,
  staging), §Context and sessions (the 131,072 per-slot limit), the FP16 KV
  lane table.
- `[receipt phase4-disagg-20260820/130815-g900091/]` — the 1,823,184,896 B
  deep payload (`payload_bytes` verified); decomposition per
  `[docs/kvpack-merge-handoff §3 D1, §6]`.
- `[docs/decode-dispatch-gap-20260815.md]` — the exactness-contract
  precedent cited in §22.8.
- [Ch 15](15-kv-store-and-the-ring.md) — the per-layer-class split this
  chapter costed; owns the store kernels and layouts.
- [Ch 23](23-the-swa-ring-and-the-growing-cache.md) — the ring machinery,
  the growing plane, context-shift policy.
- `[ferrite-book Ch 14]` — the ancestor's paged-Q8 cache: the dual-derivation
  device (ported in §22.3) and the "grows with context" framing (corrected
  in §22.8).
