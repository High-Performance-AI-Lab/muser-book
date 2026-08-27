# Chapter 8 — The DFlash draft
> **status:** polished  ·  **path:** Muse Glimmer, pinned Muser tree
>
> *Prerequisites: Chapters 5–7. You know the kquant blocks the draft is made
> of, the lanes it may run on, and — from Chapter 7's Fallback B — the lane
> it may not. This chapter is about a model small enough to hold in your
> head, and about the contract that lets something *approximate* participate
> in *exact* inference.*

Chapter 7 ended with a refusal: native NVFP4 plus DFlash is rejected by
the receiver configuration, so speculative serving stays on the qualified
kquant lane. This chapter opens that last clause. [DFlash](../glossary.md#dflash) is Muser's
[draft model](../glossary.md#draft-model) — the small assistant that guesses
tokens cheaply so the 52-layer target only has to *confirm* them — and it
is a kquant artifact from end to end, down to the very blocks you unpacked
in Chapter 6.

One boundary statement before anything else. Speculative decoding has two
halves: *drafting* (this chapter) and *accept/verify* (the algorithm that
decides which guesses become output). The algorithm —
`verify_full_speculative_mt_ordered`, the carried-frontier state machine,
the rollback choreography — gets its full treatment in
[Ch 33](33-speculation-and-the-distributed-verdict.md). Here we build the
draft, load it, condition it, and pin down what it must guarantee so that
Chapter 33's verification can stay exact.

---

## 8.1 Why a draft model at all

Chapter 1's fourth lever: move the work somewhere cheaper. Decode is
bandwidth-bound because each token reads the whole 16.76 GB model for one
token's worth of math. Speculation attacks the *denominator*: if a cheap
model proposes k tokens and the expensive target verifies them in **one
batched forward pass**, then each accepted proposal cost a fraction of a
full target read. The target still does all the deciding — that is what
keeps the output exact — but it does k+1 rows of decision per read
instead of one. The draft is pure overhead that pays for itself only when
its guesses are good; the entire engineering of this chapter is making the
guesses good and the overhead small.

The measured stakes, stated with the campaign's own scope language: in
retained fixed-window synthetic packets, kquant DFlash decode ratios
(llama ÷ muser means) are **1.23692× at 2,048, 1.20323× at 16,384, and
1.19616× at 32,768 tokens, with 5/5 exact-token reps per depth**
`[claims #15]`. Hold onto both halves of that sentence — the ratios *and*
the scope. We return to them in §8.6.

## 8.2 What DFlash is

The module doc says it in five lines:

```rust
// crates/muser-engine/src/dflash.rs:1
//! Five-layer DFlash assistant extracted from Ferrite's accepted CPU oracle.
//!
//! The target hook is Muse-specific, but the assistant math and artifact
//! format supports both the development SafeTensors export and the official
//! llama.cpp-compatible k-quant GGUF sidecar.
```

Concretely, from the config contract (`DFlashConfig`,
`[crates/muser-engine/src/dflash/config.rs:69-98]`) and the bench shape
table Chapter 6 quoted:

- **Five layers**, enforced: `validate` refuses anything else — "release
  assistant must have exactly 5 layers" (`config.rs:263-268`).
- **It reads the target's mind.** This is DFlash's defining trick and the
  reason it can be so small: it does not re-derive the target's thinking
  from raw tokens. It is *fed* the target's hidden states from five pinned
  target layers (`DFlashConfig.dflash_config.target_layer_ids`,
  `config.rs:66`), captured by the engine at
  `DFlashHiddenCache::write_rows`, which accepts rows **only** for layers
  in that list (`[crates/muser-engine/src/dflash/hidden.rs:44-52]`).
- **Its front door is `fc.weight`**, shape 33,280 → 6,656
  (`[crates/muser-bench/src/m16.rs:219-225]`). Do the arithmetic and the
  architecture falls out: 33,280 = 5 × 6,656 — five sampled target hidden
  states (the target's hidden width, Chapter 6) concatenated, projected
  into the draft's 6,656-wide hidden stream.
- **Its own attention is small GQA**: k/v projections are 6,656 → 1,024
  (eight KV heads × head-dim 128 — the same per-token element count the
  context-geometry test uses, `config.rs:380-386`), with QK-norms and RoPE
  like an ordinary transformer block (`config.rs:310-335` lists the tensor
  names: q/k/v/o projections, q_norm/k_norm, SwiGLU MLP).
- **It proposes in the target's vocabulary by borrowing the target's own
  machinery**: `draft_greedy` embeds its 16-token block with
  `target.embed_tokens` and scores the draft's block output with
  `target.project_hidden` (`[crates/muser-engine/src/dflash/spec.rs:744-752]`)
  — the draft owns only its five layers, and the shared embedding/LM-head
  work is booked under assistant time in the telemetry
  (`spec.rs:62-64`). A proposal the target cannot score would be
  worthless, so the draft never needed a vocabulary of its own.
- **It is kquant**: `draft.k` is Q4_K, `draft.v` Q6_K, `draft.fc` Q4_K
  (Figure 6.6), and the GGUF loader *requires* every dense projection to
  be Q4_K/Q5_K/Q6_K (`[crates/muser-engine/src/dflash/weights.rs:148-158]`).
- **It is cheap to keep**: 1,631,205,312 B on disk
  (`[docs/memory-footprint.md]` artifact manifest), "loaded only when
  configured."

The draft runs **16-row blocks**: `block_size` defaults to 16
(`config.rs:138-140`), and drafting produces up to 15 proposals in one
block forward — which is exactly why the M=16 batch kernels of Chapter 6
exist.

## 8.3 The context ABI: a 64-row sink plus a trained window

The draft's attention does not see the full conversation. Its context is a
fixed ABI: the first 64 rows are pinned forever, and a trailing window
slides over the rest:

```rust
// crates/muser-engine/src/dflash/config.rs:59
/// The sink span used by the released DFlash cache ABI. It is made explicit
/// in every newly enrolled combined identity; it is not a receiver default.
pub const DFLASH_CONTEXT_SINK_SIZE: usize = 64;

// crates/muser-engine/src/dflash/config.rs:9
/// The context-cache shape bound to one enrolled DFlash identity.
///
/// The trained window comes from the sidecar metadata. The 64-row sink is
/// part of Muser's DFlash cache ABI rather than GGUF metadata, so enrollment
/// stamps it into both peers' identity configs explicitly. Receivers must
/// never infer either value from a local fallback.
pub struct DFlashContextGeometry {
    pub layers: usize,
    pub elements_per_token: usize,
    pub sink_size: usize,
    pub window_size: usize,
}
```

The sink-plus-window shape matters because the draft conditions on
*target-derived* context rows. Figure 8.1 draws the ABI:

```
 DFlash context — one row per committed token, per layer (5 layers, f32)

 position:  0 …………………… 63 │ 64 …………………………………… (64 + W − 1)
            ┌───────────────┐ ┌────────────────────────────────┐
            │     SINK      │ │          WINDOW (W)            │
            │  pinned rows  │ │   trailing rows, slides as the │
            │  never evicted│ │   conversation grows           │
            └───────────────┘ └────────────────────────────────┘
             sink_size = 64       window_size = trained sliding_window
                                   (2,048 on the release sidecar)

 rows the ABI may ever hold, per layer: sink + window = 64 + 2,048 = 2,112
 buffered bytes (both planes, all layers):
   5 × 2 × 2,112 × 1,024 × 4 B = 86,507,520 B
```
*Figure 8.1: The draft's context cache. The sink is ABI (64, stamped at
enrollment); the window comes from `dflash.attention.sliding_window` in the
sidecar metadata. The byte bound is identity-derived, not a constant
(`DFlashContextGeometry::buffered_byte_limit`,
`[crates/muser-engine/src/dflash/config.rs:42-56]`).*

The geometry declares how many buffered rows the ABI may ever hold, with
that exact byte bound computed from the identity itself — for the release
geometry (5 layers, 1,024 elements per token, sink 64, window 2,048) the
arithmetic above yields 86,507,520 B, asserted by the config's own test
(`config.rs:380-395`).

That doc comment's last line — "Receivers must never infer either value
from a local fallback" — is fail-closed culture in one sentence: a remote
handoff that stamped one geometry must never be served by a receiver that
quietly assumed another. The same struct is bound into the cluster config
alongside the draft's
SHA-256 identity (`[crates/muser-cluster/src/config.rs:44-49]`).

## 8.4 The window bug — read this twice

Where does `window_size` come from? The sidecar metadata
(`dflash.attention.sliding_window`), and the code documents what happens
when you *don't* read it:

```rust
// crates/muser-engine/src/dflash/config.rs:86
/// Trained sliding-window span for the draft's context attention, from
/// `dflash.attention.sliding_window`. The draft must be conditioned on
/// exactly this many trailing target rows: measured 2026-08-21, feeding
/// it half (the previously hardcoded 1024) or far more (32768) collapses
/// natural-text acceptance from 72.5% to 2.2%.
```

For an entire campaign, Muser hardcoded sink 64 + window 1,024 and never
read the key; the sidecar was trained at 2,048. The draft ran on **half
its trained window**, and — this is the part to sit with — *the
synthetic fixtures kept passing*, because a period-8 synthetic stream is
predictable from token identity alone and cannot detect a conditioning
defect `[ledger §ROOT CAUSE FOUND AND FIXED]`. The fix read the real
window, took natural-text acceptance from 1.1% to 72.7% on the python
suffix-8,192 cell, made natural-text cells token-exact, and *lowered*
every synthetic spec number by ~5% — the draft had suddenly started doing
real work `[ledger §Spec re-measurement at the fixed window]`.

Two durable consequences landed in code. First, a sidecar without the key
still loads, but **loudly**: `resolve_sliding_window` prints a warning
that "draft conditioning may be wrong" rather than silently defaulting
(`config.rs:109-125`). Second, the effective geometry now travels with
every result — `DFlashSpecStats` carries `draft_sink_size` and
`draft_sliding_window` "so every receipt self-identifies. Conditioning
the draft on the wrong window silently invalidated a whole campaign's
spec numbers (2026-08-21)"
(`[crates/muser-engine/src/dflash/spec.rs:44-50]`).

The general lesson is Chapter 5's error discipline at systems scale: a
draft that is conditioned wrongly does not crash and does not fail parity
on easy fixtures — it just gets quietly worse at the one job it has.

## 8.5 Loading: one validated loader, two artifacts

`DFlashWeights::load` dispatches on what you point it at — a *file* is the
GGUF sidecar, a *directory* is the SafeTensors development export
(`[crates/muser-engine/src/dflash/weights.rs:35-39]`). Both paths share
one config contract (`DFlashConfig::from_artifact`,
`config.rs:143-150`), and the GGUF path is strict about metadata: the
architecture must be `dflash`, the target-layer list arrives one-based
(llama.cpp converter convention) and is converted to zero-based with
rejection of invalid entries (`config.rs:200-215`), and the resolved
dtype is recorded as `"gguf-kquant"` (`config.rs:251`).

The production Metal path loads a **projection shell**: norm vectors are
expanded to f32, and *nothing else is* — every dense projection stays
mmap'd in its kquant representation for the GPU to consume directly:

```rust
// crates/muser-engine/src/dflash/weights.rs:42
/// Load the official assistant without expanding its projection matrices.
/// Norm vectors remain f32; all large matrices stay mmap'd in their GGUF
/// k-quant representation and are consumed directly by Metal.
```

Why "without expanding" is worth a doc comment: the f32 compatibility path
costs real memory — "Expanding the official 1.5 GiB k-quant assistant
into ~6 GiB of f32 matrices wastes startup time and resident memory"
(`weights.rs:55-59`). Chapter 6's block formats are not just a disk story;
they are what makes a resident draft affordable. The SafeTensors path
remains the development oracle — the CPU forward (`dflash/forward.rs`) is,
like the target's, the correctness spec the Metal drivers are checked
against.

## 8.6 What the draft must guarantee for exact verification

Speculation is only lossless if the *target* makes every decision. The
draft participates in exactness by guaranteeing four things.

**1. It proposes; it never decides.** Acceptance runs on the CPU against
**full target distributions** — every proposed token is scored against
the target's complete probability row by
`verify_full_speculative_mt_ordered`
(`[crates/muser-engine/src/sampling.rs:1033]`), using the Leviathan-style
rule `accept if rng ≤ min(p/q, 1)` with a residual-corrected resample on
rejection, all on the source-pinned MT19937 draw stream
(`sampling.rs:1051-1088`; the stream is deliberately isolated from the
generic RNG so "a `rand` algorithm or conversion change" cannot alter
tokens, `sampling.rs:1001-1007` — MT19937 being a reproducible random
generator Muser re-implements bit-for-bit to match llama.cpp's,
[Ch 21](21-sampling-argmax-and-grammar.md)). A draft can be *arbitrarily bad* and the
output distribution stays exact — badness only costs speed. That single
property is why the window bug (§8.4) was a performance catastrophe and
not a correctness one.

**2. Its rounds are deterministic and replayable.** Proposals are
submitted "in deterministic round order. Qualification compares this
trace exactly" (`DFlashSpecStats.draft_token_trace`,
`[crates/muser-engine/src/dflash/spec.rs:26-28]`). The remote qualifier
compares 256 greedy tokens plus every full target-logit row, with an
acceptance floor of 0.95 (`DFLASH_ACCEPTANCE_MINIMUM`,
`[crates/muser-bench/src/remote.rs:3-8, :33]`) — exactness as a gate,
not a hope.

**3. Its block size and verify lengths are pinned.** `draft_greedy`
builds a 16-token block (a seed token followed by 15 mask tokens),
forwards it once, and takes the argmax of rows 1..=verify_length
(the index of each row's largest score);
verify lengths are exactly **3, 7, or 15** — anything else is an error
before any GPU work (`[crates/muser-engine/src/dflash/spec.rs:730-741]`).
While tracing is enabled, the engine even checks the block head's
*sanity*: row 0 saw the real seed, and "a correctly conditioned block
head must reconstruct the seed there; if it does not, the conditioning
carries no usable signal" (`spec.rs:753-757`).

**4. Its context writes are transactional.** A speculation round may fail
and roll back, so the target's KV planes checkpoint before the block:
`MetalSpeculativeCheckpoint` rewinds NoPE-plane *metadata only*, while
SWA rings retain "the ≤16 rows a block may overwrite"
(`[crates/muser-engine/src/decode.rs:213-226]`). Sixteen is not a magic
number — it is the block size; a speculative block can touch at most one
window's worth of rows, and the checkpoint is sized to exactly that ABI.

And conditioning — the window story of §8.4 — is the fifth guarantee in
disguise: not exactness-breaking, but throughput-breaking, and therefore
gated (§8.7).

## 8.7 The measured scope — and its honest edges

Now the numbers, with their scope language intact.

**The synthetic matrix (current, post-window-fix).** "In retained
fixed-window synthetic packets, exact-token decode ratios (llama/Muser
means) are 1.23692× at 2,048, 1.20323× at 16,384, and 1.19616× at 32,768,
with 5/5 exact reps per depth" `[claims #15]` (receipts:
`muser-receipt://spec-prefill-fix-20260822/aggregate-a2/…`
and `…/respec2-deep-20260822/aggregate-a1/…`). The claims row's own
instruction is part of the claim: *"Never generalize this to natural
text, native NVFP4, or untested depths."* The deeper cells of the same
family: exact-token in 5/5 reps at all four depths including 131,008
`[claims #3]`, and the funded-fix 131,008/48 packet crossed end-to-end
**wall** parity for the first time at 1.02536× `[claims #16]`.

**The old bar, and why you must not use it as a result.** The campaign's
famous 107.9 tok/s (107.9136 median, ratio 1.3273 vs llama's 81.3047) was
measured **before** the 2026-08-21 draft-window fix; it survives in the
record only as the kquant spec *bar* that later lanes were judged against
`[ledger §L2 Stage B verdict]`. The pre-fix ratios 1.3273/1.3012 are
superseded numbers — quoting them as current performance is one of this
book's standing landmines.

**Natural text is a different regime.** On real corpora, cross-engine
outputs diverge (so speed stands without an exactness gate), and the
picture splits: spec decode *wins* python-like content (16,384: 1.186;
8,192 suffix: 1.321) and **loses** high-acceptance shallow text (rust at
2,048: 0.931, improving only to 0.945 at verify-length 7) — llama's
lighter draft wins there `[docs/benchmarks.md §2]`. That asymmetry froze
the serving verify-length at 7 while the comparison harness pins 15:
"the best decode and the most robust acceptance on natural text"
`[docs/benchmarks.md §2]`.

**The engine distrusts its own draft.** Speculation can be disabled
per-request when it stops paying. The gate reads only *recent* evidence —
eight rounds, after a two-round warmup ("the rounds immediately after
prefill are the coldest of the request"), requiring ≥32 proposals, and
closing when windowed acceptance drops below 0.25
(`[crates/muser-engine/src/dflash/spec.rs:117-133]`) — and a disabled
request re-qualifies after a doubling cooldown (64 → 512 tokens) so cold
starts don't permanently cripple a session (`spec.rs:199-230`). The
comment on the windowed design records the one-way-latch failure that
motivated it, dated like a scar: "a cumulative rate cannot recover once
drafting stops… (2026-08-21 root cause)" (`spec.rs:184-190`).

## 8.8 Tradeoffs

**Draft cost vs verify cost, measured.** In the L1 in-process
qualification (five reps × 256 tokens, verify length 15): median cycle
~157.2 ms, of which **draft ≈ 26.9 ms** (embed 0.04 + block forward 22.3
+ LM-head/argmax 4.5) and **verify ≈ 130.2 ms** (forward 128.4, decision
1.8) `[ledger §Stage B L1]`. The draft is the *small* side of its own
loop — which is why Chapter 6's optimization energy went to the 16-row
verify kernels (the n32 tile: verify forward 202.3 → 128.4 ms), and why
making the draft *correctly conditioned* (§8.4) was worth more than
making it faster. A draft that proposes 15 tokens per 27 ms is already
cheap; a draft whose proposals are accepted is priceless.

**Synthetic vs natural — the fixture that lied.** The period-8 synthetic
stream "certified a broken draft lane for an entire campaign"
`[ledger §ROOT CAUSE FOUND AND FIXED, consequence 2]`. Since 2026-08-21,
natural-text cells are a standing part of the matrix even though they
cannot carry an exactness gate. The general rule this book keeps
re-stating: a measurement is only as good as its fixture's ability to
detect the failure mode you care about.

**Why the draft is kquant-only (Fallback B, again).** Chapter 7 measured
the alternative: native NVFP4 batched verification at 6.805 tok/s against
the 107.9 bar `[claims #4]`. The draft itself could be anything — the
barrier is the *target's* verify arithmetic in the NVFP4 lane's W4A4
batch shape. So the smallest model in the system inherits the reference
lane's format: Q4_K/Q6_K blocks, llama-pinned batch kernels, exactness
gated by lossless token equality. When your only exactness instrument is
bitwise comparison, you build on the lane that can support it.

## 8.9 What comes next — two hooks, one promise

The accept/verify algorithm itself — the `min(p/q, 1)` rule applied to
full distributions, the carried-frontier state machine, the Mirror-SD
overlap that splits the target graph at a capture layer, and the measured
rejection of the distributed verifier — is
[Ch 33](33-speculation-and-the-distributed-verdict.md)'s subject; this
chapter deliberately stopped at the draft's edge of the contract.

But Part II now closes, and Part III opens with a debt we have been
accumulating for four chapters: we have quantized, packed, and dispatched
the weights of a model we have never actually met. What *is* the
52-layer graph these formats encode? Which layers slide, which layers
have no position at all, why are there two KV heads for thirty-two query
heads, and why does a sigmoid gate sit on the attention output?
[Ch 9](09-muse-glimmer-architecture.md) is the Muse Glimmer architecture
— the model that all three lanes exist to serve.

---

## References

- `[crates/muser-engine/src/dflash.rs:1-5]` — the module contract (five
  layers, SafeTensors + kquant GGUF, Ferrite oracle lineage).
- `[crates/muser-engine/src/dflash/config.rs:59-61]` —
  `DFLASH_CONTEXT_SINK_SIZE = 64`; `:15-57` `DFlashContextGeometry` and
  the exact byte bound; `:86-91` the window-collapse doc (72.5% → 2.2%);
  `:109-136` the loud fallback; `:180-255` the GGUF config path
  (one-based target layers, `gguf-kquant`); `:263-268` the five-layer
  enforcement.
- `[crates/muser-engine/src/dflash/weights.rs:35-66]` — the dual loader
  and the projection shell; `:115-161` `validate_quantized_gguf_layouts`
  (Q4_K/Q5_K/Q6_K requirement, exact shapes).
- `[crates/muser-engine/src/dflash/hidden.rs:44-52]` — target-layer-gated
  hidden-state capture.
- `[crates/muser-engine/src/dflash/spec.rs:16-98]` — `DFlashSpecStats`
  (trace, geometry self-identification); `:117-134` the disable-gate
  constants; `:184-230` windowed gate + re-qualification;
  `:730-770` `draft_greedy` (verify lengths 3|7|15, seed echo).
- `[crates/muser-engine/src/sampling.rs:1001-1097]` — the MT-pinned
  stream and `verify_full_speculative_mt_ordered`.
- `[crates/muser-engine/src/decode.rs:213-226]` —
  `MetalSpeculativeCheckpoint` (≤16 SWA rows, NoPE metadata rewind).
- `[crates/muser-cluster/src/config.rs:44-49]` — enrollment-stamped
  `dflash_context_geometry` bound to the component digest.
- `[crates/muser-bench/src/m16.rs:202-225]` — draft shapes (fc 33280→6656,
  k/v →1024) and dtypes.
- `[crates/muser-bench/src/remote.rs:3-8, :33]` — the 256-token exact
  compare and `DFLASH_ACCEPTANCE_MINIMUM = 0.95`.
- `[docs/memory-footprint.md]` — DFlash GGUF 1,631,205,312 B.
- `[claims #15]`, `[claims #3]`, `[claims #16]`, `[claims #4]` —
  `docs/launch-claims.md`: the fixed-window synthetic ratios and their
  prohibited generalizations; exact-token 5/5 depths; 131,008 wall parity
  1.02536; native spec fail-closed.
- `[docs/benchmarks.md §2]` — verify-length conventions and the
  natural-text wins/losses (1.186/1.321 vs 0.931/0.945).
- `[ledger §L2 Stage B verdict]`, `[ledger §Stage B L0/L1]`,
  `[ledger §ROOT CAUSE FOUND AND FIXED]`,
  `[ledger §Spec re-measurement at the fixed window]` —
  `docs/goal-parity-ledger-2026-08.md`: the pre-fix 107.9136/1.3273 bar,
  the M16 microbenchmark lineage, and the window postmortem.
- [Ch 6](06-the-kquant-family.md) — the blocks the draft is built from
  and the M16 kernels its 16-row blocks run on.
- [Ch 7](07-nvfp4-native-lane.md) — Fallback B and the 6.805 no-go.
- [Ch 33](33-speculation-and-the-distributed-verdict.md) — the
  accept/verify algorithm this chapter deferred.
