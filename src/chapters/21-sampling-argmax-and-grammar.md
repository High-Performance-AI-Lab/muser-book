# Chapter 21 — Sampling, argmax, and grammar
> **status:** polished  ·  **path:** Muse Glimmer, pinned Muser tree
>
> *Prerequisites: [Ch 2](02-metal-compute-model.md) (threadgroups,
> `threadgroup_barrier`, the 1,024-thread limit), [Ch 16](16-attention-decode-kernels.md)
> (softmax, defined there for attention), [Ch 20](20-final-norm-lm-head-softcap.md)
> (the capped `[202048]` logits this chapter consumes). This chapter adapts
> the kernel skeleton: two GPU reduction kernels, then the CPU-side policy
> layers — sampler state, grammar, and exact speculative acceptance — that
> decide why the read-back is what it is.*

---

## 21.1 What it computes

[Ch 20](20-final-norm-lm-head-softcap.md) left 202,048 capped logits in a
GPU buffer. This chapter picks **one token id** — a single integer in
`[0, 202048)` — and in doing so closes the decode walk. Two families of
selection exist:

> **Greedy (argmax).** Emit the highest-scoring token every step. No
> randomness; the same prompt + weights + precision always produce the
> same sequence. Deterministic by construction.

> **Sampling.** Convert logits to a probability distribution (a
> [softmax](../glossary.md#softmax) — the attention chapter's normalization,
> applied to the vocab — usually temperature-scaled), then draw from it,
> possibly restricted to top-k / top-p / typical-p candidates. Different
> sequence every run.

Muser implements both, plus a third consumer of the same distribution:
**exact speculative acceptance** — the CPU-side step that decides how many
DFlash draft tokens to keep ([Ch 8](08-the-dflash-draft.md),
[Ch 33](33-speculation-and-the-distributed-verdict.md)). All three need
different amounts of the logits vector, and that fact explains this
chapter's architecture: the *reduction* (argmax) can run on the GPU in two
phases and cost a 4-byte read-back, but the *policy* layers need the whole
distribution on the CPU — so the serving path reads the full vocab row
back every token, and the GPU argmax lives on the no-readback routes.

## 21.2 Why greedy is the reference — determinism is the gate

The book's measured spine is exact-token parity against pinned llama.cpp
([Ch 38](38-measuring-against-llama-cpp.md)): five-repetition cells whose
verdict is *token equality*, plus full-logit SHA comparisons. Only a
deterministic policy can be gated that way — you cannot diff two random
sequences. So greedy (`--temp 0` on the comparator side) is the
qualification policy for every throughput number this book cites, and the
sampler exists for serving, not for measurement. The discipline cuts both
ways, and Muser honors the second direction too: even the *sampled* path is
pinned bit-for-bit to the comparator's RNG (§21.7), so a seeded request
replays identically across engines — determinism as a cross-engine
contract, not just a bench convenience.

## 21.3 The reduction problem — two phases under the 1,024-thread limit

Argmax is a reduction: collapse 202,048 floats to one (value, index) pair.
The natural GPU pattern is the **tree reduction** — at each step half the
threads fold in their neighbor, the active stride halves, `log₂(n)` steps
finish. The catch is geometry: a tree lives in
[threadgroup memory](../glossary.md#threadgroup) and Apple Silicon caps a
threadgroup at 1,024 threads (`maxThreadsPerThreadgroup`,
[Metal-PG]). The vocab is ~197× that. So: split the input into chunks,
reduce each chunk independently, then reduce the partials — the
**two-phase reduction** (Figure 21.1).

```
   logits [202048]                                                one token id
   ┌──────────────────────────────────────────────────────────┐     ┌───┐
   │ chunk 0 (1024) │ chunk 1 │ … │ chunk 196 │ chunk 197(320)│     │   │
   └──────────────────────────────────────────────────────────┘     └─┬─┘
       │ phase 1: one threadgroup per chunk        │        │          │
       ▼                ▼                         ▼        ▼          │
   [val,idx]₀      [val,idx]₁        …      [val,idx]₁₉₆ [val,idx]₁₉₇ │ phase 2
       └───────────────────┬───────────────────────────┘              │
                           │ 198 partials → 1 winner                  ▼
                           ▼                                      result[0]
                       (value, index) of the global max             = u32
```
*Figure 21.1: The two-phase reduction at Muse's vocab.
`⌈202,048 / 1,024⌉ = 198` chunks — `197 × 1,024 = 201,728`, so the last
chunk holds `202,048 − 201,728 = 320` elements (the ragged tail the
`gid < n` guard absorbs).*

The comparison is **strictly greater**, so ties keep the lower index —
"matching the scalar first-maximum convention" of the reference sampler
(`argmax_f32.metal:4-6`) and of the CPU helper (`api.rs:1842-1849`). That
deterministic tiebreak is load-bearing for byte-identical diffs.

## 21.4 The Metal kernels — the two-phase tree, and the greedy variant

Phase 1, verbatim — one threadgroup of 1,024 threads per chunk:

```metal
// crates/muser-engine/src/shaders/ferrite/argmax_f32.metal:7
kernel void argmax_f32_phase1(
    device const float* x           [[ buffer(0) ]],
    device       float* partial_val [[ buffer(1) ]],
    device       uint*  partial_idx [[ buffer(2) ]],
    constant     uint&  n           [[ buffer(3) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint lid  [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]])
{
    threadgroup float tg_val[1024];
    threadgroup uint tg_idx[1024];
    uint gid = tgid * 1024u + lid;
    float best_val = -INFINITY;
    uint best_idx = 0u;
    if (gid < n) {
        best_val = x[gid];
        best_idx = gid;
    }
    tg_val[lid] = best_val;
    tg_idx[lid] = best_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 512u; stride > 0u; stride >>= 1u) {
        if (lid < stride && lid + stride < tg_size && tg_val[lid + stride] > tg_val[lid]) {
            tg_val[lid] = tg_val[lid + stride];
            tg_idx[lid] = tg_idx[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) {
        partial_val[tgid] = tg_val[0];
        partial_idx[tgid] = tg_idx[0];
    }
}
```

Line by line: each thread loads one element (or `−INFINITY` if past `n` —
the 320-element tail and the idle lanes of a short phase 2 can never win a
max); one initial barrier makes all slots visible; then ten
stride-halving passes (`512 → 256 → … → 1`), a barrier per pass because
thread `lid` reads what thread `lid+stride` wrote last pass; lane 0
publishes the chunk winner. `argmax_f32_phase2` (`:41-70`) is the same
body over the 198 partials with `result[0] = tg_idx[0]` — the output is a
single `u32`, the token id. The file's own header notes the lineage:
"Exact extraction of Ferrite's two-phase GPU greedy reduction"
(`argmax_f32.metal:4-5`) — the ancestor's device [ferrite-book Ch 20],
ported with Muse's vocab arithmetic.

The **greedy serving variant** adds two fail-closed features, and its
header comment is the specification:

```metal
// crates/muser-engine/src/shaders/ferrite/argmax_f32.metal:72
// Greedy serving variant.  The high bit of every partial index carries a
// fail-closed nonfinite flag; vocabulary indices are required to fit in the
// remaining 31 bits.  `excluded` is the request's EOG set for ignore-eos
// generation.  Masking happens only inside the reduction, so the retained
// target logits remain byte-for-byte unchanged for logprob/session uses.
kernel void greedy_argmax_f32_phase1(
    // … (same buffers plus:
    //   device const uint* excluded [[ buffer(4) ]],
    //   constant uint& n_excluded  [[ buffer(5) ]],) …
```

First, **nonfinite latching**: any `NaN`/`Inf` logit sets the high bit of
its partial index, the bit propagates through the tree, and phase 2 returns
`0xffffffff` (`:156-157`) — an *error*, not a silently wrong token; the
caller converts it into a hard failure (§21.6). Second, **EOG exclusion**:
the request's end-of-generation tokens are masked to `−INFINITY` inside
the reduction only, leaving the stored logits untouched — "the retained
target logits remain byte-for-byte unchanged for logprob/session uses."
That is the ignore-eos feature done without corrupting the distribution
other consumers see.

## 21.5 The Rust dispatch

Two wrappers record the pairs. `encode_greedy_argmax_f32` (the serving
variant) — phase 1 over `⌈202,048/1,024⌉ = 198` threadgroups, an explicit
`memory_barrier_with_resources` between the phases (phase 2 reads what
phase 1 wrote — a RAW hazard, ordered by hand here), phase 2 as one
threadgroup:

```rust
// crates/muser-engine/src/metal/encode/lmhead.rs:123  (abridged to the two dispatches)
self.bind(encoder, "greedy_argmax_f32_phase1");
encoder.set_buffer(0, Some(values.metal()), 0);
// … (partials, n, excluded set + count) …
encoder.dispatch_thread_groups(MTLSize::new(blocks as u64, 1, 1), MTLSize::new(1024, 1, 1));
let partial_barrier: [&metal::ResourceRef; 2] =
    [partial_values.metal(), partial_indices.metal()];
encoder.memory_barrier_with_resources(&partial_barrier);
self.bind(encoder, "greedy_argmax_f32_phase2");
// … (partials in, result at result_offset) …
encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1024, 1, 1));
```

Its sibling `encode_argmax_f32_rows` (`lmhead.rs:83-120`) loops the plain
pair over rows of a logits matrix — one winner per row — for the DFlash
verify lane. In prose: **phase 1 grid `(198, 1, 1)` × `(1024, 1, 1)`; one
inter-phase barrier; phase 2 grid `(1, 1, 1)` × `(1024, 1, 1)`.**

## 21.6 Reading the result back — two routes, two sizes

Here is the point where Muser's design departs from the ancestor's
"4 bytes and done" story, and the departure is the chapter's core lesson.

**Route A — serving: the full distribution comes back.** `Session::decode`
holds a retained, vocab-sized CPU buffer and refills it every token:

```rust
// crates/muser-engine/src/api.rs:696
pub fn decode(&mut self, input: DecodeInput) -> Result<DecodeResult, EngineError> {
    self.validate_tokens(&[input.token_id])?;
    self.ensure_capacity(1)?;
    // The decode path refills the retained distribution in place, so a
    // token costs one vocabulary-sized copy for the result instead of two
    // fresh allocations.
    let mut logits = self.last_logits.take().unwrap_or_default();
    if let Err(error) = self.forward_into(&[input.token_id], &mut logits) {
        // … (failure leaves the previous distribution installed) …
    }
    // Fail closed: a broken row installs no distribution at all.
    // … (finite scan, diagnostics elided) …
    let next_token = argmax(&logits) as u32;
    // … (result clone, retain) …
}
```

The GPU→CPU hand-off itself is one line at the end of the batch graph —
`batch_logits.as_slice()[..token_count * self.cfg.vocab_size].to_vec()`
(`decode.rs:3659`) — a `StorageModeShared` zero-copy view followed by a
copy of **202,048 f32 = 808,192 B ≈ 789 KiB per token**
([Ch 3](03-unified-memory-and-buffers.md): unified memory makes this a
memcpy, not a device transfer). Then `argmax(&logits)` — the CPU's
five-line first-maximum scan (`api.rs:1842-1849`) — picks the greedy
token, and `ensure_finite_logits` fails closed on any nonfinite entry
(`api.rs:1851-1856`).

Why pay 789 KiB when 4 bytes would answer the greedy question? Because
serving's consumers need the *distribution*, not the winner: the sampled
chain of §21.7 (temperature, top-k, top-p, typical-p over 202,048
entries), grammar re-rolls (§21.8), logprob responses, session snapshots —
and above all **exact speculative acceptance** (§21.9), whose contract is
"acceptance against full target distributions"
(`verify_full_speculative_mt_ordered`, `sampling.rs:1033`). The one
vocabulary-sized copy is the price of every downstream policy being exact.

**Route B — the GPU-resident greedy chain: 4 bytes per token.** When (and
only when) the policy is pure greedy, Muser keeps the whole loop on the
GPU: `forward_greedy_streaming` (`decode.rs:1626`) pre-encodes a pipeline
of complete token graphs, each embedding the *previous step's argmax
result directly from GPU memory*
(`encode_embedding_q4k_from_u32_buffer` against
`dflash_argmax_results`, `decode.rs:1756-1764`), and reads back exactly
one 4-byte slot per completed token:

```rust
// crates/muser-engine/src/decode.rs:1688
let produced = self.activations.dflash_argmax_results.as_slice()[completed].to_bits();
if produced == u32::MAX || produced as usize >= self.cfg.vocab_size {
    return Err(MetalModelError::InvalidSnapshot(
        "pipelined greedy argmax observed nonfinite logits or an invalid token".into(),
    ));
}
```

The `.to_bits()` reinterprets the stored f32 slot as the `u32` the kernel
wrote — **four bytes of meaningful data cross per token** — and the
`u32::MAX` check is the greedy kernel's fail-closed flag from §21.4
becoming a hard error. This route serves the DFlash block-decode lane and
the no-readback benchmark policy ("no-per-token-host-readback" is part of
the teacher-forced comparator contract, `decode.rs:2118-2123`). It is the
ancestor's 4-byte device [ferrite-book Ch 20], alive on exactly the routes
where the policy permits it.

## 21.7 The sampler — pinned RNG, per-request state

The sampled path's first commitment is the random number generator: a
bit-for-bit reimplementation of libc++'s `std::mt19937` — "the
`std::mt19937` engine used by the source-pinned llama.cpp sampler",
kept in-tree "rather than using `StdRng`, whose algorithm is deliberately
unspecified … makes seeded API results stable across Rust and `rand`
releases" (`sampling.rs:53-56`), with libc++'s exact `uniform_f32`/`f64`
conversions (`:107-120`) and snapshot/restore for durable sessions
(`:88-105`). The engine test pins known vectors
(`mt19937_matches_libcxx_engine_and_uniform_distributions`,
`:1105-1133`).

The distribution chain is a scalar, ordered pipeline —
`distribution_ordered` (`sampling.rs:399`) — applying, in source order:
`top_n_sigma` masking against the max (llama's newer filter), `top_k`
truncation, `typical_p`, `top_p` nucleus cutoff, each over the full
candidate list with llama-matching tie and ordering conventions (the
source comments mark each: "Upstream masks in place and intentionally
leaves candidate order untouched", `:432-434`; "Locally-typical order is
part of the source contract", `:477-479`).

The state lives per request in the server, as four separated RNG streams
plus sampler scalars — snapshottable for session persistence:

```rust
// crates/muser-server/src/openai.rs:4331
struct RequestSamplerState {
    distribution_rng: Mt19937,
    xtc_rng: Mt19937,
    mirostat_rng: Mt19937,
    mirostat_mu: f32,
    adaptive: AdaptiveSamplerState,
}
```

The separation is deliberate: each stochastic feature burns its own stream,
so enabling XTC cannot shift the draws the distribution sampler sees.
Per-slot independence in serving is the scheduler's business
([Ch 34](34-scheduler-and-slots.md)); this struct is the per-request
half of that story, and its snapshot/restore is what lets a migrated
session resume its exact draw sequence.

## 21.8 Grammar-constrained sampling — GBNF on the CPU

Structured-output requests constrain the token stream to a grammar — JSON
schemas, quoted literals. The engine surface is a pinned llama-style
**GBNF** matcher, and its module header is the specification:

```rust
// crates/muser-server/src/grammar.rs:1
//! Pinned llama-style GBNF parsing and incremental UTF-8 matching.
//!
//! The matcher is an Earley recognizer over Unicode code points. It keeps all
//! ambiguous stacks alive, accepts token byte fragments that end in a partial
//! UTF-8 sequence, and exposes acceptance separately so EOS is legal only at
//! a completed root rule.
```

An **Earley recognizer** is a chart parser that keeps every viable parse
stack alive simultaneously — necessary because a tokenizer can split one
grammar-legal string several ways. The matcher consumes *bytes*, so a
token ending mid-UTF-8-sequence is accepted as a partial and the grammar
state advances only when the code point completes; EOS is admitted only at
the root, so the grammar can never strand an incomplete literal.

Selection uses llama-server's **rejection sampling** discipline, with the
reason in source:

```rust
// crates/muser-server/src/openai.rs:4469
// Pinned llama-server uses grammar rejection sampling: run the
// ordinary chain first, accept an eligible result immediately,
// and only rerun with a grammar mask after a rejected result. The
// rejected draw deliberately advances every stochastic sampler.
```

So the sampler of §21.7 runs unmolested; the winner is checked with
`grammar_allows(grammar, model, first, eos)` (`openai.rs:4982`); only a
rejection triggers a masked rerun. And the rejected draw still advances
every RNG — replicating llama-server's draw-stream semantics exactly,
which is what keeps seeded, grammar-constrained generations comparable
across engines. This is another reason the full distribution rides to the
CPU: the mask-and-rerun needs the whole candidate vector.

## 21.9 Exact speculative acceptance — the CPU contract

The third consumer closes the loop with [Ch 8](08-the-dflash-draft.md)'s
draft model. When DFlash proposes tokens, acceptance is computed on the
CPU against the full target distribution — the function the campaign
calls the exactness anchor for speculation:

```rust
// crates/muser-engine/src/sampling.rs:1033
pub fn verify_full_speculative_mt_ordered(
    draft_tokens: &[u32],
    draft_probabilities: &[Vec<f32>],
    target_probabilities: &[Vec<f32>],
    target_orders: &[Vec<u32>],
    rng: &mut Mt19937,
) -> Result<SpeculativeDecision, SamplingError> {
    // … (geometry validation elided) …
    for (index, (&token, (draft, target))) in draft_tokens
        .iter()
        .zip(draft_probabilities.iter().zip(target_probabilities))
        .enumerate()
    {
        let token = token as usize;
        // … (bounds check elided) …
        let q = draft[token];
        let p = target[token];
        let acceptance = if q <= 0.0 { 1.0 } else { (p / q).min(1.0) };
        if rng.uniform_f32() <= acceptance {
            continue;
        }
        let mut residual = target
            .iter()
            .zip(draft)
            .map(|(&p, &q)| (p - q).max(0.0))
            .collect::<Vec<_>>();
        let total = residual.iter().sum::<f32>();
        if total <= 0.0 {
            residual.clone_from(target);
        } else {
            for probability in &mut residual {
                *probability /= total;
            }
        }
        // … (lines elided: the filter that builds `order` from
        //     `target_orders[index]`, keeping tokens with positive
        //     residual; see file) …
        return Ok(SpeculativeDecision {
            accepted: index,
            next_token: sample_distribution_mt_ordered(&residual, &order, rng)?,
        });
    }
    // … (all-accepted path: sample from the last target row) …
}
```

Read the math: each draft token is accepted with probability
`min(p/q, 1)` — the standard speculative-decoding rejection rule, which
makes the *combined* draft+verify process sample exactly from the target
distribution. On the first rejection, the residual
`max(p − q, 0)` is renormalized and the replacement token is drawn from it
through the same pinned Mt19937 stream (`uniform_real_distribution<float>`
per attempt, the libc++ double draw for selection — `sampling.rs:1001-1007`).
"Exact" is not a hope here; the qualifier compares every full target-logit
row in its gates (256 greedy tokens plus all rows,
[crates/muser-bench/src/remote.rs:8-10]). The engine-level driver
(`Session::verify_batch`, `api.rs:913`; the Metal mirror-SD split at
`decode.rs:3298`) gets the full story in
[Ch 33](33-speculation-and-the-distributed-verdict.md) — including the
measured fate of moving this acceptance off the Mac.

## 21.10 Tradeoffs

**Full-distribution read-back vs GPU-only selection.** Route A costs
~789 KiB of unified-memory copy per token — arithmetic against the token's
weight read (~16.76 GB at the artifact scale, [Ch 1](01-why-inference-is-a-memory-problem.md))
puts it near 0.005 %, though it is a *serial* addition on the critical
path, not a bandwidth event; no retained measurement isolates its wall
cost **[unverified]**. What it buys is exactness for three consumers at
once: the sampler chain, grammar re-rolls, and speculative acceptance
against full target distributions. Route B (4 bytes/token) exists precisely
for the policy that needs none of them — pure greedy — and is the
no-readback comparator policy. The ancestor's framing — the GPU already
found the max; copying the vector buys nothing *for greedy* — survives
intact, scoped to the routes where it is true.

**Two-phase tree vs one big kernel.** A single tree over 202,048 logits
would need one threadgroup of 202,048 threads — impossible under the
1,024-thread cap, and the required threadgroup memory
(`202,048 × 8 B ≈ 1.54 MiB`) would blow the per-threadgroup budget on its
own. Two phases cost one extra dispatch and one barrier (§21.5) — noise
against the 924.6 MB LM head that precedes them.

**Rejection-sampled grammar vs masked-first sampling.** Masking the
distribution *before* sampling would guarantee grammar legality in one
draw, but it changes the draw stream and the probabilities the sampler
consumes; llama-server's rejection form (ordinary chain, check, re-roll
with mask, rejected draw still advances every RNG — `openai.rs:4469-4472`)
preserves the comparator's semantics at the cost of an occasional second
sample. Muser pins the rejection form for the same reason it pins
mt19937: cross-engine reproducibility is the product feature.

**CPU acceptance vs GPU acceptance for speculation.** The acceptance rule
itself is trivially parallelizable, but its inputs are two full
vocab-sized distributions per draft token and its outputs feed the pinned
RNG; keeping it on the CPU next to the sampler keeps one authoritative
draw stream. The distributed lane that moved verification *off the Mac*
was measured and rejected on throughput, not on exactness — the
verifier-only ceilings of 20.15/40.04/55.96 tok/s against the 107.9 tok/s
kquant bar [nvfp4-distributed-speculative-frontier-20260818; [Ch 33](33-speculation-and-the-distributed-verdict.md)]
— and the native W4A4 variant collapsed to 6.805 tok/s with verification
consuming 35.915 s of a 37.619 s span [nvfp4-fast-lane-evidence; ledger
F-series]. Exactness was never the casualty; time was.

## 21.11 Where the gap lives

**Sampling is not the Metal gap — it is not on the Metal graph at all.**
The dispatch-gap accounting covers closures from embedding to softcap;
selection runs host-side on the read-back (this chapter, §21.6), and the
GPU argmax pair exists for routes that *remove* host round-trips. The
+196-closure story of [Ch 19](19-downproj-and-residual.md) §19.8 is
untouched by any of this. If anything, selection is where the *other*
currency is spent: the read-back's serialization and the sampler's scalar
CPU loops are the kind of cost [Ch 34](34-scheduler-and-slots.md)'s
rendezvous budget has to absorb — measured in phase timings like
`grammar` and `argmax_ns` (`state.rs:1566`, `api.rs:719-727`), not in
dispatch groups.

---

*And with that, the kernel walk is done: eleven chapters from an embedding
lookup to a chosen token, every dispatch accounted. But look back at the
path — every attention chapter from [Ch 14](14-qk-norm-and-rope.md) onward
quietly depended on a structure this Part never costed: the KV that Q·Kᵀ
read and the store kernel wrote, the ring that wrapped at 2,048, the planes
that grew without bound. Each layer's attention was borrowing memory at
one thousand twenty-four bytes per token per layer
([Ch 15](15-kv-store-and-the-ring.md)) and we never once asked what the
loan costs. That debt is about to become the whole story — what context
weighs, why it — not the weights — decides how many slots a 96 GB Mac can
serve, and how a cache becomes an asset you can seal, move, and trust.
Part V begins with the bill: [Ch 22](22-the-price-of-context.md).*

---

## References

- `crates/muser-engine/src/shaders/ferrite/argmax_f32.metal:7-39` —
  `argmax_f32_phase1` (the chunk tree); `:41-70` `argmax_f32_phase2`;
  `:72-158` the greedy pair with the nonfinite latch and EOG exclusion
  (headers quote the contract).
- `crates/muser-engine/src/metal/encode/lmhead.rs:83-120` —
  `encode_argmax_f32_rows`; `:123-161` `encode_greedy_argmax_f32` (the
  inter-phase barrier).
- `crates/muser-engine/src/api.rs:696-737` — `Session::decode`: the
  retained distribution, CPU argmax, fail-closed finite scan;
  `:1842-1849` the CPU `argmax`; `:1851-1856` `ensure_finite_logits`.
- `crates/muser-engine/src/decode.rs:1626-1728` —
  `forward_greedy_streaming`, the GPU-resident chain; `:1688-1693` the
  4-byte read and the `u32::MAX` fail-closed check; `:1730-1765` the
  embedding-from-argmax-buffer link; `:3659` the batch graph's full-vocab
  read-back; `:2118-2123` the teacher-forced no-readback contract.
- `crates/muser-engine/src/sampling.rs:53-120` — `Mt19937` and its libc++
  conversions; `:399-509` `distribution_ordered` (the sampler chain);
  `:1008-1097` `verify_full_speculative_mt` / `_mt_ordered` (exact
  acceptance); `:1105-1133` the libc++ vector test.
- `crates/muser-server/src/openai.rs:4266-4379` —
  `AdaptiveSamplerState` / `RequestSamplerState` (four RNG streams,
  snapshot/restore); `:4413-4509` `sample_or_argmax` (logit bias, repeat
  penalties, grammar rejection sampling); `:4982` `grammar_allows`.
- `crates/muser-server/src/grammar.rs:1-71` — the GBNF Earley matcher's
  specification and types.
- `crates/muser-bench/src/remote.rs:8-10,36-37` — the exact-verification
  comparator policy (256 tokens + all logit rows; 0.95 acceptance floor).
- [nvfp4-distributed-speculative-frontier-20260818] — the rejected
  distributed-verifier lane (verifier-only ceilings vs the 107.9 bar).
- [nvfp4-fast-lane-evidence-20260817] / [ledger F-series] — the native
  W4A4 verification no-go (6.805 tok/s).
- [Ch 8](08-the-dflash-draft.md), [Ch 33](33-speculation-and-the-distributed-verdict.md)
  — the draft model and the full speculative story this chapter previews.
- [Ch 34](34-scheduler-and-slots.md) — slots, rendezvous, and where the
  CPU-side selection cost lands; [Ch 38](38-measuring-against-llama-cpp.md)
  — the exact-token gate that makes greedy the reference policy.
- [ferrite-book Ch 20] — the ancestor's argmax chapter: the two-phase
  tree, the deterministic tiebreak, and the 4-byte read, all ported and
  re-scoped to Muser's two read-back routes.
- [Metal-PG] — `maxThreadsPerThreadgroup` and the threadgroup-memory
  budget behind §21.3's limits.
