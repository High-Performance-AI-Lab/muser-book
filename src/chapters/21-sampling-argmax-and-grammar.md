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
different amounts of the logits vector, and that is the question this
chapter is really about. Finding a maximum is the easy part; the hardware
does that in its sleep. The hard part is deciding how much of the
distribution has to be standing in front of the policy at the moment it
chooses — because that, and not the reduction, is what sets the traffic
between GPU and CPU. Hence the architecture: the *reduction* (argmax) can
run on the GPU in two
phases and cost a 4-byte read-back, but the *policy* layers need the whole
distribution on the CPU — so the serving path reads the full vocab row
back every token, and the GPU argmax lives on the no-readback routes.

## 21.2 Why greedy is the reference — determinism is the gate

Everything that follows turns on a question that is easy to walk straight
past: which policy is allowed to be the *reference* — the one the product
ships, or the one a test can check?

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

Where does the difficulty in picking a maximum actually live? Not in the
comparison — that is one instruction. It lives in the geometry of who is
allowed to talk to whom.

Argmax is a reduction: collapse 202,048 floats to one (value, index) pair.
The natural GPU pattern is the **tree reduction** — at each step half the
threads fold in their neighbor, the active stride halves, `log₂(n)` steps
finish. The catch is exactly that geometry: a tree lives in
[threadgroup memory](../glossary.md#threadgroup) and Apple Silicon caps a
threadgroup at 1,024 threads (`maxThreadsPerThreadgroup`,
[Metal-PG]). The vocab is ~197× that. So: split the input into chunks,
reduce each chunk independently, then reduce the partials — the
**two-phase reduction** (Figure 21.1). Put the other way round: the
hardware will not let you hold one conversation among two hundred thousand
threads, so you hold a small conversation per chunk and then one more among
the chunk winners.

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
deterministic tiebreak is load-bearing for byte-identical diffs. It is worth
sitting with why: if two logits landed exactly equal and the tiebreak
wobbled between them — a different chunk winning on a different run — a
parity cell would fail for a reason that has nothing to do with arithmetic,
and the failure would be intermittent, which is the worst kind to chase.
The strict `>` is what makes ties boring.

## 21.4 The Metal kernels — the two-phase tree, and the greedy variant

Two things are worth watching for as you read the kernel below, because
they are the questions a tree reduction always has to answer. What does a
thread do when there is no element for it to load? And why does a barrier
appear *inside* the loop rather than once before it?

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

That is the reduction as a textbook would leave it. Serving asks two more
questions of it. What should a maximum-finder do when one of the numbers it
is comparing is not a number at all? And how do you honour a request that
says "never stop" without lying to everyone else about what the model
actually scored? The **greedy serving variant** answers both, and its
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

A two-kernel reduction has a hazard the one-kernel version does not: the
second kernel reads a buffer the first one wrote, and nothing in Metal
volunteers to order that for you. Getting it wrong does not crash — it
silently reads stale partials, which is how you end up debugging a sampler
that is occasionally, unreproducibly wrong. So the ordering is stated by
hand.

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

Everything so far argues for a tiny read-back, and a tiny read-back is
exactly what we expected to ship. The ancestor had already settled the
question: the GPU already found the maximum, so copy the maximum and
nothing else — "4 bytes and done". We carried that expectation into the
serving path, and it did not survive there: every consumer serving cares
about turned out to want the numbers the argmax throws away.

This is the point where Muser's design departs from the ancestor's story,
and the departure is the chapter's core lesson: the size of the read-back
is not decided by the reduction. It is decided by the policy standing
behind it. Muser therefore has two routes, and which one a request takes
is a statement about what that request intends to do with the numbers.

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

The GPU→CPU hand-off itself is one line at the end of the batch graph:
`batch_logits.as_slice()[..token_count * self.cfg.vocab_size].to_vec()`
(`decode.rs:3659`). Read it in two halves. The `as_slice()` is a
`StorageModeShared` zero-copy view — the CPU is looking straight at the
buffer the GPU wrote, and nothing has moved yet
([Ch 3](03-unified-memory-and-buffers.md): unified memory makes this a
memcpy, not a device transfer). The `to_vec()` is the half that costs:
**202,048 f32 = 808,192 B ≈ 789 KiB per token**, copied so the caller owns
a row that will not change under it.

What happens to that row next is deliberately dull. The CPU's own
five-line first-maximum scan picks the greedy token, and
`ensure_finite_logits` refuses the row outright if any entry is nonfinite —
the same fail-closed instinct as the kernel's high-bit latch, arriving by a
different road. Both helpers sit a few lines apart in the same file
(`api.rs:1842-1849` and `api.rs:1851-1856`).

Why pay 789 KiB when 4 bytes would answer the greedy question? Because
serving's consumers need the *distribution*, not the winner: the sampled
chain of §21.7 (temperature, top-k, top-p, typical-p over 202,048
entries), grammar re-rolls (§21.8), logprob responses, session snapshots —
and above all **exact speculative acceptance** (§21.9), whose contract is
"acceptance against full target distributions"
(`verify_full_speculative_mt_ordered`, `sampling.rs:1033`). The one
vocabulary-sized copy is the price of every downstream policy being exact.

Said the other way round, because this is the sentence the rest of the
chapter hangs on: the read-back is not sized by what the winner costs to
report, it is sized by what the hungriest consumer on the route needs to
see. Greedy alone would be cheap. Greedy plus a grammar plus a speculator
is not, and you do not get to find out which one you are until the request
arrives.

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

What breaks if the random number generator is wrong? Nothing you can see
in a single response — and everything you can see in a diff. Two engines
given the same seed and the same prompt have to walk the same sequence of
draws, or the cross-engine determinism contract from the top of this
chapter is a slogan.

So the sampled path's first commitment is not a sampler at all; it is an
engine. Muser carries a bit-for-bit reimplementation of libc++'s
`std::mt19937` — "the `std::mt19937` engine used by the source-pinned
llama.cpp sampler". It is kept in-tree deliberately, "rather than using
`StdRng`, whose algorithm is deliberately unspecified … makes seeded API
results stable across Rust and `rand` releases". The same commitment runs
downward: libc++'s exact `uniform_f32`/`f64` conversions rather than
Rust's, and snapshot/restore of the engine state, so a durable session
reopened later resumes mid-stream instead of quietly reseeding. The
evidence trail is short and worth keeping in one place — the engine and
its rationale at `sampling.rs:53-56`, the conversions at `:107-120`,
snapshot and restore at `:88-105`, and the test that pins known vectors
against libc++ at `mt19937_matches_libcxx_engine_and_uniform_distributions`
(`:1105-1133`).

On top of the engine sits the distribution chain: a scalar, ordered
pipeline, `distribution_ordered` (`sampling.rs:399`). The order is not
ours to choose. It applies, in source order, `top_n_sigma` masking against
the max (llama's newer filter), `top_k` truncation, `typical_p`, then the
`top_p` nucleus cutoff, each over the full candidate list. What makes it
delicate is that the tie and ordering conventions must match upstream even
where upstream looks careless — and the source comments say so out loud:
"Upstream masks in place and intentionally leaves candidate order
untouched" (`:432-434`), "Locally-typical order is part of the source
contract" (`:477-479`). Those two comments are the difference between a
filter that *agrees* with llama.cpp and one that merely resembles it.

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

Now the awkward case. A request demands JSON, and the sampler — which
knows nothing about JSON — draws a token that would break it. What should
happen to that draw?

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

The third consumer is the hungriest, and the one with the most to lose.
Speculation is supposed to be free: a small model guesses ahead, the big
model checks the guesses, and the output is *the same text you would have
got anyway*, only sooner. Get the check subtly wrong and it stops being a
speed-up and becomes a quality change nobody asked for and nobody can see
in a benchmark.

That closes the loop with [Ch 8](08-the-dflash-draft.md)'s
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

In plainer words: the draft model is allowed to be wrong as often as it
likes, and the residual step is what makes that harmless. It is not
allowed to change what the target model would have said. The accept rule
and the residual draw are two halves of one guarantee, and dropping either
half turns speculation from a lossless optimization into an approximation.
"Exact" is not a hope here; the qualifier compares every full target-logit
row in its gates (256 greedy tokens plus all rows,
[crates/muser-bench/src/remote.rs:8-10]). The engine-level driver
(`Session::verify_batch`, `api.rs:913`; the Metal mirror-SD split at
`decode.rs:3298`) gets the full story in
[Ch 33](33-speculation-and-the-distributed-verdict.md) — including the
measured fate of moving this acceptance off the Mac.

## 21.10 Tradeoffs

**Full-distribution read-back vs GPU-only selection.** Route A costs
~789 KiB of unified-memory copy per token. Set that beside the traffic the
same token already generates — a weight read of ~16.76 GB at the artifact
scale ([Ch 1](01-why-inference-is-a-memory-problem.md)) — and the copy
lands near 0.005 % of it, which sounds like the end of the argument. It is
not, and this is the part that trips people up. The copy is not a
bandwidth event competing with the weights; it is a *serial* addition on
the critical path, and cheap bytes on a critical path are still time
somebody waits for. How much time, we cannot say: no retained measurement
isolates its wall cost **[unverified]**.

What the copy buys is exactness for three consumers at
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
is trivially parallelizable, and for a while that looked like an
invitation. Verification is the expensive half of speculation; the Mac is
the busy machine; there was other hardware sitting on the network. Why not
move the verifier off the Mac and let the Mac get on with decoding?

We built that lane and measured it, expecting the Mac's returned time to
pay for the wire. It did not come close. Take the wire and the draft model
out of the accounting entirely and ask only how fast the remote verifier
could go on its own: the verifier-only ceilings came in at
20.15/40.04/55.96 tok/s against the 107.9 tok/s kquant bar — the *best
imaginable* case for the lane was already far under the number it had to
beat, so no amount of tuning downstream could rescue it. That run is
retained [nvfp4-distributed-speculative-frontier-20260818;
[Ch 33](33-speculation-and-the-distributed-verdict.md)].
The native W4A4 variant went further in the wrong direction: 6.805 tok/s,
with verification alone consuming 35.915 s of a 37.619 s span
[nvfp4-fast-lane-evidence; ledger F-series]. At that point the verifier
was not a step inside the decode loop; it *was* the loop.

What the failure taught is the sentence to take away from the whole
section: exactness was never the casualty, time was. Nothing about moving
acceptance would have made it less exact — the rule is the rule wherever
it runs. What moving it does change is everything around it: the rule's
inputs are two full vocab-sized distributions per draft token, and its
outputs feed the pinned RNG. So keeping acceptance on the CPU beside the
sampler is not a purity argument. It is the arrangement that leaves one
authoritative draw stream, and no network between the two halves of a
decision that has to be made for every token.

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
