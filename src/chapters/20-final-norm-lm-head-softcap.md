# Chapter 20 — Final norm, LM head, and the soft cap
> **status:** polished  ·  **path:** Muse Glimmer, pinned Muser tree
>
> *Prerequisites: [Ch 6](06-the-kquant-family.md) (the Q5_K block),
> [Ch 12](12-rmsnorm-and-the-dual-eps-sandwich.md) (RMSNorm; the final norm
> is one of those), [Ch 13](13-the-qkv-gate-matvec-family.md) (the pinned
> ggml matvec family — this chapter runs it at its largest shape),
> [Ch 19](19-downproj-and-residual.md) (the last tail wrote
> `activations.hidden` through the final norm). This chapter leaves the
> per-layer loop and produces the model's raw scores — 202,048 of them.*

---

## 20.1 What it computes

Every chapter of the decode walk so far has handed the residual stream back
to itself, one layer richer. This one spends it. The question we are finally
answering is easy to ask and expensive to compute: given the model's final
thought, what score does it assign to every word it knows?

After layer 51's tail, the [residual stream](../glossary.md#residual-stream-hidden-state)
is one 6,656-vector: the model's final thought. Three operations turn it
into a prediction:

1. **The final norm** — an RMSNorm with the `output_norm.weight` γ vector
   and the ordinary `rms_eps = 1e-5`. On the teacher-forced route there is
   no separate dispatch for it at all: layer 51's tail *is* the final norm,
   because its `next_output` selection writes `activations.hidden` through
   `output_norm` (`decode.rs:5869-5876`, [Ch 19](19-downproj-and-residual.md)
   §19.5). The serving batch graph can also run it as its own
   `encode_rms_norm_mul` node (`decode.rs:5229-5241`) — node shape follows
   the pinned graph, the math does not change.
2. **The LM head** — one giant [matvec](../glossary.md#matvec) projecting the
   normalized 6,656-vector up to a **[logit](../glossary.md#logits)** per
   vocabulary entry:

```
logits = W_lm · normed              W_lm : [202048 × 6656], Q5_K
logits[i] = Σ_{j=0..6655} W_lm[i, j] · normed[j]
```

   A logit is a raw, unnormalized score — one f32 per token in the
   vocabulary, no probability yet, no bound yet.
3. **Scale, then soft cap** — the chapter's titular kernel:

```
value   = logits[i] × logit_scale          logit_scale = 0.196116 (GGUF metadata)
logits[i] = 20 × tanh(value / 20)          final_logit_softcap = 20 (GGUF metadata)
```

   Every logit ends up in the open interval (−20, 20).

## 20.2 Why it exists — the unprojection, and why bound it

The **LM head** (`output.weight`) is the *unembedding*: the inverse role of
the token embedding of [Ch 11](11-token-embedding-lookup.md). The embedding
mapped one hot token id to a learned 6,656-vector; the head scores every
vocabulary row against the final hidden state — 202,048 independent dot
products, one per token, each of length 6,656. The largest of them wins and
becomes the next token ([Ch 21](21-sampling-argmax-and-grammar.md)). It is
the only per-token projection that is not repeated per layer: a one-shot
bandwidth hit, not a 52× multiplier.

**Why a soft cap?** Without it, logits are unbounded — a very confident
dot product can score 50, 100, or more, and training with such outliers is
unstable (huge gradients through the softmax). A soft cap squashes the
tails smoothly: `20·tanh(x/20)` is ≈ `x` for small `x` and approaches ±20
asymptotically, so ordinary scores pass nearly unchanged while outliers are
bent into the bound. Said another way: near the origin the cap is invisible,
the identity map to within a rounding error, and it only becomes a real
transformation once a score is large enough that its exact magnitude was
never trustworthy anyway.

Where the idea came from is a question we can only half-answer. The mechanism is
Gemma-2-lineage; that it is *why Muse Glimmer's* authors adopted it is not in
the Muser tree — **[unverified]**, and we would rather say so than invent a
motive for a checkpoint we did not train. What we *can* verify is the wiring
and the constants. The engine reads both from the checkpoint
(`config.rs:190-197`); llama.cpp defaults the softcap to 30.0 when the key is
absent; this checkpoint carries `final_logit_softcapping = 20` with
`logit_scale = 0.196116` [docs/release-provenance.md:822-823].

One property matters enormously downstream and costs one line to prove:
**the transform is strictly increasing.** `x ↦ 20·tanh((x·s)/20)` with
`s > 0` preserves order — if `logits[a] < logits[b]` before, then after.
Greedy token selection ([Ch 21](21-sampling-argmax-and-grammar.md)) is
therefore *invariant* to the soft cap: the argmax index cannot move. What
the cap changes is the *gaps* (§20.7), and gaps are exactly what logprob
comparisons consume.

## 20.3 The matrix operation — the largest matvec, by hand

Where does the time in this stage go? Almost all of it goes into dragging one
matrix off memory, so before anything else we want that matrix's size in
bytes — derived rather than asserted, so you can re-derive it yourself and
catch us if we are wrong.

The shapes, verified from the artifact's verify-shape table
(`lm_head 6656->202048 q5k`, `crates/muser-bench/src/m16.rs:195-197`):
`rows = 202,048` (vocab), `cols = 6,656` (hidden), Q5_K.

Derive the byte size step by step — Q5_K packs 256 elements per 176-byte
block ([Ch 6](06-the-kquant-family.md)):

```
  blocks per row        = 6,656 / 256            =     26
  bytes per row         = 26 × 176                =  4,576 B
  total weight bytes    = 202,048 × 4,576         =  924,571,648 B
                                                   ≈  924.6 MB  (SI)
  share of artifact     = 924,571,648 / 16,756,681,056  ≈  5.52 %
  bits per element      = 176/256 × 8              =  5.5 bits
```

*924.6 MB in one matrix* — larger than any single layer's entire FFN
(224–259 MB, [Ch 18](18-swiglu-ffn.md)), read **once per token**. The
"17× more rows than the next-largest projection" framing of the ancestor
book [ferrite-book Ch 19] recurs here with new numbers: the next-largest
output width is the FFN's 19,968; the head's 202,048 is **10.1×** that,
and 30× the attention block's 6,656-output projections.

The output is also the largest activation the model materializes:

```
  logits buffer = 202,048 × 4 B = 808,192 B ≈ 789 KiB of f32
```

Still three orders of magnitude smaller than the weights that produced it.
Figure 20.1 shows the stage end to end.

```
   normed [6656]        W_lm [202048 × 6656]  Q5_K              logits [202048]
   ┌           ┐        ┌──────────────────────────────────┐    ┌───────────┐
   │ 26,624 B  │   ×    │ 202,048 rows × 4,576 B/row      │ →  │ 808,192 B │
   └           ┘        │ = 924,571,648 B read once/token │    └───────────┘
   fits in cache        └──────────────────────────────────┘    one score per
                                                               vocab token
```
*Figure 20.1: The LM-head matvec. The row count is the vocabulary; the
26 KiB input is re-read by every threadgroup and lives in cache; the
924.6 MB weight stream is the work.*

## 20.4 The Metal kernel — `muser_scale_softcap_inplace`, and what actually runs

Two kernels in this tree compute the soft cap, and the one that reads best is
not the one that runs. Start with the readable one, because it is also the
definition.

The fused kernel, verbatim — it is the whole soft-cap formula in eight
lines:

```metal
// crates/muser-engine/src/shaders/muse_reference.metal:15
kernel void muser_scale_softcap_inplace(
    device float *logits [[buffer(0)]],
    constant uint &count [[buffer(1)]],
    constant float &scale [[buffer(2)]],
    constant float &softcap [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) {
        float value = logits[index] * scale;
        logits[index] = softcap > 0.0f ? softcap * tanh(value / softcap) : value;
    }
}
```

One thread per logit; multiply by `scale`, divide by the cap, `tanh`,
multiply by the cap, overwrite in place. The `softcap > 0` guard means the
same kernel serves uncapped models (scale only).

But here is this chapter's twist, in the same spirit as
[Ch 18](18-swiglu-ffn.md)'s flag story: **on the serving route this fused
kernel is not what runs.** When the pinned llama.cpp metallib is loaded,
the wrapper deliberately decomposes the soft cap into *four separately
published unary nodes* — the source comment states why:

```rust
// crates/muser-engine/src/metal/encode/lmhead.rs:242
// Match pinned llama.cpp's graph literally: the LM head is
// followed by four independently published unary nodes.  The
// previous combined kernel used a different tanh implementation
// and expression tree, so equal pre-softcap logits did not yield
// equal public bytes.
self.encode_ggml_unary_inplace(encoder, scale_pipeline, logits, count, scale);
if softcap > 0.0 {
    for (pipeline, factor) in [
        (scale_pipeline.as_ref(), 1.0f32 / softcap),
        (tanh_pipeline.as_ref(), 0.0),
        (scale_pipeline.as_ref(), softcap),
    ] {
        let barrier: [&metal::ResourceRef; 1] = [logits.metal()];
        encoder.memory_barrier_with_resources(&barrier);
        self.encode_ggml_unary_inplace(encoder, pipeline, logits, count, factor);
    }
}
```

Read that comment for what it is: a retracted attempt, kept in the source
where it can still teach. The fused kernel above was the obvious engineering
answer — one dispatch, one pass over the buffer, algebraically identical to
the step-by-step form. We shipped it, expecting byte-identical logits out of
it, because the arithmetic is the same arithmetic. The bytes disagreed.
Metal's `tanh` and the fused expression tree round differently from llama's
node-per-op graph, and the failure surfaced exactly where it costs the most:
"equal pre-softcap logits did not yield equal public bytes." The lesson is
one this book keeps relearning in new costumes — algebraic identity is not
floating-point identity, and a comparator that gates on published bytes will
find every place the two come apart.

So the serving route decomposes instead. Four dispatches — `×scale`,
`×(1/20)`, `tanh`, `×20` — with a memory barrier between each, using the
comparator's own `kernel_…_unary` PSOs: not kernels that behave like
llama.cpp's, but llama.cpp's own, registered as ggml unary ops 10 = scale
and 100 = tanh (`encode.rs:288-289`). The fused kernel is not deleted, it is
demoted: it still runs when the metallib is absent or the count is not a
multiple of four (`lmhead.rs:230-241`, `:261-266`), and a strict cross-vendor
split — scale / barrier / scale / barrier / tanh / barrier / scale — exists
for CUDA-parity lanes (`lmhead.rs:197-228`).

The CPU oracle states the reference semantics — same formula, scalar
order (`crates/muser-engine/src/reference.rs:559-569`):

```rust
let mut logits = vec![0.0f32; t * cfg.vocab_size];
matmul(&self.w("output.weight"), &hidden, t, &mut logits);
for l in logits.iter_mut() {
    *l *= cfg.logit_scale;
}
if cfg.final_logit_softcap > 0.0 {
    let cap_v = cfg.final_logit_softcap;
    let inv = 1.0 / cap_v;
    for l in logits.iter_mut() {
        *l = cap_v * (*l * inv).tanh();
    }
}
```

## 20.5 The Rust dispatch

What does it take to launch the largest matvec in the model? Less than you
would guess, and that is the point worth carrying away: the head goes through
the same wrapper as the smallest projection in the layer loop, with nothing
special-cased for its size.

The head itself is the stock projection wrapper —
`self.project(command, &self.output, &self.activations.hidden,
&self.activations.logits)` (`decode.rs:5892-5897`) — routing through
`encode_projection` to the one-token pinned path of
[Ch 13](13-the-qkv-gate-matvec-family.md). `output.weight` is **Q5_K**, so
the dtype table gives `(block_bytes, rows_per_group) = (176, 1)` and the
launch is:

```
  grid        = 202,048 ÷ (1 row/group × 2 simdgroups) = 101,024 threadgroups
  threadgroup = (32, 2, 1) = 64 threads (two SIMD groups, one row each)
  kernel      = kernel_mul_mv_q5_K_f32   (pinned llama.cpp metallib)
```

(`qkv.rs:429-450`; the Muser fallback family has no Q5_K one-token
specialization — `muser_matvec_q5k_4sg` exists in `muse_reference.metal`
but the pinned path is taken whenever the metallib is present,
`docs/quickstart.md:16`'s fail-closed note again.)

The soft-cap entry point is `encode_scale_softcap`
(`lmhead.rs:163-171`), which delegates to `encode_scale_softcap_count`
with the full vocab count; the constants arrive from the config:

```rust
dispatch(command, |encoder| {
    self.kernels.encode_scale_softcap(
        encoder,
        &self.activations.logits,
        cfg.logit_scale,
        cfg.final_logit_softcap,
    );
});
```
(`decode.rs:5898-5905` — the last dispatch of the token graph.)

**Where `logit_scale` comes from — metadata, not code.** This is worth a
paragraph of its own because it is easy to get wrong: `0.196116` is *not*
a constant in the engine. It is read from the GGUF key
`muse-glimmer.logit_scale` at load, fail-closed on absence
(`config.rs:190-192`), and the pinned artifact's value is `0.196116`
[docs/release-provenance.md:822-823]. Numerically that is `1/√26 ≈
0.1961161…`; the only place a `1/√26` expression appears in the tree is a
fixed Metal unit test — `kernels.encode_scale_softcap(encoder, &logits,
1.0 / 26.0f32.sqrt(), 20.0)` (`crates/muser-engine/src/metal.rs:68`) —
which pins the *formula's* behavior, not the model's constant. Whether the
checkpoint's authors derived the value from `1/√26` (26 is, coincidentally
or not, the number of Q-blocks per LM-head row, §20.3) is
**[unverified]**. The chapter's rule: the engine reads the number; the
book cites the metadata; nothing else is claimed.

## 20.6 The access pattern

Two questions to hold while reading the byte bill below. What does this stage
cost in traffic? And what did the parity decision of the previous section
actually charge us for it?

```
  LM head:
    read  W_lm (Q5_K)   202,048 × 4,576 B      = 924,571,648 B  ≈ 924.6 MB
    read  normed        6,656 × 4 B            =      26,624 B  (cache-resident)
    write logits        202,048 × 4 B          =     808,192 B  ≈ 789 KiB

  Soft cap (serving route, four unary nodes):
    read + write logits, four times over       = 4 × 1,616,384 B ≈ 6.16 MiB
    (fused fallback: one read + one write      =    1,616,384 B ≈ 1.54 MiB)
```

The four-node decomposition re-touches the logits buffer four times —
about 4.6 MiB of extra traffic per token versus the fused form. Against
the 924.6 MB weight stream that produced those logits, that is ~0.5 % of
the stage's traffic: the byte cost of byte-exactness, paid deliberately
(§20.4's comment). This is the same trade as [Ch 19](19-downproj-and-residual.md)'s
104 rejected groups, in miniature and in the *other* direction — here
Muser adds dispatches to preserve bits rather than refusing to add them.

For the per-token budget: the head's 924.6 MB is 5.52 % of the artifact,
paid once per token — comparable to the whole 52-layer attention-block
share and second only to the FFN family. Its arithmetic intensity is the
same one-MAC-per-weight-byte decode shape as every other matvec
([Ch 1](01-why-inference-is-a-memory-problem.md)).

## 20.7 Tradeoffs

**Soft cap on vs off — what it does to comparisons.** Order-preserving,
so greedy decoding is unchanged (§20.2). But the cap is a *compressive*
map: gaps between large logits shrink. Worked by hand at the 20 cap:

```
  raw scaled logits a = 12.0, b = 6.0        gap = 6.00
  capped:  20·tanh(0.6) = 10.741
           20·tanh(0.3) =  5.826             gap = 4.915  (−18 %)
  raw scaled logits a = 40.0, b = 20.0       gap = 20.0
  capped:  20·tanh(2.0) = 19.281
           20·tanh(1.0) = 15.232             gap = 4.049  (−80 %)
```

Two consequences follow, one local and one that reaches across the book.

The local one is about probabilities. For softmax over capped logits, and for
logprobs, the transform is emphatically not a no-op: probabilities come out
flatter than an uncapped engine would produce from the same hidden state, and
any cross-engine comparison of logprobs or bounded-logit deltas is only
meaningful if both engines apply the same scale-and-cap in the same order.
That is why Muser's parity work pins the *post-softcap public bytes* rather
than anything upstream of the cap — the `reference.rs` `result_output`
capture and the four-node decomposition of §20.4 both aim at that same pinned
surface.

The far-reaching one is about units, and the reason it matters for the
handoff is this: a tolerance is meaningless until you know which numbers it
was measured on. The disaggregated lane of
[Ch 32](32-precision-across-the-handoff.md) accepts or rejects a remote
engine's work through "declared bounded-logit policies" — acceptance rules of
the form *max delta < 11, mean < 1.25*. Those thresholds were measured on
exactly these capped logits, and we kept the run that produced them: the
wizard's native-lane rule
[nvfp4-fast-lane-evidence-20260817 §Determinism; ledger wizard attempt 9].
So the cap is part of the contract's units, not a cosmetic tail step. Without
it the same deltas would be much larger and the tolerance would have to be
re-derived.

**Q5_K for the head — 5.5 bits on the score-setter.** The head is the one
tensor that decides, by small margins, which token wins; the artifact
spends 5.5 bits/element here against 4.5 on the Q4_K bulk. A Q4_K head at
the same shape would be 202,048 × 26 × 144 = 756,467,712 B ≈ 756.5 MB —
the Q5_K choice costs +168.1 MB (+22.2 %) per token of extra read
(arithmetic from §20.3's blocks-per-row). That the precision lands here
*because* small score differences decide the token is the recipe's
rationale, inherited from the quantized artifact, not re-measured here
**[unverified]** — the same honesty note as the ancestor's Q6_K LM head
[ferrite-book Ch 19].

**Four unary nodes vs one fused kernel.** The war story is told above; here
is only its price tag, from §20.6: ~4.6 MiB/token plus three extra dispatches
and barriers, bought to make the public bytes match llama.cpp's. The
receipt for the rejected direction stays where a future maintainer will trip
over it, in the source comment itself (`lmhead.rs:243-246`). Under the J0
anchor — llama's own bytes as the gate
[ledger Stage A; see [Ch 38](38-measuring-against-llama-cpp.md)] — the
ranking is not close: parity outranks dispatch count.

**Why not vocab-block or resident layouts?** The ancestor book explored
vocab-blocked LM-head layouts [ferrite-book Ch 19]; Muser's engine keeps
the plain row-major head and the pinned matvec, and spends its layout
ingenuity elsewhere (KV planes, [Ch 15](15-kv-store-and-the-ring.md)).
No retained Muser measurement compares LM-head layouts **[unverified]**;
the pinned-kernel parity argument (same kernel as the comparator) is the
documented reason the simple route stays.

## 20.8 Where the gap lives

**Not the gap — with one instrumentation lesson attached.** In the
one-token closure accounting, the LM head and softcap sit in the "common
math closures" family: 406 production vs 406 legacy, delta 0
[docs/decode-dispatch-gap-20260815.md]. The serving route's four-node
softcap even *adds* work relative to the fused legacy form and stays:
exactness is the constraint, dispatch count is not.

The lesson is in how that verdict was reached. We went into the gap
investigation expecting to read the head's cost straight off the retained
baseline, and the first thing the baseline told us was false: "production
labels omitted `lm_head`, so its time was attributed to the following
`softcap` label" [same doc, Instrumentation correction]. The head's time was
never missing — it was wearing the next stage's name, which is the worst way
for a measurement to be wrong, because the total still adds up. A 924.6 MB
stage is exactly the kind of thing a label defect hides in plain sight. So
the investigation's first act was fixing its own instrument, and only then
did it draw a conclusion about the engine: measure the instrument before the
engine ([Ch 38](38-measuring-against-llama-cpp.md) formalizes this culture).

The logits exist — 202,048 capped scores in a buffer on the GPU. The
model has an opinion about the next token; nothing has *chosen* one yet.
Choosing is a different kind of computation — a reduction, a policy, and
in Muser's case a deliberate trip back to the CPU — and it is the final
chapter of the decode walk, [Ch 21](21-sampling-argmax-and-grammar.md).

---

## References

- `crates/muser-engine/src/shaders/muse_reference.metal:15-25` —
  `muser_scale_softcap_inplace`, the fused scale+tanh kernel (fallback
  route).
- `crates/muser-engine/src/metal/encode/lmhead.rs:163-267` —
  `encode_scale_softcap` / `encode_scale_softcap_count` /
  `encode_scale_softcap_legacy`, including the four-unary-node serving
  route and the "equal pre-softcap logits did not yield equal public
  bytes" comment (`:242-247`); `:36-80` `encode_ggml_unary_inplace`.
- `crates/muser-engine/src/metal/encode.rs:288-289` — ggml unary scale
  (op 10) and tanh (op 100) PSO registration from the pinned metallib.
- `crates/muser-engine/src/decode.rs:5869-5876` — final norm fused into
  layer 51's tail; `:5892-5905` LM head + softcap dispatches;
  `:5229-5241` the batch route's separate final norm.
- `crates/muser-engine/src/config.rs:190-197` — `logit_scale` (required
  GGUF key `muse-glimmer.logit_scale`) and `final_logit_softcap`
  (`muse-glimmer.final_logit_softcapping`, llama.cpp 30.0 default, this
  checkpoint 20.0).
- `crates/muser-engine/src/metal.rs:68` — the fixed test's
  `1.0 / 26.0f32.sqrt()` (formula pin, not the model constant).
- `crates/muser-engine/src/reference.rs:549-574` — the CPU oracle tail:
  final norm, head, scale, cap, `result_output` capture.
- `crates/muser-bench/src/m16.rs:195-197` — `lm_head 6656->202048 q5k`
  shape/dtype evidence.
- [docs/release-provenance.md:822-823] — `logit_scale=0.196116` with
  `final_logit_softcapping=20` on the qualified dumps.
- [docs/decode-dispatch-gap-20260815.md] — common-math closure accounting
  and the lm_head label-defect correction.
- [ledger wizard attempt 9 / nvfp4-fast-lane-evidence §Determinism] — the
  bounded-logit native-lane rule (max < 11, mean < 1.25) measured on
  softcapped logits; see [Ch 32](32-precision-across-the-handoff.md).
- [Ch 6](06-the-kquant-family.md) — the Q5_K 176-byte block behind
  §20.3's arithmetic; [Ch 11](11-token-embedding-lookup.md) the embedding
  the head inverts; [Ch 13](13-the-qkv-gate-matvec-family.md) the pinned
  matvec at its smaller shapes.
- [Ch 32](32-precision-across-the-handoff.md) — bounded-logit policies
  over the handoff; [Ch 38](38-measuring-against-llama-cpp.md) the J0
  anchor and instrument-first culture.
- [ferrite-book Ch 19] — the ancestor's LM-head chapter: "17× more rows",
  the one-shot-bandwidth framing, and the Q6_K-head precision note, all
  re-derived here with Muse's Q5_K geometry.
