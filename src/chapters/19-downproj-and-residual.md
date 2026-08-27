# Chapter 19 — The down projection + residual
> **status:** polished  ·  **path:** Muse Glimmer, pinned Muser tree
>
> *Prerequisites: [Ch 6](06-the-kquant-family.md) (Q4_K and Q6_K block
> layouts), [Ch 12](12-rmsnorm-and-the-dual-eps-sandwich.md) (the dual-eps
> sandwich and `muser_fused_norm_residual_rms_norm_32sg`),
> [Ch 13](13-the-qkv-gate-matvec-family.md) (the pinned ggml matvec family),
> [Ch 17](17-sigmoid-gate-and-oproj.md)–[Ch 18](18-swiglu-ffn.md) (the layer
> so far; `ffn_mid` is waiting). This chapter closes the layer — and then
> prices what closing it costs in dispatch groups, which is where this book's
> central tradeoff becomes concrete.*

---

## 19.1 What it computes

[Ch 18](18-swiglu-ffn.md) left a `[19,968]` vector in
`activations.ffn_gate` — the gated, activated FFN intermediate. Two
operations finish the layer:

```
1.  projected = W_down · ffn_mid                W_down : [6656 × 19968]
2.  residual += post_norm(projected)            (eps 1e-8)
    next_input = rms_norm(residual, next_norm)  (eps 1e-5)
```

Operation 1 is a [matvec](../glossary.md#matvec) — the last one of the layer,
and on some layers the most expensive single weight read in it (§19.7).
Operation 2 is the *second* fused dual-eps tail: the sandwich of
[Ch 12](12-rmsnorm-and-the-dual-eps-sandwich.md) paying off, where the same
kernel that adds the FFN delta into the residual also computes the **next
layer's** normalized input (or, after layer 51, the final norm that feeds
the LM head of [Ch 20](20-final-norm-lm-head-softcap.md)).

After operation 2, `activations.normed` holds the residual stream plus
layer `l`'s attention delta and FFN delta, normalized for layer `l+1` — and
the 52-layer loop takes its next turn.

## 19.2 Why it exists — closing the block and opening the next

The down projection is the FFN's exit: without it the layer would emit a
19,968-wide vector into a 6,656-wide stream — wrong shape, and the next
layer's attention would have nothing to read. The residual add is what
makes deep transformers deep: each layer *contributes* into a running sum
rather than replacing it, so the gradient path to early layers stays
near-identity (`∂(x + f(x))/∂x = 1 + f'(x)`). And the tail's second norm
exists because Muse Glimmer sandwiches every sub-block between a pre-norm
and a post-norm ([Ch 12](12-rmsnorm-and-the-dual-eps-sandwich.md)) — the
post-FFN norm (1e-8) scales the delta before it lands in the residual, and
the *next* pre-norm (1e-5) prepares the stream for the next attention.
Figure 19.1 lays out the whole layer with both tails.

```
  ┌────────────── layer l ────────────────────────────────────────────┐
  │  [Ch 17] attention ─► gate ─► o_proj ─► TAIL#1: residual +=       │
  │                                              post_norm(o) (1e-8); │
  │                                              ffn_in = norm (1e-5) │
  │  [Ch 18] ffn_in ─► gate·x, up·x ─► silu⊙ ─► ffn_mid [19968]       │
  │  [Ch 19] ffn_mid ─► W_down ─► TAIL#2: residual +=                 │
  │                                    post_ffn_norm(·) (1e-8);       │
  │                                    next_in = norm(residual, 1e-5) │
  └───────────────────────────────────────────────────────────────────┘
                                    │ after layer 51: next_in feeds
                                    ▼ the final norm → LM head (Ch 20)
```
*Figure 19.1: One layer, two tails. This chapter is the `W_down` matvec and
TAIL#2.*

## 19.3 The matrix operation — and the Q6_K wrinkle that is real here

The matvec is mechanically the family you know: 6,656 output rows, each a
dot product over 19,968 inputs. What distinguishes it is the *dtype mix*.
On the kquant release artifact, `ffn_down` tensors come in **both Q4_K and
Q6_K** — the verify-shape table lists `ffn_down-q4k 19968->6656 q4k` and
`ffn_down-q6k 19968->6656 q6k` side by side
(`crates/muser-bench/src/m16.rs:179-194`), and the quickstart warns that a
build without the pinned llama.cpp metallib "fails closed … because Q6_K
tensors route through" it (`docs/quickstart.md:16`). The per-layer split —
which of the 52 layers carry which variant — lives in the GGUF tensor
headers and is not recorded in the repo docs this book cites
**[unverified]**; what is verifiable is that both variants exist on live
paths and both dispatch through the pinned metallib.

The two formats in bytes, step by step (layouts from
[Ch 6](06-the-kquant-family.md)):

```
  W_down row = 19,968 inputs = 19,968/256 = 78 super-blocks
  Q4_K: 78 × 144 B = 11,232 B/row  →  6,656 × 11,232  =  74,760,192 B ≈  74.76 MB
  Q6_K: 78 × 210 B = 16,380 B/row  →  6,656 × 16,380  = 109,025,280 B ≈ 109.03 MB
                                        ─────────────────────────────────────
  Q6_K / Q4_K = 210/144 = 1.4583      →  +45.8 % bytes for the 6-bit format
```

So a Q6_K-down layer reads ~258.6 MB of FFN weights (gate 74.76 + up 74.76 +
down 109.03) against a Q4_K-down layer's 224.3 MB. The ancestor book's
Q4_K_M mix table [ferrite-book Ch 18] taught exactly this device — spend
extra bits on the projection whose output lands directly in the residual
stream — and Muse Glimmer's artifact realizes the same idea with its own
(per-layer, GGUF-internal) split. Whether the *quality* payoff justifies the
+45.8 % on the layers that take it is the checkpoint author's call, inherited
not measured **[unverified]**.

A hand-sized worked example of operation 2, since the tail is this chapter's
kernel. Take `n = 4`, `residual = [1, 2, 3, 4]`, `projected = [4, 0, 0, 0]`,
`post_weight = [1,1,1,1]`, `next_weight = [1,1,1,1]`, both eps tiny:

```
post_norm(projected, 1e-8):  rms = √(16/4) = 2  →  [2, 0, 0, 0]
residual +=                 →  [3, 2, 3, 4]
next_norm(residual, 1e-5):  rms = √((9+4+9+16)/4) = √9.5 ≈ 3.082
next_input                  →  [0.973, 0.649, 0.973, 1.297]
```

One kernel, two normalizations, one add — with a device-memory publication
*between* them that turns out to be load-bearing (§19.9).

## 19.4 The Metal kernel — `muser_fused_norm_residual_rms_norm_32sg`

The tail kernel, verbatim. It is the decode-only member of the sandwich
family [Ch 12](12-rmsnorm-and-the-dual-eps-sandwich.md) introduced; read it
here as the layer-exit machine:

```metal
// crates/muser-engine/src/shaders/ferrite/rmsnorm_batch_tail.metal:142
// Decode-only dual-epsilon tail fusion:
//   hidden += rms_norm(src, eps1) * weight1
//   output  = rms_norm(hidden, eps2) * weight2
// Muse uses eps1=1e-8 for sandwich post-norms and eps2=1e-5 for the
// following pre-norm, so the older single-epsilon batch kernel is not valid.
kernel void muser_fused_norm_residual_rms_norm_32sg(
    device float* hidden [[buffer(0)]],
    device const float* src [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const float* weight1 [[buffer(3)]],
    device const float* weight2 [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant float& eps1 [[buffer(6)]],
    constant float& eps2 [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_simdgroup]],
    threadgroup float* shared [[threadgroup(0)]]) {
    const uint n4 = n >> 2u;
    device float4* hidden4 = (device float4*)(hidden + row * n);
    device const float4* src4 = (device const float4*)(src + row * n);
    device float4* output4 = (device float4*)(output + row * n);
    device const float4* weight14 = (device const float4*)weight1;
    device const float4* weight24 = (device const float4*)weight2;

    float sum_src = 0.0f;
    for (uint i = tid; i < n4; i += 1024u)
        sum_src += dot(src4[i], src4[i]);
    sum_src = simd_sum(sum_src);
    if (lid == 0u) shared[sgitg] = sum_src;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float total = 0.0f;
        for (uint group = 0u; group < 32u; ++group) total += shared[group];
        shared[32] = rsqrt(total / float(n) + eps1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv_src = shared[32];
    float sum_hidden = 0.0f;
    for (uint i = tid; i < n4; i += 1024u) {
        const float4 value = hidden4[i] + src4[i] * inv_src * weight14[i];
        hidden4[i] = value;
        sum_hidden += dot(value, value);
    }
    sum_hidden = simd_sum(sum_hidden);
    if (lid == 0u) shared[sgitg] = sum_hidden;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float total = 0.0f;
        for (uint group = 0u; group < 32u; ++group) total += shared[group];
        shared[32] = rsqrt(total / float(n) + eps2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv_hidden = shared[32];
    for (uint i = tid; i < n4; i += 1024u)
        output4[i] = hidden4[i] * inv_hidden * weight24[i];
}
```

Three passes, two barriers apiece:

- **Pass 1 (sum of squares of `src`).** Every thread strides the row's
  `float4`s (`i += 1024`); each of the 32 SIMD groups reduces with
  `simd_sum`, writes one partial to `shared[sgitg]`; thread 0 sums the 32
  partials **in order** and posts `rsqrt(mean + eps1)` to `shared[32]`.
  This 32-partial, fixed-order reduction is the pinned llama.cpp shape —
  the dispatch-gap investigation found the earlier four-SIMD-group `rsqrt`
  variant "not exact" and replaced it with precisely this
  [docs/decode-dispatch-gap-20260815.md, "The corrected fusion"].
- **Pass 2 (the residual add — the in-place mutation).** Each thread
  computes `value = hidden + src·inv_src·weight1`, writes it **back into
  `hidden`**, and accumulates `dot(value, value)` for the second norm. This
  is where `hidden += …` physically happens: the same buffer is read,
  added into, and rewritten, `float4` by disjoint `float4`.
- **Pass 3 (the next norm).** With `inv_hidden` from the second reduction,
  `output[i] = hidden[i] · inv_hidden · weight2`. `weight1` is the layer's
  `post_ffw_norm`; `weight2` is the *next* layer's `attn_norm` (or the
  final `output_norm` after layer 51) — chosen by the caller (§19.5).

Safety of the in-place add, in full: **one threadgroup owns one row**
(grid = rows), and within the threadgroup each thread owns a disjoint
strided set of `float4` slots — no two threads touch the same `i` in any
pass, so no in-dispatch race on `hidden`. The barriers order the *shared*
`rsqrt` handoffs, not the data writes (each thread's pass-3 reads the same
slots it wrote in pass 2). Across dispatches, the consumer of `hidden`/
`output` is the next layer's first kernel, sequenced by the tracked-buffer
ordering of the single-encoder token graph ([Ch 17](17-sigmoid-gate-and-oproj.md)
§17.7; the taxonomy formalizes in [Ch 35](35-ordering-hazards-and-the-dispatch-gap.md)).
And the geometry note in the wrapper: 32 SIMD groups keep "the 6,656-wide
Muse tail resident" — 1,024 threads, one row per threadgroup
(`crates/muser-engine/src/metal/encode/norm.rs:236-240`), with 33 floats of
threadgroup memory padded to 144 bytes for alignment.

## 19.5 The Rust dispatch — the layer exit in source

The down projection is the stock `project` wrapper — same pinned ggml
matvec, dtype routed automatically (Q4_K **or** Q6_K `ffn_down` both land
on `kernel_mul_mv_q{4,6}_K_f32`, `qkv.rs:429-450` with the
`(144, 2)/(210, 2)` rows-per-group table of [Ch 13](13-the-qkv-gate-matvec-family.md)).
The interesting dispatch is what follows — the tail and its `next_norm`
selection:

```rust
// crates/muser-engine/src/decode.rs:5863
self.project(
    command,
    &layer.ffn_down,
    &self.activations.ffn_gate,
    &self.activations.projected,
);
let (next_norm, next_output) = if layer_index + 1 < cfg.n_layers {
    (
        &self.layers[layer_index + 1].attn_norm,
        &self.activations.post_norm,
    )
} else {
    (&self.output_norm, &self.activations.hidden)
};
dispatch(command, |encoder| {
    self.kernels.encode_fused_norm_residual_rms_norm_32sg(
        encoder,
        &self.activations.normed,
        &self.activations.projected,
        next_output,
        &layer.post_ffn_norm,
        next_norm,
        cfg.hidden_dim,
        cfg.post_norm_eps,
        cfg.rms_eps,
    );
});
```

Read the buffer wiring carefully — it is the residual-stream bookkeeping of
the whole graph in one call:

- `hidden` ← `activations.normed`: the running residual, updated in place.
- `src` ← `activations.projected`: the down-proj output (scratch, like
  o_proj's output in [Ch 17](17-sigmoid-gate-and-oproj.md)).
- `next_output` ← `activations.post_norm` for layers 0..50 (the next
  layer's normalized input), but **`activations.hidden` for layer 51** —
  after the last layer the tail's second norm *is* the final norm, writing
  the vector the LM head consumes ([Ch 20](20-final-norm-lm-head-softcap.md)
  picks it up there). That is what "fused into last tail
  (decode.rs:5875)" means: there is no separate final-norm dispatch on this
  route.
- `weight1` ← `layer.post_ffn_norm` (1e-8), `weight2` ← the *next* layer's
  `attn_norm` (1e-5) — the sandwich hand-off.

The wrapper itself (`norm.rs:163-241`) binds the six buffers, pushes `n`,
`eps1`, `eps2` inline, sets 144 bytes of threadgroup memory, and dispatches
`(rows, 1, 1) × (1024, 1, 1)` — one row per threadgroup, 32 SIMD groups.
The first tail of the layer (post-attention, `decode.rs:5806-5818`) is the
identical call with `post_attn_norm`/`ffn_norm` — same kernel, both
boundaries.

## 19.6 The access pattern

Down projection per layer:

```
  Q4_K: read W_down 74,760,192 B   read ffn_mid 79,872 B   write projected 26,624 B
  Q6_K: read W_down 109,025,280 B  (same activation traffic)
  weight : activation ratio ≈ 74.8 MB : 106 KiB ≈ 700:1  — pure weight stream
```

Tail per layer:

```
  read  src (projected)      26,624 B
  read+write hidden (normed) 26,624 B read + 26,624 B written (+ reread in pass 3)
  write output (post_norm)   26,624 B
  read  two weight vectors   2 × 6,656 f32 = 53,248 B
                          ≈ 181 KiB total — rounding error next to W_down
```

The layer's total weight read: 224.3 MB (Q4_K-down) or 258.6 MB (Q6_K-down),
of which the down projection is a third to 42 %. Across 52 layers the FFN
family is the plurality of the 16.76 GB artifact — the arithmetic of
[Ch 18](18-swiglu-ffn.md) §18.7 plus this chapter's down numbers.

## 19.7 Tradeoffs

**Q6_K vs Q4_K on the down projection — +45.8 % bytes on the layers that
take it.** The arithmetic of §19.3: 109.03 MB vs 74.76 MB per layer, a
deliberate precision spend on the last projection before the residual — the
same reasoning the ancestor's Q4_K_M mix table documented for Qwen
[ferrite-book Ch 18], realized differently here. Both engines pay it
equally (both read the same GGUF through equivalent pinned kernels), so it
is a quality-vs-bytes decision, not a parity hazard. The M16 verify-shape
bench measured the two side by side in its candidate sweep — Q4_K ffn_down
0.891 → 0.533 ms and Q6_K ffn_down 0.897 → 0.539 ms per dispatch under the
winning n32 tile [ledger Stage B close-out, L0 "Winner (n32)"] — which also
tells you the two formats cost nearly the same *time* per byte on this
GPU; the +45.8 % is a byte bill, not a kernel-efficiency bill.

**One fused tail vs three separate kernels.** Unfused, the layer exit is:
norm the projection (1e-8), add into the residual, norm the residual
(1e-5) — three dispatches and a materialized post-norm intermediate. The
32sg tail is one dispatch that never materializes the intermediate. But
note precisely *what* it preserves: the source comment at
`decode.rs:1328-1330` says the fused kernel "reproduces the two pinned ggml
f32x4 norm reductions **and their intervening f32 device-memory
boundary**" — pass 2 writes `hidden` to device memory and pass 3 reads it
back, exactly where llama.cpp's graph publishes between nodes. It is a
fusion of *dispatches*, not of *arithmetic boundaries*; that restraint is
why it can be exact at all, and the diagnostic split route survives behind
`MUSER_NO_FUSED_PREFILL_DUAL_NORM` (`decode.rs:1331`) as the control.

**The 104-group question — fusion rejected where it would matter most.**
This is the tradeoff this chapter exists for; §19.8 gives it its own
section with the numbers.

## 19.8 Where the gap lives — the 104 norm-boundary groups

The one-token dispatch-gap investigation reconciled the production
(serving) graph's 760 profiling closures against the legacy fused route's
564 — a difference of +196 — into four families plus one copy
[docs/decode-dispatch-gap-20260815.md, "Corrected closure-count diff at
position 2,048"]:

| family | production | legacy | delta | disposition |
|---|---:|---:|---:|---|
| Entry/attention norm boundary | 53 | 2 | +51 | fusion not exact; reject |
| SWA wrapped-ring staging | 39 | 0 | +39 | keep until bit-exact replacement |
| KV publication and attention | 104 | 52 | +52 | session structure, keep |
| Post-attention residual + FFN norm | 104 | 52 | +52 | fusion not exact; reject |
| Post-FFN residual + next/output norm | 53 | 52 | +1 | fusion not exact; reject |
| Last-row copy | 1 | 0 | +1 | removed, bit-exactly |

The three norm-boundary rows are the **104 separated norm-boundary groups**
(+51 +52 +1): they are *this chapter's* boundaries — the layer exits and
entries that TAIL#2 fuses on the teacher-forced route but that the serving
graph keeps separated, publishing each norm as its own node through the
pinned ggml kernels. The instrument's own label diff names them: production
carries `layer.*.post_attn_norm` + `layer.*.ffn_norm` where the legacy
route has one `post_attn_ffn_norm`, and `layer.*.post_ffn_norm` +
`output_norm` where legacy has `post_ffn_next_norm`
[docs/decode-dispatch-gap-20260815.md, label table].

**The fusion that would remove those 104 groups exists and was rejected
because it changes logprobs beyond contract.** The hybrid postmortem is
exact about how far beyond: reusing a retained-activation schedule with
fast fused boundaries preserved the greedy token but produced a full-logit
maximum absolute error of `4.6300888e-4`, a normalized-logprob maximum
error of `3.197146176834309e-4` against the **`1e-4` contract** — with
201,970 of 202,048 logits differing and the first KV divergence in *layer
1*, value plane element 524,115, one f16 ULP apart (bits 39,892 vs 39,893)
[docs/decode-dispatch-gap-20260815.md, "Rejected hybrid postmortem";
receipts `muser-receipt://pinned-token-parity-20260814-v{3,4}/`].
One rounding difference in the first layer's residual chain, amplified
through 51 more layers, breaks a public numerical commitment. The attempt
"was removed rather than hidden behind a tolerance or shipped as an
alternate route" — and the routing comment of [Ch 18](18-swiglu-ffn.md)
§18.6 (`decode.rs:2085-2091`) is the standing consequence: serving decode
takes the batch graph with the exact pinned kernels *because* the fused
boundaries breach tolerance.

What *was* taken: the one exact removal — the last-row copy, one closure
and one 6,656-element f32 copy, worth **−0.136 ms GPU (−0.34 %) in a
single-run diagnostic**, with no wall-time claim (the +4.380 ms wall sample
was submit/wait noise) [docs/decode-dispatch-gap-20260815.md, "Landed and
rejected reductions"]. The corrected exact dual-norm fusion (the 32sg
kernel's pinned-reduction form) matched the baseline's full-logit SHA-256
but was retained as "historical self-consistency only" — useful, not
sufficient; the five-sample streamed serving number after it was 28.290
tok/s against llama's 33.428 (ratio 0.8463), Stage A still open at that
point [same doc]. The gap closed later only when the anchor itself changed
— J0 made llama's own bytes the gate and J1 transplanted llama's attention
DAG [ledger Stage A close-out, Arc 1 of the campaign; the parity outcome is
the six-depth matrix at or above 1.0 of [Ch 38](38-measuring-against-llama-cpp.md)].

So the honest accounting for this chapter: **the layer boundary is where
the +196 lives, and the fix that looks obvious is the one the contract
forbids.** Every closure in those 104 groups does required math; the
investigation's own conclusion is that "no repeated closure performing
provably identical arithmetic was found" — the boundary cost is real but
the only cheap removals change bits, and changing bits is the one thing
Muser's public logprob contract cannot buy. [Ch 35](35-ordering-hazards-and-the-dispatch-gap.md)
carries the full hazard framing; [Ch 40](40-what-we-measured-and-rejected.md)
files this as the canonical measured rejection.

The loop is closed: 52 times you have watched a layer normalize, attend,
gate, project, feed forward, and fold its delta into the stream — and the
last tail wrote `activations.hidden` through the final norm. One question
is left in Part IV: what does the model actually *say*? The 6,656-vector
now sitting in `hidden` is about to meet the largest matvec in the engine —
202,048 output rows — and the soft cap that bounds what it may score.
[Ch 20](20-final-norm-lm-head-softcap.md).

---

## References

- `crates/muser-engine/src/shaders/ferrite/rmsnorm_batch_tail.metal:142-201`
  — `muser_fused_norm_residual_rms_norm_32sg`, the dual-eps tail kernel
  (primary source; the comment block states the eps split and why the
  single-epsilon kernel is invalid).
- `crates/muser-engine/src/metal/encode/norm.rs:163-241` — the 32sg
  wrapper and its batch form (1,024 threads, 144 B threadgroup memory,
  "6,656-wide Muse tail" geometry note); `:97-160` the batch dual-eps
  sibling; the strict decomposed route under `MUSER_CROSS_VENDOR_QK`.
- `crates/muser-engine/src/decode.rs:5863-5889` — down projection + tail +
  `next_norm` selection in `encode_token`; `:5806-5818` the first
  (post-attention) tail; `:1328-1334` the fusion-control flags;
  `:2085-2091` the serving-exactness routing comment.
- `crates/muser-engine/src/metal/encode/qkv.rs:429-450` — the one-token
  pinned-metallib matvec path (Q4_K/Q6_K rows-per-group table);
  `crates/muser-engine/src/metal/encode.rs:278-280` PSO registration.
- `crates/muser-bench/src/m16.rs:179-198` — `ffn_down-q4k` / `ffn_down-q6k`
  shape-dtype evidence for the mix; `docs/quickstart.md:16` the Q6_K
  fail-closed routing note.
- `crates/muser-engine/src/reference.rs:526-541` — the oracle's post-FFN
  norm, residual add, and `l_out-{il}` capture point.
- [docs/decode-dispatch-gap-20260815.md] — the 760/564 reconciliation, the
  104 norm-boundary groups, the rejected hybrid postmortem (3.197e-4 vs
  the 1e-4 contract), the −0.136 ms exact copy removal, and the 0.8463×
  Stage-A-open snapshot.
- [receipt `muser-receipt://pinned-token-parity-20260814-v3/`],
  [receipt `muser-receipt://pinned-token-parity-20260814-v4/`]
  — retained evidence for the rejected hybrid.
- [ledger Stage B close-out] — `docs/goal-parity-ledger-2026-08.md`, L0
  "Winner (n32)": the Q4_K/Q6_K ffn_down per-dispatch timings.
- [Ch 12](12-rmsnorm-and-the-dual-eps-sandwich.md) — the sandwich and the
  tail family's first appearance; [Ch 13](13-the-qkv-gate-matvec-family.md)
  the pinned matvec; [Ch 18](18-swiglu-ffn.md) the FFN budget this chapter
  completes.
- [Ch 35](35-ordering-hazards-and-the-dispatch-gap.md),
  [Ch 38](38-measuring-against-llama-cpp.md),
  [Ch 40](40-what-we-measured-and-rejected.md) — the downstream chapters
  that build on §19.8.
- [ferrite-book Ch 18] — the ancestor's Q4_K_M mix table and the
  "where the bytes are, not where the gap is" device, ported with Muse's
  real mix.
