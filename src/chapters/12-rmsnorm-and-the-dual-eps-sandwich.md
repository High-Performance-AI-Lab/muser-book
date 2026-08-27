# Chapter 12 — RMSNorm and the dual-epsilon sandwich
> **status:** polished  ·  **path:** Muse Glimmer, pinned Muser tree
>
> *Prerequisites: [Ch 2](02-metal-compute-model.md) (SIMD groups,
> `simd_sum`, `threadgroup_barrier`), [Ch 9](09-muse-glimmer-architecture.md)
> (the sandwich-norm architecture), [Ch 10](10-the-forward-pass-at-a-glance.md)
> (the per-layer kernel chain), [Ch 11](11-token-embedding-lookup.md) (the
> residual stream's birth). This chapter defines normalization from zero; no
> transformer-paper background is assumed.*

Chapter 11 ended with the residual stream freshly born — 6,656 f32s from
one quantized table row — and named its first consumer: the entry RMSNorm.
This is the norm story, and on Muse Glimmer it is an unusual one: two
different epsilons, a llama.cpp constant the checkpoint does not carry, and
a fused kernel that exists because fusing carelessly would change the
model's public numbers.

---

## 12.1 What it computes

[RMSNorm](../glossary.md#rmsnorm) takes a vector `x` of length `n` and produces
a vector of the same length whose magnitude is predictable — close to unit
scale — regardless of how large or small `x` was:

```
mean(x²) = (1/n) Σⱼ xⱼ²                    (mean of squares)
rms      = √(mean(x²) + ε)                 (root-mean-square, ε-guarded)
y_i      = x_i / rms × γ_i                 (normalize, then per-channel gain)
```

On Muse Glimmer `n = hidden_dim = 6,656` for the stream norms
(`crates/muser-engine/tests/muse_golden.rs:96`), `γ` is a learned
per-channel weight (`attn_norm.weight`, `ffn_norm.weight`, …,
`config.rs:300-314`), and — the reason this chapter exists — **ε is not one
value but two**: `1e-5` for every norm the GGUF declares, and a hard-coded
`1e-8` for the two "post" norms of the sandwich, a llama.cpp graph constant
that the checkpoint does not carry at all (`config.rs:23-28`). One kernel in
this chapter, `muser_fused_norm_residual_rms_norm_32sg`, exists precisely
because a single-epsilon fused kernel is *not valid* for this model
(`shaders/ferrite/rmsnorm_batch_tail.metal:142-146`).

## 12.2 Why it exists — 52 layers of multiplication drift

A transformer layer is a chain of multiplies. The [matvec](../glossary.md#matvec)
of [Ch 13](13-the-qkv-gate-matvec-family.md), an activation, another matvec —
and at every step the numbers on the [residual
stream](../glossary.md#residual-stream-hidden-state) get multiplied together.
Multiplications compound. If each of Muse Glimmer's 52 layers scaled the
stream by 1.2, after the stack the magnitudes have grown by
`1.2^52 ≈ 39,000`. If each scaled by 0.8, the signal has shrunk to
`0.8^52 ≈ 1.4 × 10⁻⁵`. Either way the network drifts out of a usable range,
and the deeper layers see garbage. Normalization is the valve: it pins the
magnitude back to a known band before each sub-block, so the layer stack
stays numerically stable. Skip it and the model still *runs* but its
late-layer arithmetic saturates or vanishes — on this model, exact-parity
decode would be the first casualty, because every ULP that drifts here is
amplified by everything downstream.

Muse Glimmer is a Gemma-2-style *sandwich*: each layer normalizes **four
times**, not two (`config.rs:300-314` names all four tensors):

```
  attn_norm (1e-5) → attention → post_attention_norm (1e-8) → ffn_norm (1e-5)
                                 → FFN        → post_ffw_norm    (1e-8)
                                 → next layer's attn_norm        (1e-5)
```

The two "post" norms normalize each sub-block's *output* before it is added
to the residual; the "pre" norms normalize the stream before each sub-block
reads it. That is the sandwich — pre-norm, block, post-norm, residual — and
the two epsilons live on opposite sides of it.

### Why RMS and not standard deviation?

**LayerNorm** centers *and* scales: it subtracts the mean and divides by
the [standard deviation](../glossary.md):

```
μ = (1/n) Σ xⱼ ;  σ² = (1/n) Σ (xⱼ − μ)² ;  y_i = (x_i − μ)/√(σ²+ε) × γ_i
```

**RMSNorm** drops the mean subtraction and divides by the root-mean-square
only. Two reasons, one costed and one empirical:

1. **Cost.** LayerNorm needs two global reductions over `x` — one for `μ`,
   one for `σ²` (which itself needs `μ`). RMSNorm needs *one*: the sum of
   squares. On a GPU a reduction is the expensive part of normalization —
   it is the one place every thread must wait for every other thread — so
   one reduction instead of two is a real saving, repeated 100+ times per
   token on this model.
2. **Empirical equivalence.** The RMSNorm paper reports that dropping the
   mean changes downstream quality negligibly across the models tested
   `[arxiv:1910.07467]`. **[unverified]** for Muse Glimmer specifically —
   that is the paper's claim, not a measurement in this campaign; the
   architecture chose RMSNorm upstream and Muser implements what the GGUF
   declares.

A useful identity links the two: `mean(x²) = σ² + μ²`, so
`rms = √(σ² + μ²) ≥ σ`, with equality exactly when `μ = 0`. When `x` is
already mean-centered, dividing by `rms` and dividing by `σ` are the same
operation — the sense in which the two norms' scaling steps coincide.

### The two epsilons, precisely

- `rms_eps = 1e-5` is read from the GGUF key
  `muse-glimmer.attention.layer_norm_rms_epsilon`; absence is a hard load
  error (`config.rs:184-188`). It feeds every norm this chapter dispatches
  *except* the two post norms.
- `post_norm_eps = 1e-8` is **not in the GGUF**. llama.cpp's graph builder
  hard-codes it (`src/models/muse-glimmer.cpp:67`,
  `const float post_norm_eps = 1e-8f;`), and Muser mirrors the constant with
  a doc comment saying exactly that (`config.rs:23-28`). This is a rare
  thing: a numerical constant that is *part of the comparator's identity*
  rather than the checkpoint's. An engine that read `1e-5` everywhere —
  the "obvious" simplification — would disagree with pinned llama.cpp in
  the last bits of every post-norm, and the parity gates of
  [Ch 38](38-measuring-against-llama-cpp.md) exist to catch exactly that
  class of drift.

The ε inside the root has one job: if `x` is all zeros, `mean(x²) = 0` and
`√0 = 0`, which would divide by zero. `+ ε` keeps the denominator strictly
positive. Why *two different* tiny values matter numerically is harder to
say — the difference between `√(m + 1e-5)` and `√(m + 1e-8)` is invisible
for any `m` of real magnitude — but the exactness contract does not grade
on "of real magnitude": it grades on bits, and llama's graph says 1e-8,
so 1e-8 it is. The deeper motivation for the sandwich itself is a
model-design question this engine inherits `[unverified]`.

## 12.3 RMSNorm by hand: `x = [3, 4, 0, 0]`, ε = 1e-5

Take `n = 4`, `x = [3, 4, 0, 0]`, `ε = 1e-5`, and first `γ = [1,1,1,1]`:

```
 Step 1 — sum of squares
   x² = [9, 16, 0, 0] ;  Σx² = 25 ;  mean(x²) = 25/4 = 6.25

 Step 2 — root-mean-square and its reciprocal
   rms     = √(6.25 + 0.00001) ≈ 2.500002
   inv_rms = 1/rms            ≈ 0.3999997   (call it 0.4)

 Step 3 — normalize, then apply γ
   y_i = x_i × inv_rms × γ_i
   y   = [3×0.4×1, 4×0.4×1, 0, 0] = [1.2, 1.6, 0, 0]
```

*Figure 12.1: RMSNorm worked example. The input whose RMS was 2.5 comes out
with RMS 1 (before γ).*

Check: `mean(y²) = (1.44 + 2.56)/4 = 1`, so `rms(y) = 1`. Now change γ to
`[2, 0.5, 1, 1]`: `y = [2.4, 0.8, 0, 0]` — the RMS is no longer 1, *on
purpose*. Normalization fixes the scale; γ re-learns the shape. On Muse
Glimmer, one special γ is *programmer-chosen rather than learned*: the entry
norm binds an all-ones vector (`decode.rs:1216`), because the graph needs
the RMSNorm operation but the model has no entry-norm tensor.

Everything the kernels do below is a parallel implementation of those three
steps, at `n = 6,656`.

## 12.4 The Metal kernels

Four kernels from two lineages serve the stream norms. Read them in order of
increasing specialization.

### 12.4.1 `rms_norm_batch` — the ferrite-lineage base kernel

The unfused workhorse, byte-for-byte from the Ferrite shader pull at
`a85048a90` (`docs/extraction-manifest.md`):

```metal
// crates/muser-engine/src/shaders/ferrite/rmsnorm_batch_tail.metal:1
kernel void rms_norm_batch(
    device const float* x      [[ buffer(0) ]],  // [B × n]
    device const float* weight [[ buffer(1) ]],  // [n] (shared)
    device       float* out    [[ buffer(2) ]],  // [B × n]
    constant     uint&  n      [[ buffer(3) ]],
    constant     float& eps    [[ buffer(4) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint tid [[ thread_index_in_threadgroup ]],
    uint sgitg [[ simdgroup_index_in_threadgroup ]],
    uint lid [[ thread_index_in_simdgroup ]],
    threadgroup float* shared [[ threadgroup(0) ]])
{
    const uint batch = tgid;
    device const float* xb  = x   + batch * n;
    device       float* ob  = out + batch * n;
    device const float4* xb4 = (device const float4*)xb;
    device const float4* wb4 = (device const float4*)weight;
    device float4* ob4 = (device float4*)ob;
    const uint n4 = n >> 2u;

    float sum_sq = 0.0f;
    for (uint i = tid; i < n4; i += 128u)
        sum_sq += dot(xb4[i], xb4[i]);
    sum_sq = simd_sum(sum_sq);
    if (lid == 0u) shared[sgitg] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u)
        shared[4] = rsqrt((shared[0] + shared[1] + shared[2] + shared[3]) / float(n) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = shared[4];
    for (uint i = tid; i < n4; i += 128u)
        ob4[i] = xb4[i] * inv_rms * wb4[i];
}
```

This is the reduction pattern of [Ch 2](02-metal-compute-model.md) in
miniature, and every kernel in this chapter repeats it:

1. **Vectorized load** — the `float4` cast reads four floats per
   transaction; `n4 = n/4` iterations instead of `n`.
2. **Strided partial sum** — thread `tid` walks `i = tid, tid+128, …`,
   accumulating `dot(xb4[i], xb4[i])` (a 4-way sum of squares per load).
3. **`simd_sum`** — one hardware instruction collapses 32 lanes to their
   sum, no barrier, in-lockstep.
4. **Threadgroup handoff** — lane 0 of each of the 4 SIMD groups publishes
   its group's partial into `shared[sgitg]`; **one barrier** makes all four
   visible.
5. **Single-thread combine** — thread 0 sums the four partials and computes
   the reciprocal root; **a second barrier** broadcasts it.
6. **Normalize-and-scale** — every thread re-walks its elements and writes
   `x × inv_rms × γ`.

Two barriers per row is the price of crossing SIMD-group boundaries;
`simd_sum` inside a group is free. Note `rsqrt(...)` — that single function
name is load-bearing, and §12.7 comes back to it.

### 12.4.2 The pinned ggml norm — `kernel_rms_norm_mul_f32_4`

On the serving path the standalone norm almost never runs, because
`encode_rms_norm_mul` prefers the **pinned llama.cpp metallib** kernel
whenever the library is loaded and `n` is a multiple of 4:

```rust
// crates/muser-engine/src/metal/encode/norm.rs:244
pub fn encode_rms_norm_mul(
    &self, encoder: &ComputeCommandEncoderRef,
    input: &GpuBuffer, weight: &GpuBuffer, output: &GpuBuffer,
    dim: usize, eps: f32, rows: usize,
) {
    // …(cross-vendor strict branch elided — §12.7)…
    if dim.is_multiple_of(4) {
        if let Some(pipeline) = self.ggml_rms_norm_mul() {
            self.encode_ggml_rms_norm(
                encoder, pipeline, input, weight, weight, output, dim, eps, rows, false,
            );
            return;
        }
    }

    self.bind(encoder, "rms_norm_batch");
    // …(buffers 0–2, dim, eps; elided)…
    encoder.set_threadgroup_memory_length(0, 32);
    encoder.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(128, 1, 1));
}
```

The ggml pipeline is `kernel_rms_norm_mul_f32_4` from the metallib pinned by
`MUSER_GGML_METALLIB` (`metal/encode.rs:287`, loaded per
[Ch 4](04-pso-and-three-kernel-sources.md)), and `encode_ggml_rms_norm`
(`norm.rs:53-94`) packs llama's own `GgmlMetalKargsNorm` argument struct and
picks llama's own thread count — `(dim/4).next_power_of_two().clamp(32,
1024)`, which is **1,024 threads** at `dim = 6,656` (`norm.rs:89`). The
ferrite-lineage `rms_norm_batch` at 128 threads is the fallback when the
metallib is absent. Same math, two reduction shapes — and since
floating-point addition is not associative, the two shapes can differ in the
last bits. On the serving route the pinned one is the point.

### 12.4.3 The dual-eps fused tail — `muser_fused_norm_residual_rms_norm_32sg`

The heart of the chapter. After each sub-block, the graph must do *three*
things to the stream: add the block's post-norm-normalized output into the
residual, then produce the next sub-block's pre-normed input. The fused
kernel does all three in one dispatch, with **two different epsilons**:

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

Read it as §12.3 twice, with a residual add between:

- **Pass 1** reduces `src` (the sub-block's output, e.g. the o_proj result)
  and computes `inv_src` with **eps1 = 1e-8** — the post-norm.
- **Pass 2** does the residual update `hidden += src × inv_src × weight1`
  *and*, in the same loop, accumulates the sum of squares of the *updated*
  hidden — the read of `hidden` and the write and the reduction share one
  pass over the data.
- **Pass 3** computes `inv_hidden` with **eps2 = 1e-5** — the next pre-norm
  — and writes the next sub-block's input.

The reduction here spans **32 SIMD groups** (1,024 threads), not 4: lane 0
of each group writes `shared[sgitg]`, and thread 0 sums 32 slots serially.
Why 32 groups for 6,656 elements? The dispatch wrapper's comment says it
outright:

```rust
// crates/muser-engine/src/metal/encode/norm.rs:236
// 32 SIMD groups keep the 6,656-wide Muse tail resident and match the
// accepted Ferrite geometry. 33 floats are padded to Metal's 16-byte
// dynamic-threadgroup-memory alignment.
encoder.set_threadgroup_memory_length(0, 144);
encoder.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(1024, 1, 1));
```

`n4 = 6,656/4 = 1,664` float4s; 1,024 threads means each thread owns one or
two float4s — the whole vector stays resident in registers with no second
strided loop iteration for most threads. The threadgroup allocation is 33
floats (32 partials + 1 broadcast slot) = 132 bytes, padded to 144 for
Metal's 16-byte alignment. This kernel is a Muser addition to the Ferrite
file — the `muser_` prefix marks it — because the ancestor had no
dual-epsilon model to need it.

### 12.4.4 The exact batch-graph tail — `…_batch_dual_eps`

One more variant, and the distinction between the last two is the most
important routing fact in this chapter. The serving *batch* graph uses
`muser_fused_norm_residual_rms_norm_batch_dual_eps`
(`rmsnorm_batch_tail.metal:250`), which does the same triple work but goes
out of its way to reproduce the two **pinned ggml** reductions bit for bit —
32-SIMD-group reduction, `1.0f / sqrt(...)` instead of `rsqrt`, llama's
exact multiply/add expression shape, and even a forced
`threadgroup_barrier(mem_flags::mem_device)` between the two norms to
reproduce the f32 device-memory publication boundary of the split route
(`rmsnorm_batch_tail.metal:277-303`). Its comment records the stakes:
"changing any of these moved public logprobs beyond the contract in the
rejected four-group fusion." The call sites confirm the split:
`decode.rs:4513`/`4656` (batch graph) take the exact variant;
`decode.rs:5807`/`5878` (the legacy one-token `encode_token` graph this
book narrates) take the 32sg variant.

## 12.5 The Rust dispatch — where the norms run in one token

From `encode_token` (`decode.rs:5515-5906`), the norm dispatches of one
token are:

| # | call | wrapper (file:line) | kernel | ε |
|---|------|--------------------|--------|---|
| 1 | entry norm | `encode_rms_norm_mul` `decode.rs:5534-5544` | ggml pinned (or `rms_norm_batch`) | 1e-5 |
| 2 | layer-0 attn norm | `encode_rms_norm_mul` `decode.rs:5553-5565` | same | 1e-5 |
| 3 | post-attn + FFN pre-norm, ×52 | `encode_fused_norm_residual_rms_norm_32sg` `decode.rs:5806-5818` | `…_32sg` | 1e-8 then 1e-5 |
| 4 | post-FFN + next attn norm, ×52 | same wrapper `decode.rs:5877-5889` | `…_32sg` | 1e-8 then 1e-5 |

Layers 1..51 receive their attention pre-norm *fused into the previous
layer's tail* — the `(next_norm, next_output)` selection at
`decode.rs:5869-5876` binds the next layer's `attn_norm` as the tail's
second weight, and for layer 51 it binds `output_norm` and writes the final
norm straight to the buffer the LM head reads. Layer 0's attn norm is the
only standalone pre-norm in the graph, plus the entry norm before it (whose
γ is the all-ones vector of §12.2). Counting dispatches per token: 2
standalone + 2×52 fused = **106 norm-carrying dispatches**, each folding
two norms in the fused case — the number the gap accounting of §12.8
reconciles. On top of these, the per-head QK-norms of
[Ch 14](14-qk-norm-and-rope.md) add 2 dispatches per layer (104 per token)
through the same `encode_rms_norm_mul` wrapper at width `head_dim = 128`
(`decode.rs:5599-5618`) — same math, tiny `n`, per-head γ that turns out to
be a constant broadcast. That is the preview; Ch 14 owns the full story.

## 12.6 The access pattern

One row of one stream norm at `n = 6,656`: read `x` (26,624 B) + two γ
vectors (26,624 B each for the fused tail) and write `hidden` (26,624 B) +
`output` (26,624 B) — roughly **106 KB per fused tail**, of which the γ
weights are read by all 52 layers' *different* tensors (52 × 2 × 26 KB of
distinct weights per token ≈ 2.7 MB) while the activations are re-touched
per layer. Across a token: ~2×52 fused tails ≈ 5.5 MB of activation traffic
plus ~2.7 MB of γ — against the ~16.76 GB artifact stream (`lib.rs:14`),
that is ~0.05 %. The norms are latency-motivated (keep 1,024 threads'
worth of the vector resident, cross as few barriers as possible), not
bandwidth-motivated. The γ bytes are the only *weight* traffic here, and
they are one-thousandth of the per-layer matvec weights of
[Ch 13](13-the-qkv-gate-matvec-family.md).

## 12.7 Tradeoffs

**The rejected 104-group fusion — the measured heart of this chapter.** The
obvious optimization is to fuse *more*: merge each layer's separated
norm-boundary dispatches (the +104 closures of §12.8) into the tails. It
was tried. The retained-activation hybrid preserved greedy tokens but
breached the public numerical contract: full-logit max absolute error
`4.6300888e-4`, normalized logprob max error `3.197146176834309e-4` — above
the `1e-4` contract — with 201,970 of 202,048 logits differing and the
first KV difference one f16 ULP in layer 1's value plane (bits 39,892 vs
39,893) `[docs/decode-dispatch-gap-20260815.md §Rejected hybrid
postmortem]`, receipts
`muser-receipt://pinned-token-parity-20260814-v{3,4}/`. It was
removed, not hidden behind a tolerance. The corrected fusion — the
`…_batch_dual_eps` kernel of §12.4.4, which reproduces the pinned ggml
reductions exactly — kept the baseline's full-logit SHA-256 and measured
39.274 ms GPU vs the 40.330 ms baseline (655 vs 760 closures), a single-run
diagnostic `[docs/decode-dispatch-gap-20260815.md §Landed and rejected
reductions]`. Lesson, in this campaign's voice: a fusion is eligible only
if it is *bit-exact*, and "almost the same reduction" is not.

**`rsqrt` vs `1.0f / sqrt` — one function name, contract-sized
consequences.** `rsqrt` is Metal's fast-math reciprocal square root — one
hardware instruction, a few ULP short of the IEEE-rounded result;
`1.0f / sqrt(x)` is fully rounded. The ferrite-lineage port
`rms_norm_llamacpp.metal` exists entirely to document and fix this
difference: its header explains that Ferrite's default `rsqrt(mean + eps)`
"differs from `1/sqrt` by a few ULP per call," that those ULPs "compound
into knife-edge logit flips past ~50 tokens," and that llama uses the
multi-simdgroup reduction *and* `1.0f / sqrt`
(`shaders/ferrite/rms_norm_llamacpp.metal:14-19, 96-101`). Muser's serving
answer is blunter: route the standalone norms through llama's *own*
metallib kernel (§12.4.2) and make the fused batch tail reproduce llama's
reduction bit for bit (§12.4.4). The 32sg kernel's `rsqrt` is one reason it
is confined to the legacy one-token graph — which, per the routing comment
at `decode.rs:2085-2091`, is itself confined to teacher-forced and
profiling duty because its fused kernels' rounding "diverges from the
source-pinned llama Metal graph enough to breach public logprob tolerance."

**One fused dispatch vs three split dispatches.** Unfused, each tail is:
post-norm (read `src`+γ1, write normed), residual add (read `hidden`+normed,
write `hidden`), pre-norm (read `hidden`+γ2, write output) — 5 reads and 3
writes over 26 KB buffers plus two extra dispatch boundaries. The fused
kernel: read `src`, γ1, γ2, `hidden`; write `hidden`, output — one pass
fewer over the residual, two fewer dispatches, ×104 per token. That is the
+104-closure saving the rejected fusion chased; the exact tail captures it
for the two norm pairs it covers without touching the reduction order.

**The cross-vendor seam — when the norm must split in two.** Under
`MUSER_CROSS_VENDOR_QK`, every norm wrapper here reroutes to a decomposed
*no-fast-math* pair — unweighted RMS in one dispatch, explicit
learned-weight multiply in a second, with a barrier between
(`norm.rs:25-50`, `norm.rs:115-146`). The reason is the disaggregated lane:
vLLM's producer materializes the weightless norm output in F16 and applies
its scale as a *second* F16 operation, so the seam-exact receiver must
preserve that intermediate rounding point (`norm.rs:280-284`). The fused
kernel "loses that intermediate rounding point and is therefore not
seam-exact." A fusion that is exact against llama is still wrong against
CUDA — exactness is always *relative to an anchor*.

## 12.8 Where the gap lives

This chapter is the **most direct resident of the gap** in Part IV. The
one-token dispatch-gap reconciliation counts **104 separated norm-boundary
closures** (52 post-attention + 52 post-FFN boundary pairs) in the
production graph's +196-closure delta — the largest single family
`[docs/decode-dispatch-gap-20260815.md §Corrected closure-count diff]`. And
the accounting's sharpest lesson lives here too: those 104 groups look like
pure waste, but every cheap removal tried changed bits (§12.7's 3.197e-4
logprob breach), so they are *kept*, classed "fusible adjacent ops —
existing fusion is not exact; reject." The gap survived bit-exactness until
the anchor itself changed (the J0/J1 story of
[Ch 38](38-measuring-against-llama-cpp.md) and
[Ch 40](40-what-we-measured-and-rejected.md)). When a later chapter says
"the norm boundary is the gap," this table row is what it means.

## 12.9 What comes next

The stream is normalized and sits in `activations.post_norm`. Four weight
matrices are about to read it — Q, K, V, and the attention gate — in the
one concurrent dispatch set where the token's real bandwidth bill starts
running. That is [Ch 13](13-the-qkv-gate-matvec-family.md), the hero
chapter of Part IV.

## References

- `crates/muser-engine/src/shaders/ferrite/rmsnorm_batch_tail.metal:1-33` —
  `rms_norm_batch`, the ferrite-lineage base kernel.
- `crates/muser-engine/src/shaders/ferrite/rmsnorm_batch_tail.metal:142-201`
  — `muser_fused_norm_residual_rms_norm_32sg`, the dual-eps fused tail.
- `crates/muser-engine/src/shaders/ferrite/rmsnorm_batch_tail.metal:250-322`
  — `muser_fused_norm_residual_rms_norm_batch_dual_eps`, the pinned-exact
  batch-graph tail (and its reproduced-publication comment at :277-303).
- `crates/muser-engine/src/shaders/ferrite/rms_norm_llamacpp.metal:1-37,
  49-109` — the bit-exact llama port; the rsqrt-vs-`1/sqrt` header and the
  accumulation-order commentary.
- `crates/muser-engine/src/metal/encode/norm.rs:244-278` —
  `encode_rms_norm_mul` (ggml-pinned preference, 128-thread fallback).
- `crates/muser-engine/src/metal/encode/norm.rs:53-94` — `encode_ggml_rms_norm`
  (kargs packing, llama's thread-count rule).
- `crates/muser-engine/src/metal/encode/norm.rs:163-241` — the 32sg wrapper
  and its 1,024-thread / 144-byte geometry comment.
- `crates/muser-engine/src/metal/encode/norm.rs:25-50, 280-303` — the
  cross-vendor split-norm seam (vLLM F16 boundary).
- `crates/muser-engine/src/metal/encode.rs:287` — `kernel_rms_norm_mul_f32_4`
  from the pinned metallib.
- `crates/muser-engine/src/config.rs:23-28` — `MUSE_POST_NORM_EPS = 1e-8` and
  the `muse-glimmer.cpp:67` provenance comment.
- `crates/muser-engine/src/config.rs:117-120, 184-188` — `rms_eps`/`post_norm_eps`
  fields; the required GGUF epsilon key.
- `crates/muser-engine/src/config.rs:300-314` — the four per-layer norm
  tensors asserted at load.
- `crates/muser-engine/src/decode.rs:5534-5544, 5806-5818, 5869-5889` — the
  entry norm, both fused-tail call sites, and the next-norm selection.
- `crates/muser-engine/src/decode.rs:2085-2091` — the routing comment that
  confines the 32sg graph to teacher-forced duty.
- `crates/muser-engine/src/decode.rs:1216` — the all-ones entry-norm γ.
- `[docs/decode-dispatch-gap-20260815.md]` — the +196 reconciliation (104
  norm-boundary groups), the rejected hybrid postmortem, and the
  landed-reductions table.
- `muser-receipt://pinned-token-parity-20260814-v3/`, `-v4/` —
  retained evidence for the rejected hybrid.
- `[arxiv:1910.07467]` — Zhang & Sennrich, *Root Mean Square Layer
  Normalization* (RMSNorm; the LayerNorm-equivalence claim).
- [Ch 2](02-metal-compute-model.md) — `simd_sum`, barriers, threadgroup
  memory.
- [Ch 14](14-qk-norm-and-rope.md) — the per-head QK-norm at width 128.
- [Ch 38](38-measuring-against-llama-cpp.md) — the parity gates the 1e-8
  constant protects.
- `[ferrite-book Ch 10]` — the ancestor's RMSNorm chapter (worked example
  and reduction pedagogy ported from it).
