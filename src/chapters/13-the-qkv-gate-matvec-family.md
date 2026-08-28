# Chapter 13 — The QKV + gate matvec family
> **status:** polished  ·  **path:** Muse Glimmer, pinned Muser tree
>
> *Prerequisites: [Ch 2](02-metal-compute-model.md) (SIMD groups, dispatch),
> [Ch 6](06-the-kquant-family.md) (Q4_K/Q5_K/Q6_K blocks and the pinned ggml
> `kernel_mul_mv_q*_K_f32` kernels), [Ch 9](09-muse-glimmer-architecture.md)
> (GQA 32:2, the sigmoid gate), [Ch 12](12-rmsnorm-and-the-dual-eps-sandwich.md)
> (the normed input these projections read). This is deliberately the longest
> kernel chapter so far: this one operation family is where the token's
> bandwidth bill is actually incurred, and where Muser's parity discipline is
> most visibly different from "write a fast kernel."*

---

## 13.1 What it computes

Chapter 12 left the stream normalized in `activations.post_norm`, ready
for the four weight matrices that read it. Before any code appears, ask
the question this chapter exists to answer: when a token spends its time,
where does the time go? Overwhelmingly, it goes here. This is the family
those matvecs belong to — and the place where the token's bandwidth bill
is actually incurred. A **matrix–vector multiply** — [matvec](../glossary.md#matvec), or GEMV — is:

```
y = W · x        y_i = Σ_{j=0..cols-1} W[i,j] · x[j]     for each output row i
```

`W` is a learned weight matrix, `x` the input vector, `y` the output. A
matvec is `rows` independent [dot products](../glossary.md#dot-product), one
per output row, each of length `cols`. The twist is that `W` is not f32:
it is packed as a [kquant](../glossary.md#kquant) format (Q4_K's 144 bytes per
256 elements, Q5_K's 176, Q6_K's 210 — [Ch 6](06-the-kquant-family.md)), so
every `W[i,j]` must be dequantized on the fly:

```
W[i,j] = d × sc_sub(j) × nibble(i,j) − dmin × m_sub(j)      (Q4_K form)
```

Muser never materializes a dequantized weight. The nibbles fly from DRAM
through the ALU into an accumulator without ever landing in a buffer.

Muse Glimmer runs **four** of these matvecs per layer, as one concurrent
dispatch set, all reading the same normed input (`decode.rs:5569-5598`):

| projection | shape (in × out) | output feeds |
|---|---|---|
| `attn_q.weight` | 6,656 × 4,096 | the 32 query heads |
| `attn_k.weight` | 6,656 × 256 | the 2 KV heads' keys |
| `attn_v.weight` | 6,656 × 256 | the 2 KV heads' values |
| `attn_gate.weight` | 6,656 × 4,096 | the sigmoid attention gate ([Ch 17](17-sigmoid-gate-and-oproj.md)) |

Those shapes are not arbitrary; they fall out of the geometry the
architecture chapter fixed. `hidden = 6,656` sets every
matrix's input side. `n_heads × head_dim = 32 × 128 = 4,096` sets the
output side of Q and of the gate. `n_kv_heads × head_dim = 2 × 128 = 256`
sets K's and V's, narrow because grouped-query attention shares each KV
head across many query heads. We read those extents off the golden test,
which is where the anchor is kept
`[crates/muser-engine/tests/muse_golden.rs:96-100]`, and
`config.rs:300-307` asserts them again at load, so a checkpoint whose
tensors disagree fails before it can produce a wrong number. Muse Glimmer is
*not* a fused-QKV architecture: there is no single `[6,656 × 4,608]` QKV
matrix. There are four independent tensors, dispatched as four independent
matvecs that share one read-only input and write disjoint outputs — "one
concurrent set," in the call site's own words (`decode.rs:5566-5568`). The
gate is the fourth, unusual member; [Ch 17](17-sigmoid-gate-and-oproj.md)
explains what it gates.

## 13.2 Why it exists — decode is matvec, not matmul

> **Matvec vs [GEMM](../glossary.md#gemm).** A GEMM multiplies a matrix by a
> *matrix* of many columns. Prefill is a GEMM: the whole prompt is one batch
> of tokens, so each weight byte is reused across the batch
> ([Ch 36](36-prefill-vs-decode-paths.md)). Decode is a matvec: there is
> exactly **one token**, the batch collapses to one column, and every weight
> byte is used exactly **once** before being discarded until the next token.

That single-use property is the whole memory problem of
[Ch 1](01-why-inference-is-a-memory-problem.md). With no reuse across a
batch, the work is compute-light (one multiply-accumulate per weight) and
memory-heavy (the entire matrix must stream from DRAM). Decode is
bandwidth-bound by construction, and the matvec family *is* the workload
that streams the weights: strip the norms, RoPE, attention, and activations
out of a token and what remains, line for line, is a sequence of quantized
matvecs — these four, the o_proj, the FFN's three, and the LM head.

## 13.3 The matrix operation, explained from zero

If you have not multiplied a matrix by a vector recently, do this 2×2 by
hand (Figure 13.1). Take

```
   W (2×2)           x (2×1)        y (2×1)
  ┌          ┐       ┌     ┐        ┌     ┐
  │  1    2  │   ×   │  1  │    =   │  y₀ │
  │  3    4  │       │  5  │        │  y₁ │
  └          ┘       └     ┘        └     ┘

  y₀ = 1·1 + 2·5 = 11
  y₁ = 3·1 + 4·5 = 23
```

*Figure 13.1: a 2×2 matvec by hand. Two dot products of length 2; two
outputs. Every matvec in this chapter is this, with `cols = 6,656` and
`rows` from 256 to 202,048, and with `W`'s entries unpacked from nibbles.*

Now the storage fact that shapes every kernel here: GGUF weight matrices
are **row-major** — row 0's elements are contiguous, then row 1's, and so
on. One output row's weights are therefore one *contiguous span* of DRAM,
and contiguous spans are what DRAM serves efficiently. The naive mapping —
one thread per output row, walking all `cols` bytes serially — is correct
but starves the memory system: a single thread cannot keep enough loads
[in flight](../glossary.md) (memory requests the hardware has accepted but
not yet answered) to saturate the bus. Every real
kernel in this family instead splits each dot product across the 32 lanes
of a [SIMD group](../glossary.md#simd-group) and splits rows across
threadgroups. The exact split differs per kernel; the shared skeleton is
"dequant-and-MAC in the inner loop, `simd_sum` to combine the 32 partials,
lane 0 writes the output row."

One more piece of vocabulary you need for §13.4: ggml's kargs. llama.cpp's
Metal kernels do not take `rows`/`cols` as friendly scalars — they take a
packed C struct of tensor extents and byte strides (`ne00`, `nb01`, …),
the same struct the CPU backend passes. Muser builds that struct in Rust:

```rust
// crates/muser-engine/src/metal/encode/qkv.rs:1292
impl GgmlKargsMulMv {
    fn for_matmul(rows: usize, cols: usize, block_bytes: usize, nr0: i32) -> Self {
        let row_bytes = (cols / 256 * block_bytes) as u64;
        Self {
            ne00: cols as i32,          // elements per row
            ne01: rows as i32,          // rows
            // …(strides elided: nb00 = block_bytes, nb01 = row_bytes, …)
            nr0,                        // rows reduced per threadgroup
            // …
        }
    }
}
```

*(Fields elided; see the file for the full 112-byte struct.)*

## 13.4 The kernels — three sources, one family

[Ch 4](04-pso-and-three-kernel-sources.md) established that Muser runs
kernels from three libraries. All three meet in this chapter's dispatch.

### 13.4.1 The pinned ggml matvec — `kernel_mul_mv_q*_K_f32`

Start with the question that decides everything else in this section:
whose code actually runs when Muse Glimmer projects a token? Not ours.
The primary decode matvec is **llama.cpp's own kernel**, loaded from the
metallib pinned by `MUSER_GGML_METALLIB`
(`crates/muser-engine/src/metal/context.rs:122-131`) at llama.cpp commit
`89e0aa6fd362…` (`PINNED.md`). Registration names the exact functions:

```rust
// crates/muser-engine/src/metal/encode.rs:278
ggml_q4k: ggml_matvec_pipeline(context, "kernel_mul_mv_q4_K_f32")?,
ggml_q5k: ggml_matvec_pipeline(context, "kernel_mul_mv_q5_K_f32")?,
ggml_q6k: ggml_matvec_pipeline(context, "kernel_mul_mv_q6_K_f32")?,
```

We do not quote these kernels' bodies in this book — they are not in the
Muser tree; they are *pinned binary provenance*, compiled from llama.cpp's
source at the comparator commit and built by
`scripts/compile_llama_metallib.sh` (`scripts/` per the repo's tooling).
What Muser owns is the dispatch, and here it is, the `tokens == 1` branch
of `encode_quantized_matmul`:

```rust
// crates/muser-engine/src/metal/encode/qkv.rs:429
if tokens == 1 {
    if let Some(pipeline) = self.ggml_matvec(dtype) {
        let (block_bytes, rows_per_group) = match dtype {
            GgmlType::Q4_K => (144, 2),
            GgmlType::Q5_K => (176, 1),
            GgmlType::Q6_K => (210, 2),
            _ => unreachable!("ggml_matvec returned only for K-quant projections"),
        };
        let args =
            GgmlKargsMulMv::for_matmul(n_out, n_in, block_bytes, rows_per_group as i32);
        encoder.set_compute_pipeline_state(pipeline);
        set_value(encoder, 0, &args);
        encoder.set_buffer(1, Some(weights.metal()), weights.offset() as u64);
        encoder.set_buffer(2, Some(input.metal()), 0);
        encoder.set_buffer(3, Some(output.metal()), 0);
        let simdgroups = 2usize;
        encoder.dispatch_thread_groups(
            MTLSize::new(n_out.div_ceil(rows_per_group * simdgroups) as u64, 1, 1),
            MTLSize::new(32, simdgroups as u64, 1),
        );
        return;
    }
    // …(standalone fallbacks: §13.4.2)…
}
```

Read the geometry off the dispatch:

- **threadgroup** `(32, 2)` — 64 threads = **2 SIMD groups**, llama's own
  `N_SG` for these kernels (the function constants pinned at registration
  bake `nsg=2`, `encode.rs:844-851`).
- **grid** `ceil(n_out / (rows_per_group × 2))` — each threadgroup owns
  `rows_per_group` output rows: **2 rows per group for Q4_K and Q6_K, 1 for
  Q5_K**, matching llama's per-format `nr0`.
- Buffers ride in at llama's slots: kargs at `buffer(0)`, weights at 1
  (with the mmap-arena offset), input at 2, output at 3.

For the Q projection: `n_out = 4,096`, `rows_per_group = 2` →
`4,096 / 4 = 1,024` threadgroups of 64 threads. For K and V: `n_out = 256`
→ 64 threadgroups each. For the gate: 1,024 again.

Why dispatch llama's kernels instead of writing better ones? We arrived at
this fork holding evidence for the other branch. The ancestor line had
written its own Q4_K GEMV and measured it: it beat llama's kernel by 6–10 %
in isolated A/B, with bit-identical output
`[ferrite-book Ch 11]` — a Ferrite-lineage measurement on A18 Pro, never a
Muser result. The expectation we carried into Muser was that the same trick
would transfer, and that owning the inner loop would be strictly better
than borrowing one.

That expectation did not survive contact with Muser's gate, and the reason
has nothing to do with speed. Muser's
gate is stricter than Ferrite's was: full-logit and logprob parity against
a pinned comparator, not token parity. Put plainly, matching the reference
is not a property of the answer, it is a property of the *route to* the
answer — reduction order, rounding points, accumulation tree and all. A
hand-written kernel must therefore prove bitwise equality with llama's
route, and "bit-identical" is a claim you re-earn at every shape, every
dtype, every batch width, forever. That is the lesson: the cheapest way to
match a reference's floating-point behavior is to run the reference's own
compiled code.

It helps that the prize was small anyway. Even in the ancestor's own
telling, the inner loop was proven *not* to be the engine's gap — the win
we would have been defending was never the one that mattered. Under Muser's
constraint, "pin the kernel" dominates "beat the kernel."

### 13.4.2 The standalone fallbacks — `muser_matvec_q4k_4r2s` and friends

What if the pinned metallib is not on the machine at all? Muser does not
refuse to run — it falls back to kernels it owns. That fallback is also the
only member of this family whose source we can open on the page, which
makes it the place to learn what the pinned kernel is doing behind its
compiled wall. Keep one thing in reserve while you read it, though: the
tradeoffs section returns to argue that "falls back" and "runs the same
computation" are not the same sentence.

When the metallib is absent, the same wrapper falls back to Muser-owned
kernels (`qkv.rs:451-474`): Q4_K routes to `muser_matvec_q4k_4r2s`
(grid `n_out/8`, 64 threads), Q5_K to `muser_matvec_q5k_4sg` (grid
`n_out`, 128 threads). The 4r2s kernel — "4 rows, 2 SIMD groups" — is the
Muser-authored adaptation of the Ferrite lineage, and its body is worth
reading because it shows the dequant fused into the MAC concretely:

```metal
// crates/muser-engine/src/shaders/muse_reference.metal:735
kernel void muser_matvec_q4k_4r2s(
    device const uchar *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &cols [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    uint block_count = cols / 256;
    uint row_bytes = block_count * 144;
    uint base_row = group * 8 + simd * 4;
    if (base_row >= rows) return;
    uint active_rows = min(4u, rows - base_row);
    float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    device const uchar *row[4] = {
        weights + ulong(base_row) * ulong(row_bytes),
        // …(rows +1..+3 elided: same pattern)…
    };
    for (uint block_index = 0; block_index < block_count; ++block_index) {
        for (uint row_index = 0; row_index < active_rows; ++row_index) {
            device const uchar *block = row[row_index] + block_index * 144;
            uint delta = *reinterpret_cast<device const uint *>(block);
            float d = float(as_type<half>(ushort(delta & 0xffff)));
            float dmin = float(as_type<half>(ushort(delta >> 16)));
            uint sd0 = *reinterpret_cast<device const uint *>(block + 4);
            uint sd1 = *reinterpret_cast<device const uint *>(block + 8);
            uint sd2 = *reinterpret_cast<device const uint *>(block + 12);
            float d_scale[8];
            float neg_min[8];
            muser_decode_all_q4k_scales(d, dmin, sd0, sd1, sd2, d_scale, neg_min);
            uint input_base = block_index * 256;
            for (uint quant_group = 0; quant_group < 4; ++quant_group) {
                uint packed = uint(block[16 + quant_group * 32 + lane]);
                float low = input[input_base + quant_group * 64 + lane];
                float high = input[input_base + quant_group * 64 + 32 + lane];
                accumulator[row_index] +=
                    fma(d_scale[quant_group * 2], float(packed & 0x0f), neg_min[quant_group * 2]) * low;
                accumulator[row_index] +=
                    fma(d_scale[quant_group * 2 + 1], float(packed >> 4), neg_min[quant_group * 2 + 1]) * high;
            }
        }
    }
    for (uint row_index = 0; row_index < active_rows; ++row_index) {
        accumulator[row_index] = simd_sum(accumulator[row_index]);
    }
    if (lane == 0) {
        for (uint row_index = 0; row_index < active_rows; ++row_index) {
            output[base_row + row_index] = accumulator[row_index];
        }
    }
}
```

*(Rows +1..+3 of the `row[4]` initializer elided — identical pointer
arithmetic at `muse_reference.metal:750-755`.)*

The anatomy, in the order the kernel uses it:

- **Four output rows per threadgroup, four per SIMD group's lane set** —
  `base_row = group * 8 + simd * 4`: 8 rows per threadgroup (2 SIMD groups
  × 4), each lane holding four private accumulators, one per row.
- **Pre-decoded scales** — `muser_decode_all_q4k_scales`
  (`muse_reference.metal:41-62`) unpacks all eight sub-block scale/min
  pairs *once per super-block per row*, folding in `d` and pre-negating the
  min: `d_scale[j] = d × sc_j`, `neg_min[j] = −(dmin × m_j)`. The inner
  loop then uses them as constants.
- **Byte-wise nibble read with `fma`** — lane `lane` reads one byte
  `block[16 + quant_group*32 + lane]`; its low nibble covers one 32-element
  sub-block, its high nibble the partner sub-block (`low`/`high` input
  elements). Each element costs one fused multiply-add —
  `fma(d_scale, nibble, neg_min)` computes the fully dequantized weight
  with a single rounding — plus one multiply-accumulate into the row's
  accumulator.
- **`simd_sum` per row, lane 0 writes** — the 32 lanes each covered 32 of
  every 256 elements; one hardware instruction per row recombines them.

There was a genuine fork inside that inner loop, and the ancestor line
took both branches. What you just read is the *pre-decoded-scales* design
point `[ferrite-book Ch 12]`: unpack every scale up front, then run a loop
with nothing in it but loads and multiply-adds. The sibling v4 kernel chose
*deferred scaling* instead — accumulate the raw nibble products and apply
the scales once per sub-block at the end, spending fewer multiplies for a
more tangled loop. Neither branch is wrong. They are two ways to spend the
same arithmetic, and on their own terms both are correct.

The reason to care is downstream of correctness. Fold the scales in early
and you round early; defer them and you round late. Same function, same
inputs, different last bit. So the Muser fallback is not a slower copy of
the pinned kernel — it is a *different* kernel that happens to agree to
within a rounding step. The tree makes that policy explicit rather than
letting it happen quietly: `encode.rs:370-385` gives Q6_K *no* standalone
fallback at all, "so its math and dispatch remain comparator-exact."

### 13.4.3 The other two libraries, briefly

The **cross-vendor** library (strict-f32, no-fast-math recompile of
`muse_reference` + `nvfp4`, `context.rs:111-121`) supplies
`muser_cross_vendor_q4k/q5k/q6k` — CUDA-parity scalar routes for the
disaggregated lane, gated by `MUSER_CROSS_VENDOR_QK`
(`qkv.rs:300-335`). And the **native NVFP4 lane** replaces this whole
kquant family with `muser_nvfp4_*`/`muser_f16_matvec_c*` kernels
(`qkv.rs:68-227`); [Ch 7](07-nvfp4-native-lane.md) owns that story. The
four-projection graph structure is identical on every lane — only the
inner-loop kernels change.

## 13.5 The Rust dispatch — one concurrent set of four

Four matvecs read the same vector. Do they have to take turns? Nothing in
the math says so, and nothing in the encode says so either — the permission
to overlap is expressed by *where the calls sit*, not by a scheduler.
`encode_token` records all four projections inside a single `dispatch`
closure. The comment at the call site is the design statement:

```rust
// crates/muser-engine/src/decode.rs:5566
// llama.cpp and Ferrite issue the four independent attention
// projections as one concurrent set. They share a read-only input
// and mapped weight arena but write disjoint activations.
dispatch(command, |encoder| {
    self.encode_projection(
        encoder,
        &layer.q,
        &self.activations.post_norm,
        &self.activations.q,
        1,
    );
    self.encode_projection(
        encoder,
        &layer.k,
        &self.activations.post_norm,
        &self.activations.k,
        1,
    );
    // …(v and gate: identical shape, into activations.v / activations.gate;
    //    elided — decode.rs:5584-5597)…
});
```

`encode_projection` (`decode.rs:6044-6089`) is the dtype router: F16 →
`encode_f16_matmul`, NVFP4 → `encode_nvfp4_matmul`, kquant →
`encode_quantized_matmul` of §13.4. Each of the four calls lands on the
same 26,624-byte read of `post_norm` — cheap after the first, because by
then it is L2-resident — and on its own disjoint weight span and its own
output buffer. That disjointness is the whole permission slip: no kernel
here writes anywhere another kernel reads, so none of them needs to wait.

One closure, four kernel dispatches. That asymmetry matters beyond this
page, because it is the kind of thing that quietly corrupts a measurement.
Count closures in a profile, call the count "dispatches," and you are
wrong by a factor on exactly this line — and the QKV line is one of the
biggest in the layer, so the error does not stay small. The gap note says
it in its own words, and we kept the correction:
`decode-dispatch-gap-20260815.md` §Instrumentation correction — "the
`qkvg` closure encodes four kernel dispatches but contributes one profiler
count".

## 13.6 The access pattern — the budget, itemized

This is where the bandwidth story lives. The question is blunt: what does
one token cost in bytes, and how much of that bill do these four matvecs
sign? Per layer, per token, the four matvecs read:

```
  q    : 6,656 × 4,096 = 27,262,976 params
  k    : 6,656 ×   256 =  1,703,936 params
  v    : 6,656 ×   256 =  1,703,936 params
  gate : 6,656 × 4,096 = 27,262,976 params
                     ─────────────────────
  attention projections total = 57,933,824 ≈ 57.9 M params
```

At Q4_K's 0.5625 bytes/param that is ≈ 32.6 MB per layer. Resist writing
that figure down as *the* number, though, because it is not a property of
the architecture: which tensors ship at which dtype is a property of the
loaded GGUF. The loader reads a dtype per tensor as it maps the file, and
the only anchor the tree gives us is that the release artifact's FFN
gate/up are Q4_K. Both receipts are retained — `decode.rs:1294-1310` for
the per-tensor read, `decode.rs:5820-5821` for the anchor.

So parametrize rather than assert. At Q4_K bitrate the four
matvecs are ~32.6 MB/layer; at Q5_K (0.6875 B/param) ~39.8 MB; at Q6_K
(0.8203 B/param) ~47.5 MB. Now put the same layer's other matvecs beside
them — o_proj (27.3 M params), FFN gate/up/down (3 × 132.9 M params).
Against the layer's ~483.8 M projection parameters the attention set is
≈ 12 %, o_proj ≈ 6 %, and the FFN ≈ 82 %: the family this chapter is about
is the *smaller* share of the layer's traffic, which is worth knowing
before you spend a week optimizing it. Scale by 52 layers and add the
~1.34 B-parameter embedding and LM head each, and the per-token stream is
the artifact's 16,756,681,056 bytes (`lib.rs:14`) — the number
[Ch 1](01-why-inference-is-a-memory-problem.md) turned into a token-time
budget. Every one of those bytes crosses the DRAM bus exactly once per
token; this chapter's kernels are the mechanism for four of the nine
per-layer fractions of it.

Two structural notes on *how* the bytes are touched. First, the weights
are a view into the single mmap'd GGUF arena (`weights.offset()` in every
dispatch) — no staging, no per-tensor copies ([Ch 3](03-unified-memory-and-buffers.md)).
Second, each output row's bytes are contiguous (row-major), each
threadgroup reads whole rows, and llama's `nr0 = 2` pairing means
consecutive rows stream together — the access pattern is as close to
"linear read of a big buffer" as the layout allows, which is precisely
what a bandwidth-bound kernel wants.

## 13.7 Tradeoffs

**Four separate matvecs vs a fused QKV kernel.** The obvious alternative —
one kernel reading all four weight matrices and writing all four outputs —
saves three dispatch boundaries per layer (×52) and would let one weight
fetch serve four accumulators. Muse Glimmer cannot take it: there is no
fused QKV tensor to read, and synthesizing one would change the checkpoint.
But the deeper answer is in the routing comment that governs which graph
serves decode at all:

```rust
// crates/muser-engine/src/decode.rs:2085
// The legacy one-token graph uses Ferrite fused residual/norm and
// gate-up kernels whose rounding diverges from the source-pinned
// llama Metal graph enough to breach public logprob tolerance.
// The one-row batch graph dispatches the exact pinned kernels and
// has the same KV transition, so it is the serving correctness
// path until each fused kernel independently passes full-logit
// parity.
```

Read that comment as the war story it is. We had the fused kernels
already — the Ferrite-lineage fused residual/norm and gate-up path, fewer
dispatch boundaries per layer, exactly the win the fused-QKV argument above
is reaching for. We expected a free speedup, on the intuition that fusing
two exact operations ought to stay exact. It was not free. The fused
rounding diverged from the source-pinned llama Metal graph by enough to
breach public logprob tolerance: no bug, no lost precision anyone could
point at, just a different order of operations arriving at a different
last bit.

The lesson is the one this chapter keeps re-teaching in new words. Fusion
is not free even when it is numerically exact in isolation, because the
standard here is not "exact" — it is "exact *against the pinned
comparator*, per kernel." A fused kernel that is more accurate than the
reference still fails. So serving routes single tokens through the one-row
**batch** graph (`forward_batch`, `decode.rs:2092`) — the same op sequence,
running the pinned kernels — while the legacy `encode_token` graph this
book narrates survives for teacher-forced harnesses and phase profiling
(`decode.rs:2124-2182`, gated by `MUSER_METAL_PHASE_PROFILE`). The cost is
paid in engineering discipline, not in tokens: Muser keeps two graphs
precisely so the cheap one can be held out of serving until it proves
parity.

**Batch-width boundaries are numerical boundaries.** For multi-token
inputs the same wrapper switches kernels by token count, and the switch
comment is the campaign in one paragraph:

```rust
// crates/muser-engine/src/metal/encode/qkv.rs:476
// Match the source-pinned llama.cpp Metal dispatch boundary exactly:
// K-quant projections with four through eight activation rows use
// `mul_mv_ext`, with a token-count-specific number of rows per
// threadgroup. This changes the floating-point reduction order, so
// substituting repeated decode GEMVs here breaks embedding/logprob
// numerical parity even when every other layer is identical.
```

Decode never leaves the single-token rung, so it is fair to ask why the
rest of the ladder belongs in a decode chapter. It belongs because parity
is not a decode-only property: the verify path and the prefill path enter
through this same wrapper, and every rung of it is a different arithmetic.

At `tokens` 4..=8 the pinned `kernel_mul_mv_ext_{q4,q5,q6}_K_f32_r1_{2..5}`
pipelines run (`encode.rs:959-1003`); at 16 rows the M16 n32 tiles; at
larger multiples the SGM batch matmuls; at 512-row prefill chunks, the
`kernel_mul_mm_*` GEMMs (`qkv.rs:482-623`). Every one of these is a
different floating-point reduction order, and the wrapper keeps llama's
exact boundaries so the comparator's bits are reproducible at every batch
width. The DFlash verify path leans on the same ladder — its 16-row
matmuls are why the M16 tile family exists at all
([Ch 33](33-speculation-and-the-distributed-verdict.md)).

**Pin the kernel vs write a better one.** Covered in §13.4.1, but it
belongs in the tradeoff ledger with its labels straight: the 6–10 %
isolated-kernel win is Ferrite-lineage evidence `[ferrite-book Ch 11]`
(A18 Pro, Qwen2.5-1.5B), never measured on Muser's M3 Ultra, and Muser's
gate (full-logit parity against pinned llama.cpp) is not winnable by a
hand-written kernel that must then *prove* bitwise equality against
llama's own. The pinned-kernel strategy converts a hard numerical problem
into a build-provenance problem — solved by the metallib receipt of
[Ch 4](04-pso-and-three-kernel-sources.md).

**Fallbacks are correct, not equivalent.** The standalone 4r2s/4sg kernels
compute the same function but round differently (different lane
decomposition, pre-decoded scales). Running without `MUSER_GGML_METALLIB`
is therefore a *different numerical lane*, not merely a slower one —
Q6_K's lack of any fallback (`encode.rs:370-385`) makes that policy
explicit rather than accidental.

## 13.8 Where the gap lives

Does this family contribute to the decode gap? There are two answers, and
holding them apart is the whole point of the section.

In the **closure accounting** of `[docs/decode-dispatch-gap-20260815.md]`,
the answer is no. These four matvecs sit in the *common math* row — 406
closures, production delta 0, "required math / Keep." They are not among
the +196 closure families the note calls out (104 norm boundaries, 39 SWA
staging, 52 KV-publication splits, 1 copy). There is nothing here to
delete: no repeated matvec closure doing identical arithmetic exists to
remove.

In the **byte accounting** of
[Ch 1](01-why-inference-is-a-memory-problem.md), the answer is yes, and
emphatically so — this family is the bulk of the budget. The token's time
is the time to stream 16.76 GB through kernels like these, and how close
they sit to the machine's bandwidth ceiling is the recurring question the
book returns to at [Ch 38](38-measuring-against-llama-cpp.md).

Both answers are correct, because they answer different questions. "Is
there wasted work here?" is a closure question, and the closures say no.
"Is this where the time goes?" is a byte question, and the bytes say yes.
Keeping those two framings from contaminating each other is exactly the
epistemics the gap note exists to enforce.

## 13.9 What comes next

Q, K, V, and gate now hold raw projection outputs — un-normalized,
un-rotated, positionless. Before attention can use them, each head's slice
must be QK-normalized and, on the 39 sliding layers, rotated by RoPE. On
the 13 NoPE layers nothing rotates at all — and that asymmetry is about to
become the most consequential layout decision in the engine. That is
[Ch 14](14-qk-norm-and-rope.md).

## References

- `crates/muser-engine/src/metal/encode/qkv.rs:414-450` — the `tokens == 1`
  pinned-matvec dispatch (quoted in §13.4.1); `:451-474` the fallbacks;
  `:476-508` the `mul_mv_ext` boundary; `:550-623` the M16/SGM/GEMM batch
  ladder; `:1292-1317` `GgmlKargsMulMv`.
- `crates/muser-engine/src/metal/encode.rs:278-280` — the pinned kernel
  names; `:837-866` `ggml_matvec_pipeline` (nsg=2 function constants);
  `:959-1003` the `mul_mv_ext` groups; `:370-385` `supports_projection`
  (Q6_K pinned-only).
- `crates/muser-engine/src/metal/context.rs:122-131` — `MUSER_GGML_METALLIB`
  loading of the pinned llama.cpp metallib.
- `crates/muser-engine/src/shaders/muse_reference.metal:735-788` —
  `muser_matvec_q4k_4r2s` (quoted in §13.4.2); `:41-62` the scale decoder;
  `:790-831` `muser_matvec_q5k_4sg`.
- `crates/muser-engine/src/decode.rs:5566-5598` — the four-projection
  concurrent set; `:6044-6089` `encode_projection`; `:2085-2093` the
  serving-routing comment; `:1294-1310` per-tensor dtype gating at load.
- `crates/muser-engine/tests/muse_golden.rs:96-101` — the geometry
  (6,656 / 32 / 2 / 128) this chapter's shapes derive from.
- `scripts/compile_llama_metallib.sh` — builds the pinned metallib.
- `[docs/decode-dispatch-gap-20260815.md]` — closure-vs-dispatch
  instrumentation; the +196 reconciliation this chapter's gap section
  cites.
- `[docs/goal-parity-ledger-2026-08.md]` — the parity gates the pinned
  kernels exist to satisfy.
- [Ch 1](01-why-inference-is-a-memory-problem.md) — the 16.76 GB/token
  budget this family incurs.
- [Ch 4](04-pso-and-three-kernel-sources.md) — the three kernel sources and
  the metallib receipt.
- [Ch 6](06-the-kquant-family.md) — the block formats the kernels unpack.
- [Ch 17](17-sigmoid-gate-and-oproj.md) — what the gate output does.
- [Ch 36](36-prefill-vs-decode-paths.md) — the GEMM side of the same
  weights.
- `[ferrite-book Ch 11, Ch 12]` — the ancestor's v4/4sg GEMV chapters
  (2×2 worked example, deferred-vs-pre-decoded scaling, and the 6–10 %
  inner-loop verdict — Ferrite-lineage, labeled as such).
