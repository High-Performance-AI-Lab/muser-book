# Chapter 5 — Quantization from scratch
> **status:** polished  ·  **path:** Muse Glimmer, pinned Muser tree
>
> *Prerequisites: Chapters 1–4. You know why one decode token costs roughly
> "the whole model in bytes" (Chapter 1's bytes-per-token spine), and you know
> that Metal kernels are compiled from one of three fingerprinted sources
> (Chapter 4). This chapter never launches a kernel — it is pure arithmetic on
> how a number becomes fewer bytes. The concrete formats arrive in
> [Ch 6](06-the-kquant-family.md) and [Ch 7](07-nvfp4-native-lane.md).*

Chapter 4 left you with a working compute pipeline: `.metal` source becomes a
runnable kernel through the concatenated fast-math library, the strict-f32
cross-vendor library, or the pinned llama.cpp metallib — and a fingerprint
line tells you which one actually ran. Ready to feed it work, we now face the
problem that dwarfs every kernel decision: **the weights have to fit, and
they have to be read — every one of them, every token.**

This chapter builds quantization from zero, with no allegiance to any
particular format. We will invent a tiny 4-bit scheme, pack a block by hand,
dequantize it with every multiply written out, and measure the error we just
introduced. The real formats — kquant in Chapter 6, NVFP4 in Chapter 7 — are
industrial versions of exactly this construction.

---

## 5.1 Why fp16 alone cannot carry a 30B model

A **weight** (also called a **parameter**) is one learned coefficient of the
model. Muse Glimmer is nominally a 30B-class model; counting the tensors the
loader actually validates gives a precise number. The engine's config
asserts every tensor's shape at load
`[crates/muser-engine/src/config.rs:286]`, and the shapes are (Figure 5.1):

```
per layer (52 layers):
  attn_q          [hidden 6656 → 4096]     attn_gate  [6656 → 4096]
  attn_k, attn_v  [6656 → 256]             attn_output[4096 → 6656]
  ffn_gate, up    [6656 → 19968]           ffn_down   [19968 → 6656]
embedding table  [6656 → vocab 202,048]    lm_head    [6656 → 202,048]
```
*Figure 5.1: The tensor inventory implied by `assert_tensor_shapes`
(`config.rs:294-318`). Dimensions are GGUF `ne` order, `[in_dim, out_dim]`.*

Multiply it out (Figure 5.2 — we show the arithmetic so you can
re-derive it):

```
attention per layer : 3 × (6656×4096) + 2 × (6656×256)      =  85,196,800
ffn per layer       : 3 × (6656×19968)                      = 398,721,024
per layer total     :                                         483,917,824
× 52 layers         :                                        25,163,726,848
embedding + lm_head : 2 × (6656×202,048)                     =  2,689,662,976
TOTAL               :                                        27,853,389,824  ≈ 27.85 B
```
*Figure 5.2: Parameter count by hand. The "30B" nameplate is nominal; the
counted total is ≈ 27.85 B. Norm vectors (~1.7 M values) are lost in the
rounding here.*

Now the two walls.

**The capacity wall.** [**f16**](../glossary.md#f16) — a 16-bit IEEE float, the
"half precision" format — is the smallest widely-used *floating-point*
representation, at 2 bytes per weight:

```
27,853,389,824 × 2 B = 55,706,779,648 B ≈ 55.7 GB
```

The decode host is one Mac with an M3 Ultra and **96 GB** of unified memory
(`[docs/memory-footprint.md]` intro). Add what must live beside the weights:
the [KV cache](../glossary.md#kv-cache) — the per-token Key/Value attention
memory of Chapter 1's cost model; [Ch 22](22-the-price-of-context.md) owns
it in depth — which for the release configuration (four full-context slots
at 131,072 tokens) is 7.306 GB, the DFlash draft artifact 1,631,205,312 B,
the
vision projector 1,400,328,928 B, and ~0.99 GB of f32 batch-activation
widths for prefill `[docs/memory-footprint.md]`. That sum is already ≈ 67 GB
— and `memory-footprint.md` is explicit that *"summing artifact sizes with
the KV formula is … only a lower bound"*: the operating system, Metal's
pipelines and workspaces, and per-slot sampler state all live in the same
96 GB. An fp16 model is not a plan; it is a hope.

**The bandwidth wall — the one that settles it.** Chapter 1 established that
decode is ~99% reading weights: every token's forward pass streams the whole
model through the GPU. Quantization is how Muser attacks that stream. The
pinned kquant artifact is **16,756,681,056 bytes**
(`[docs/memory-footprint.md]` artifact manifest; the same constant is
asserted in code at `[crates/muser-engine/src/lib.rs:14]`). Per weight, that
is:

```
16,756,681,056 B × 8 bits/B ÷ 27,853,389,824 weights ≈ 4.81 bits/weight
```

Versus fp16's 16 bits: the artifact is **3.33× lighter per token**. If
decode stayed bandwidth-bound at the same effective rate — the regime
Chapter 1 proved — an fp16 model would run at roughly 35.4 ÷ 3.33 ≈ 10.6
tok/s *at best*, versus the measured kquant 35.440 tok/s `[claims #11]`.
(Derived ceiling from the bytes ratio, not a measurement.) Capacity might
survive an fp16 model on a lucky day. Bandwidth does not.

**[Quantization](../glossary.md#quantization)** is the answer: store each
weight in fewer than 16 (or 32) bits, accept a small, controlled error per
weight, and buy a 3.3× reduction in the per-token byte stream. Everything
else in Part II is the engineering of "small and controlled."

## 5.2 Numbers as bits: f32 and f16

To shrink a number we must first say what a number *is* in memory. An IEEE
754 float is a sign, an exponent, and a mantissa (fraction):

```
 f32 (32 bits):  [ sign:1 ][ exponent:8 ][ mantissa:23 ]   ~7 decimal digits
 f16 (16 bits):  [ sign:1 ][ exponent:5 ][ mantissa:10 ]   ~3 decimal digits
```
*Figure 5.3: IEEE float layouts. The mantissa fixes the relative precision;
the exponent fixes the dynamic range. f16 spans 2^-14 to 65504.*

The mantissa is a binary fraction between 1 and 2 (for normal numbers), so
an f16 weight with 10 mantissa bits is known to about one part in 1,024 —
roughly three decimal digits. Real transformer weights live near zero,
commonly within ±0.05, and f16 represents that range comfortably. What f16
cannot do is *fit two of them in a byte*. For that we leave floating point
behind.

## 5.3 The codebook idea: a number as an index

Here is the single idea behind every format in this book. Instead of storing
the weight's value, store **an integer index into a small table of allowed
values** — a [**codebook**](../glossary.md#codebook). Four bits select among
2⁴ = **16 possible values**. Each index is a [**nibble**](../glossary.md#nibble)
— half a byte, values 0–15 — and two nibbles pack into one byte, which is
where the storage win comes from: 0.5 bytes per weight.

Two families of codebook exist, and Part II contains one of each:

- **Integer (uniform) codebooks.** The 16 values are evenly spaced: a
  base value plus `index × step`. Every kquant format
  (Chapter 6) is integer codebooks all the way down.
- **Float codebooks.** The 16 values are themselves tiny floats, spread
  with *relative* (multiplicative) spacing. NVFP4's e2m1
  (Chapter 7) is exactly a 16-entry float table.

[**Dequantization**](../glossary.md#dequantize) is the act of turning an index
back into a value: look up the table (or compute `base + index × step`), and
out comes an approximation of the original weight. [**Quantization**](../glossary.md#quantization)
is the inverse: pick the index whose value is closest to the original.

The gap between the original value and its reconstruction is the
[**quantization error**](../glossary.md#quantization-error) — the price of the
whole enterprise. The rest of this chapter is about driving that error down
without spending bytes.

## 5.4 One scale for everything: too crude

The first scheme everyone writes down: one global step size for the whole
tensor, chosen from the largest magnitude weight. Say a tensor's weights
span roughly ±0.5, so a symmetric 4-bit grid with 16 levels would use a step
of 1.0/15 ≈ 0.067. Any weight is then replaced by the nearest multiple of
0.067. But real weight *blocks* are much narrower than the global span — a
few dozen neighboring weights typically cluster inside a band a tenth as
wide. A global grid spends most of its 16 levels on values that never occur
in that neighborhood, and the local error is needlessly large. This is the
same failure mode we will demonstrate concretely in §5.7 (the DC-offset
problem), and it is why no format in this book uses a single global scale.

The fix is to **quantize locally, not globally**.

## 5.5 Blocks and scales

Split the tensor into small, contiguous **[blocks](../glossary.md#block)** and
give each block its own **[scale](../glossary.md#scale)** (a local step size).
Now the 4 bits express a position *inside the narrow band this block
actually uses*:

```
value ≈ scale × index            (symmetric: one number per block)
value ≈ scale × index + min      (asymmetric: two numbers per block)
```

A block of 32 weights might span only ±0.05, so its private scale is ~0.1/15
≈ 0.0067 — ten times finer than the global grid above, from the *same*
4-bit index. Any slice of a smooth distribution spans less than the whole
distribution; that is the entire trick, and it is the idea behind every
"K-family" format in Chapter 6.

The scale itself is stored in floating point (typically f16, 2 bytes),
because it must cover a wide range of magnitudes across blocks with only a
few values of precision — the same division of labor as Figure 5.3, now
between the *scale* (coarse, wide-range) and the *index* (fine, local).

## 5.6 Symmetric vs asymmetric: the min+offset trick

**[Symmetric](../glossary.md#symmetric-quantization)** quantization assumes the
block's values are roughly centered on zero. One number per block — the
scale — and the codebook spans `−scale·max_index … +scale·max_index` with 0
landing exactly on 0.

**[Asymmetric](../glossary.md#asymmetric-quantization)** quantization adds a
second number per block — the **[min](../glossary.md#min)** (offset) —
so the codebook can start wherever the data starts: `value = scale × index
+ min`. This is the **min+offset trick**: it costs one extra stored number
per block and buys correct handling of blocks that do not live around zero.

```
 Symmetric (1 number/block)          Asymmetric (2 numbers/block)
 ────────────────────────────        ────────────────────────────────
 grid:    −A ··· 0 ··· +A            grid:    min ··· min + 15·scale
 assumes: zero-centered              assumes: nothing
 index 0 → −A (or 0)                 index 0 → min        (offset!)
 waste:   shifted blocks lose        waste:   one extra f16 per block
          half their levels
 value = scale × index               value = scale × index + min
```
*Figure 5.4: The two codebook geometries. Q4_0 and Q8_0 (Chapter 6) are
symmetric; Q4_K and Q5_K are asymmetric; NVFP4's float codebook is symmetric
by construction (its table is ± pairs).*

When does the difference bite? When a block has a **DC offset** — a mean
far from zero. A block whose values all lie in, say, [0.30, 0.42] forces a
symmetric grid to cover ±0.42, and every negative level is wasted: only
about half the grid is ever addressed. The asymmetric grid puts `min = 0.30`
at index 0 and uses all 16 levels inside the 0.12-wide band — a resolution
roughly 3.5× finer for the same 4 bits. (Whether *weight* blocks in a given
checkpoint carry enough offset to matter is an empirical property of the
checkpoint [unverified for Muse Glimmer]; the format designers clearly
thought it worth the bytes, since the release artifact pays for asymmetric
blocks on most tensors — Chapter 6.)

## 5.7 The worked example: an 8-element block, 4 bits, every step

This is the heart of the chapter — the template every later quant chapter
reuses. We quantize one block by hand. **The numbers below are schematic**,
chosen so the arithmetic is clean; they are not taken from any checkpoint.

**The block.** Eight weights:

```
x = [ -0.15,  0.29,  0.12, -0.03,  0.20,  0.06, -0.09,  0.17 ]
```

**Step 1 — scan the block.**

```
min = −0.15        max = 0.29        range = max − min = 0.44
```

**Step 2 — fit the scale.** 4 bits give 16 levels, indices 0–15. We want
index 15 to land exactly on `max`:

```
scale = range / 15 = 0.44 / 15 = 0.029333…
```

For hand arithmetic, round the scale *up* to 0.03 — quantizers really do
pick a convenient scale (Chapter 6's scales are 6-bit integers times an f16;
Chapter 7's are powers-of-two floats), and a slightly padded range is safe:
indices stay in 0–15 without clamping.

**Step 3 — quantize each element.** `index_i = round((x_i − min) / scale)`:

```
x[0] = −0.15 → ( 0.00)/0.03 =  0.0 →   0
x[1] =  0.29 → ( 0.44)/0.03 = 14.67 → 15
x[2] =  0.12 → ( 0.27)/0.03 =  9.0  →  9
x[3] = −0.03 → ( 0.12)/0.03 =  4.0  →  4
x[4] =  0.20 → ( 0.35)/0.03 = 11.67 → 12
x[5] =  0.06 → ( 0.21)/0.03 =  7.0  →  7
x[6] = −0.09 → ( 0.06)/0.03 =  2.0  →  2
x[7] =  0.17 → ( 0.32)/0.03 = 10.67 → 11
```

**Step 4 — pack.** Eight indices, 4 bits each, two per byte, low nibble
first (the convention Chapter 6 will meet in real code —
`q & 0x0F` is the low element, `q >> 4` the high one):

```
pairs : (0,15) (9,4) (12,7) (2,11)
bytes : 0x0F   0x49   0x7C   0x2B
block : [ min as f16 ][ scale as f16 ][ 0F 49 7C 2B ]
         2 bytes        2 bytes         4 bytes  → 8 bytes total
```

**Step 5 — dequantize.** `x̂_i = min + scale × index_i`. Every multiply
shown:

```
x̂[0] = −0.15 + 0.03 ×  0 = −0.15            (exact)
x̂[1] = −0.15 + 0.03 × 15 = −0.15 + 0.45 = 0.30   (true 0.29)
x̂[2] = −0.15 + 0.03 ×  9 = −0.15 + 0.27 = 0.12   (exact)
x̂[3] = −0.15 + 0.03 ×  4 = −0.15 + 0.12 = −0.03  (exact)
x̂[4] = −0.15 + 0.03 × 12 = −0.15 + 0.36 = 0.21   (true 0.20)
x̂[5] = −0.15 + 0.03 ×  7 = −0.15 + 0.21 = 0.06   (exact)
x̂[6] = −0.15 + 0.03 ×  2 = −0.15 + 0.06 = −0.09  (exact)
x̂[7] = −0.15 + 0.03 × 11 = −0.15 + 0.33 = 0.18   (true 0.17)
```

**Step 6 — measure the error.** Per-element error `x̂_i − x_i`:

```
errors: 0, +0.01, 0, 0, +0.01, 0, 0, +0.01
max  abs error = 0.01
mean abs error = 0.00375
```

Three observations that generalize far beyond this toy:

1. **The elements that defined the range are exact or nearly so.** The block
   minimum reconstructed perfectly; the maximum was off by exactly one
   scale-step because we padded the scale. A quantizer's error is worst for
   values in the *middle* of the range, never for the extremes that set it.
2. **Error is bounded by scale/2.** Every true value lies within half a step
   of a level, so `|error| ≤ scale/2` by construction. Here 0.015; we
   measured ≤ 0.01.
3. **Five of eight elements came out exact** because they happened to sit on
   the grid. Real weights don't sit on grids; real mean error lands near
   `scale/4`. The toy flatters us — keep that in mind when extrapolating.

**Step 7 — the symmetric control.** Quantize the same block symmetrically:
scale = amax/7, reading the 4 bits as signed indices −7…+7,
amax = 0.29, so scale = 0.29/7 ≈ 0.0414 — *coarser than 0.03* even before
accounting for the wasted negative levels this almost-centered block barely
uses. The offset here is mild (min = −0.15, max = 0.29); for a strongly
one-signed block the symmetric penalty is the full factor-of-two of
Figure 5.4.

## 5.8 What the error costs downstream

A weight's quantization error is not an isolated blemish — it is a
*deterministic* perturbation of the model. Each [dot
product](../glossary.md#dot-product) in the forward pass — the
multiply-and-add pairing of two equal-length vectors, the atom under every
weight matrix in this book — mixes in one error term per weight. Through 52
layers the perturbations compound multiplicatively (the drift argument
Chapter 12 makes for normalization). Three honest statements about the cost:

- **The error is fixed and knowable.** Weights are quantized once, offline;
  the dequantized value is the same every token. It is a permanent, exact
  bias — not noise. That is what makes cross-engine parity possible at all:
  Muser's kquant lane reproduces llama.cpp's numbers *bit-for-bit on the
  shared format* precisely because the bytes and the arithmetic order are
  pinned `[crates/muser-engine/src/quant/k_block.rs:169-177]`.
- **Quality loss is real but bounded by measurements, not vibes.** Muser
  measures it with gates: NVFP4-versus-kquant relative perplexity and
  top-token disagreement are budgeted per depth and content class, and one
  published content-local sensitivity (docs text at 65,536 tokens: 15.134%
  vs a 13.339% calibrated gate) is carried *as part of the claim*
  `[claims #10]`. Chapter 7 tells that story.
- **You cannot compare formats by one number.** The same 4-bit-class
  quantization is invisible in plain decode (parity-within-noise,
  §5.9) and decisive in batched speculative verification (6.81 tok/s
  no-go, Chapter 7). The cost of precision hides in batch shapes and
  content classes, and the gates exist to localize it.

## 5.9 Block size: the memory-vs-overhead dial

The block header (min + scale, say 4 bytes as two f16s) is paid once per
block. The **[bitrate](../glossary.md#bitrate)** — bits per weight — is:

```
bitrate = payload bits + header bits / block size
        = 4            + 32 / N
```

Turn the dial (Figure 5.5):

```
 block size N     header/weight     total bits/weight     local range
 ─────────────    ─────────────     ─────────────────     ───────────────
      8              4.000               8.00            very narrow
     32              1.000               5.00            narrow
    256              0.125               4.125            moderate
   1024              0.031               4.03             wide
```
*Figure 5.5: The block-size dial. Smaller blocks buy finer local scales and
lower error; bigger blocks amortize the header. (This table assumes the
naive one-scale-one-min-per-block header of §5.7.)*

Both directions fail. At N = 1,024 the header is nearly free, but a block
that wide spans much of the tensor's dynamic range and the local-scale
advantage evaporates — you drift back toward the global grid of §5.4. At
N = 8 the error is superb and you have doubled the storage. Every real
format parks somewhere in between and then *engineers the header down*:

- **Q4_K** (Chapter 6): N = 256 with an *asymmetric* header, made cheap by a
  two-level hierarchy — one f16 super-scale and one f16 super-min for the
  whole 256, plus six-bit sub-scales per 32-element sub-block. Total header:
  16 bytes per 256 weights = 0.5 bits/weight → **4.5 bits/weight**.
- **NVFP4** (Chapter 7): N = 16 with a *symmetric float* header of a single
  one-byte e4m3fn scale, plus one f32 per tensor. Total: 4 + 8/16 =
  **4.5 bits/weight** — the same bitrate as Q4_K by entirely different
  means.

That coincidence is worth pausing on: two formats, two codebook families,
two block sizes — and the same 4.5 bits. Format design is the art of
spending a fixed half-bit of overhead (on top of the 4 payload bits) in
different places. And 4.5-ish bits is also where the real artifact lands on
average: the whole-GGUF figure computed in §5.1 was 4.81 bits/weight (the
excess over 4.5 is the deliberately more precise tensors Chapter 6
identifies — Q6_K's 6.5625 and Q5_K's 5.5).

One more axis the table hides: **who pays to use the format**. A tiny block
with a cheap codebook dequantizes with one multiply (good for a hot kernel);
a 256-block with 6-bit bit-packed sub-scales costs real decode work per
block (Chapter 6 shows the kernel-side machinery). Block size is a deal
between storage, error, and kernel complexity — not just a storage number.

## 5.10 Tradeoffs

**Asymmetric vs symmetric, measured in bytes.** The min+offset trick costs
one extra stored number per block. At N = 32 with f16 headers that is
4 + 64/32 = 6 bits/weight symmetric vs 4 + 96/32 = 7 bits/weight
asymmetric — a 17% storage tax for offset robustness. Q4_K's two-level
hierarchy is precisely the invention that recovers the tax: min+offset at
4.5 bits/weight, the same bitrate a plain symmetric 32-block would waste
(`4 + 16/32 = 4.5`). Chapter 6 walks the real bytes.

**4 bits vs 16, measured in tokens.** The lane throughputs: kquant
(≈4.81 bits/weight average) 35.440 tok/s and native NVFP4 (4.5 bits/weight)
35.491 tok/s, in the same paired five-rep cell — **parity within noise,
never claimed faster** `[claims #11]` (full cell: 66-token prefix, 32
teacher-forced tokens, F16 KV, adjacent lease window, +0.1444% for NVFP4).
The measured existence of two independent ~4.5-bit artifacts running at
parity with the f16-KV llama.cpp comparator is the strongest statement this
book can make that quantization, done at this rate, does not tax decode
throughput. What 4 bits *does* tax — the batched speculative verify path —
is a Chapter 7 measurement.

**Why not 2 bits?** Nothing in this chapter's arithmetic forbids it — 2-bit
codebooks exist in the wild. But at 4 levels per block the quantization
error approaches the size of the local scale itself, and §5.8's
error-compounding has nowhere to hide. Muser's own gates localized quality
cost at 4-bit-class formats to specific content classes `[claims #10]`;
no 2-bit lane was ever qualified in this program [unverified — no
measurement exists in the retained evidence].

## 5.11 What comes next

You now own the complete template: a codebook, a local scale, an optional
min, a hand-packed block, a dequant with every multiply visible, and an
error budget. Chapter 6 deploys it on the bytes Muser actually ships — the
kquant family (Q4_K, Q5_K, Q6_K) that fills the 16,756,681,056-byte
reference artifact, byte layouts first, then the pinned
`kernel_mul_mv_q*_K_f32` Metal kernels that consume them and the per-tensor
map of which class of weight gets which format.

---

## References

- `[crates/muser-engine/src/config.rs:286]` — `assert_tensor_shapes`: every
  tensor and shape the loader validates (Figure 5.1's source).
- `[crates/muser-engine/src/lib.rs:14]` — the pinned artifact byte size
  16,756,681,056 asserted in the crate docs.
- `[docs/memory-footprint.md]` — 96 GB M3 Ultra host, KV formula and the
  7.306 GB four-slot figure, artifact manifest (16,756,681,056 /
  1,631,205,312 / 1,400,328,928 B), "lower bound" caution.
- `[claims #11]` — `docs/launch-claims.md` row 11: plain Mac NVFP4 35.491
  tok/s vs adjacent kquant 35.440 tok/s at original scope, parity within
  noise; also the five-rep cell description via the ledger (P1.3).
- `[claims #10]` — native NVFP4 quality gates and the published docs@65,536
  content-local sensitivity (15.134% vs 13.339%).
- `[crates/muser-engine/src/quant/k_block.rs:169-177]` — `dot_q4_k_f32_llama`
  doc: the pinned llama.cpp accumulation-order contract for Q4_K.
- [Ch 6](06-the-kquant-family.md) — the kquant family: real byte layouts,
  the 6-bit scale packing, and the dispatch table.
- [Ch 7](07-nvfp4-native-lane.md) — NVFP4: the float codebook and the
  native lane.
- [ferrite-book Ch 5] — the ancestor's Q4_K chapter, whose
  hand-built-superblock method this chapter ports (pedagogical lineage
  only; all numbers here are re-derived from the Muser tree).
