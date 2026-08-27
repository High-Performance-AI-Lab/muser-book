# Chapter 7 — NVFP4: the native lane
> **status:** polished  ·  **path:** Muse Glimmer, pinned Muser tree
>
> *Prerequisites: Chapters 5 and 6. You know what a codebook, a block scale,
> and a bitrate are; you have unpacked kquant super-blocks by hand. This
> chapter does the same for the other weight lane — NVFP4 — and then tells
> the lane's measured story, which is a story about what a format may and may
> not claim.*

Chapter 6 left the kquant family as integer codebooks: uniform grids under
local scales, in a 16.76 GB artifact that the reference lane streams every
token. [NVFP4](../glossary.md#nvfp4) — **N**ative **F**loat **P**recision 4-bit, the format of
NVIDIA's Blackwell FP4 tensor cores — is the same budget spent on a
different idea: the 4-bit payload is itself a tiny *floating-point number*,
and so is its block scale. Muser's **native lane** decodes the Muse Glimmer
weights directly from this format on the Mac, and the same format runs the
remote GX10 prefill producer ([Ch 28](28-the-gx10-and-vllm-nvfp4-prefill.md)).

The chapter has two halves. The first is bytes and arithmetic: the e2m1
codebook, the e4m3fn scale, the fail-closed loader, and the Metal decode
kernels. The second is discipline: what the measurements permit this lane
to claim — and the one claim it must never make.

---

## 7.1 The format in one line — and its price tag

Muser's own CPU oracle states the entire format contract in its module doc:

```rust
// crates/muser-engine/src/quant/nvfp4.rs:1
//! CPU oracle for NVIDIA NVFP4 weights.
//!
//! The product format keeps E2M1 values packed two per byte, one raw E4M3FN
//! scale per 16 values, and one f32 `scale2` per tensor.  The operation order
//! is pinned to the ModelOpt/MLX reference: `(e2m1 * e4m3fn) * scale2`.
```

Deconstruct each term:

- **E2M1** is the 4-bit float [codebook](../glossary.md#codebook): 1 sign, 2
  exponent, 1 mantissa bit. Sixteen entries, and Muser lists them all
  (`[crates/muser-engine/src/quant/nvfp4.rs:7-9]`):

```
 code      :  0     1     2     3    4    5    6    7
 value     : 0.0   0.5   1.0   1.5  2.0  3.0  4.0  6.0
 code (|S) :  8     9    10    11   12   13   14   15
 value     :-0.0  -0.5  -1.0  -1.5 -2.0 -3.0 -4.0 -6.0
```
*Figure 7.1: The complete E2M1 codebook — there is nothing else it can
mean. Note the spacing: 0.5-steps up to 2, then 3, 4, 6 — coarser as
magnitude grows, the signature of a float grid.*

- **E4M3FN** is the 8-bit block scale: 1 sign, 4 exponent, 3 mantissa, bias
  7, no infinities, maximum magnitude 448, and — the "FN," *finite* — a
  NaN at only two encodings (`0x7f`/`0xff`), which the loader rejects
  outright (`[crates/muser-engine/src/weights.rs:244-248]`).
- **One scale per 16 values**, packed two values per byte, plus **one f32
  `scale2` per tensor**.

The [bitrate](../glossary.md#bitrate) arithmetic, shown as always:

```
4 bits/weight (payload)  +  8 bits / 16 weights (E4M3FN scale)  =  4.5 bits/weight
```

Exactly 4.5 — the same figure as Q4_K, reached with a 16-element block and
a one-byte float scale instead of a 256-element super-block with 6-bit
integer sub-scales. (The per-tensor f32 `scale2` adds 4 bytes per tensor:
at the q projection's 27.3 M weights, that is 1.2×10⁻⁷ bits/weight —
nothing.) The GGUF type registry says it out loud: the companion-tensor
design "keeps the serving representation at exactly 4.5 bits/weight"
(`[crates/muser-engine/src/gguf/types.rs:33-36]`).

```
 One NVFP4 row (n_in values) — three separate regions:

   packed E2M1            E4M3FN scales          (per-tensor, once)
   ┌────────────────┐     ┌──────────────┐       ┌─────────┐
   │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│ ... │▓ ▓ ▓ ▓ ... ▓ │  ...  │ scale2  │
   └───────┬────────┘     └──────┬───────┘       └─────────┘
   n_in/2 bytes           n_in/16 bytes          f32
   2 values per byte      1 scale per 16 values  whole tensor

   16-value group detail:
   [ w0w1 ][ w2w3 ][ w4w5 ][ w6w7 ][ w8w9 ][w10w11][w12w13][w14w15] [ s ]
     1 B     1 B     1 B     1 B     1 B     1 B      1 B      1 B    1 B
   lo nibble = even index, hi nibble = odd index; s decodes by Figure 7.1.
```
*Figure 7.2: The NVFP4 layout. A group is 9 bytes of storage for 16 values
— 4.5 bits each — and the dequant order is pinned: `(e2m1 × e4m3fn) ×
scale2`, never regrouped.*

## 7.2 A worked dequant, using the repo's own fixture

Muser's unit test for the loader carries a canonical group, and we will
dequantize its first bytes with every multiply shown. The fixture: eight
packed bytes `10 32 54 76 98 BA DC FE`, one scale byte `0x38`, and
`scale2 = 0.25` (`[crates/muser-engine/src/weights.rs:462-465, 519-536]`).

**Decode the scale first.** `0x38` = `0b0011_1000`: sign 0; exponent =
`(0x38 >> 3) & 0xF` = 7; mantissa = `0x38 & 7` = 0. Normal case
(`[crates/muser-engine/src/quant/nvfp4.rs:118]`):

```
scale = (1 + 0 × 0.125) × 2^(7−7) = 1.0
```

**Then the weights**, in pinned order — look up Figure 7.1, multiply by
the scale, multiply by scale2:

```
byte 0x10 → lo nibble 0x0 → 0.0  → 0.0  × 1.0 × 0.25 =  0.000
          → hi nibble 0x1 → 0.5  → 0.5  × 1.0 × 0.25 =  0.125
byte 0x32 → lo nibble 0x2 → 1.0  → 1.0  × 1.0 × 0.25 =  0.250
          → hi nibble 0x3 → 1.5  → 1.5  × 1.0 × 0.25 =  0.375
byte 0x54 → lo 0x4 → 2.0 → 0.500      hi 0x5 → 3.0 → 0.750
byte 0x76 → lo 0x6 → 4.0 → 1.000      hi 0x7 → 6.0 → 1.500
byte 0x98 → lo 0x8 → −0.0 → −0.0      hi 0x9 → −0.5 → −0.125
```

The test asserts exactly this row — `0, 0.125, 0.25, 0.375, 0.5, 0.75,
1.0, 1.5, −0.0, −0.125, …` — bit-for-bit
(`[crates/muser-engine/src/weights.rs:527-535]`). The GPU agrees: a
dedicated fixture kernel `muser_nvfp4_dequant_fixture` computes
`(muser_e2m1(nibble) * muser_e4m3fn(scales[index/16])) * scale2` per
element and is tested bit-exact against this CPU oracle **for every finite
E4M3FN byte** — all 254 of them
(`[crates/muser-engine/src/shaders/nvfp4.metal:775-788]`,
`[crates/muser-engine/src/metal/encode/qkv.rs:674-699]`).

Two format details worth internalizing before the loader:

- **The rounding contract is pinned too.** Quantizing *into* e2m1 uses
  ModelOpt's ties-to-even midpoints (`e2m1_from_f32`, with the boundary
  table at `[crates/muser-engine/src/quant/nvfp4.rs:150-171]`; the GPU
  twin `muser_e2m1_round` at `nvfp4.metal:121`). A format is not just its
  decode table — it is also how you got there.
- **The `scale2` is a real second factor**, not decoration: it is what
  lets every per-16 scale stay small while tensors span orders of
  magnitude. The loader requires it to be finite and positive
  (`[crates/muser-engine/src/weights.rs:263-267]`).

## 7.3 The loader path: fail-closed companions

NVFP4 tensors never stand alone. Each weight matrix carries two (or three)
companion tensors, named by suffix (`[crates/muser-engine/src/weights.rs:25-27]`):

```
 token_embd.weight                    ← NVFP4_E2M1 payload (n_in/2 × n_out bytes)
 token_embd.weight.nvfp4_scale        ← F8_E4M3FN, shape [n_in/16, n_out]
 token_embd.weight.nvfp4_scale2       ← F32 scalar, one per tensor
 token_embd.weight.nvfp4_input_scale_inv  ← optional F32 scalar (W4A4 only)
```

The lane itself is selected by metadata, and the pairing is strict in both
directions (`[crates/muser-engine/src/loader.rs:72-91]`): native NVFP4
tensors present ⇒ `muser.weight_precision` must say `nvfp4`; the string
`nvfp4` with no native tensors ⇒ error; absent or `q4_k_xl` ⇒ the kquant
lane of Chapter 6. Then `MuseWeights::open` validates every companion
before any inference: the scale tensor must exist, be `F8_E4M3FN`, and
match the exact `[n_in/16, n_out]` shape; it must contain **no NaN scale
byte**; `scale2` must be scalar F32, finite, positive
(`[crates/muser-engine/src/weights.rs:220-267]`). A missing companion is a
load-time `MissingTensor` error — there is no "degrade and continue."

The optional fourth tensor decides the arithmetic mode — and the loader
enforces that too. `input_scale_inv` present ⇔ `muser.activation_precision
= nvfp4`; a mismatch fails the load
(`[crates/muser-engine/src/weights.rs:200-216]`). Its meaning:

- **absent** ⇒ **weight-only W4A16**: activations stay wide, and the Mac
  quantizes them per super-block to a Q8-K-style integer grid at compute
  time (§7.5). This is the *product* configuration — the selected candidate
  is "weight-only W4A16 and has no activation-global-scale tensor"
  `[ledger §P1.4]`.
- **present** ⇒ **W4A4**: activations are dynamically quantized to FP4 in
  groups of 16 before the dot — the compressed-tensors scheme the Blackwell
  producer's tensor cores execute natively. The Mac supports it (for
  batched verify parity with the producer), and §7.6 shows where that
  permission ends.

Row addressing falls out of Figure 7.2 and is worth one glance at real
code — `dequant_row` for a NVFP4 tensor slices `n_in/2` packed bytes and
`n_in/16` scale bytes per row (`[crates/muser-engine/src/weights.rs:82-94]`),
the same row-major contiguity kquant enjoys.

## 7.4 The Metal lane, end to end

On the Metal side, every projection routes through
`encode_projection`, which dispatches on exactly the dtype set Chapter 6
listed (`[crates/muser-engine/src/decode.rs:6044-6091]`): F16 →
`encode_f16_matmul`, NVFP4 → `encode_nvfp4_matmul`, else the kquant
ladder. The NVFP4 encoder is where the mode split becomes kernel selection
(Figure 7.3; `[crates/muser-engine/src/metal/encode/qkv.rs:128-227]`):

```
 encode_nvfp4_matmul(packed, scales, scale2, input_scale_inv, …)

 input_scale_inv = None  →  muser_nvfp4_a16_q8_matvec       (weight-only, W4A16)
                              grid (n_out/8 × columns), 256 threads
 input_scale_inv = Some  →  16-column batch, n_in % 64 == 0:
                            muser_nvfp4_w4a4_m16_n32        (weight-stationary tile)
                          else widths 16/8/4/2/1:
                            muser_nvfp4_w4a4_matvec_c{1,2,4,8,16}
                              grid n_out, 32 threads (one SIMD group per row)
```
*Figure 7.3: The native-lane projection dispatch. The `Nvfp4Args` triple
(n_in, n_out, col0) rides buffer 4; scale2 buffer 5; input_scale_inv
buffer 6. All three kernels live in the no-fast-math cross-vendor library
(Chapter 4's second source) — the integer contractions must not be
reassociated by the compiler (`[crates/muser-engine/src/metal/encode/qkv.rs:203-205]`).*

### 7.4.1 The plain decode kernel (`muser_nvfp4_matvec_c1` family)

The A16 [matvec](../glossary.md#matvec) template (the Ch 6 term — one dot
product per output row), trimmed to its arithmetic core:

```metal
// crates/muser-engine/src/shaders/nvfp4.metal:183
template <ushort NC>
inline void muser_nvfp4_matvec_impl(
    device const uchar *packed,
    device const uchar *scales,
    device const float *input,
    device float *output,
    constant muser_nvfp4_args &args,
    constant float &scale2,
    uint row,
    ushort lane) {
    // …
    for (uint group = uint(lane); group < args.n_in / 16; group += 32) {
        const float block_scale = muser_e4m3fn(scales[scale_row + group]);
        const uint packed_base = packed_row + group * 8;
        const uint element_base = group * 16;
        for (ushort column = 0; column < NC; ++column) {
            device const float *x = input + (args.col0 + uint(column)) * args.n_in;
            float block_sum = 0.0f;
            muser_nvfp4_accumulate_codes8(
                packed + packed_base, x + element_base, block_sum);
            muser_nvfp4_accumulate_codes8(
                packed + packed_base + 4, x + element_base + 8, block_sum);
            const float scaled = block_sum * (block_scale * 16384.0f);
            sums[column] += scaled;
        }
    }
    for (ushort column = 0; column < NC; ++column) {
        const float total = simd_sum(sums[column]) * scale2;
        if (lane == 0) {
            output[(args.col0 + uint(column)) * args.n_out + row] = total;
        }
    }
}
```

One lane pair per 16-value group, the group's scale applied once per block,
`simd_sum` across the 32 lanes of a [SIMD group](../glossary.md#simd-group)
(Ch 2), `scale2` once per row — the exact pinned order
of §7.2, vectorized. The one mystery is the `16384.0f` (= 2¹⁴). The
accumulate helper decodes eight E2M1 nibbles **as f16 bit patterns**
shifted into place — each value embedded as an exact half multiple of
2⁻¹⁴ — so the helper's sum is 2⁻¹⁴ × the true block sum, and the caller
"folds 2^14 into the block scale after accumulating one complete 16-value
block" (`[crates/muser-engine/src/shaders/nvfp4.metal:155-158]`). The
comment names the lineage: "the same half-bit embedding used by MLX's
native NVFP4 kernels." This is how you turn a 16-entry float LUT into pure
bit-slicing on a GPU.

### 7.4.2 The weight-only contraction (`muser_nvfp4_a16_q8_matvec`)

The product lane's decode kernel is stricter than floats: it makes the
whole block contraction **integer-exact**. Per 256-value activation
super-block, the threadgroup quantizes the activation to a Q8-K-style
integer grid *once* — signed first absolute maximum, `iscale =
−127/first_max`, magic-number round-to-nearest-even, clamp at 127 — then
shares the integers across eight output rows. Inside a 16-value group the
dot runs on raw integer codes:

```metal
// crates/muser-engine/src/shaders/nvfp4.metal:682
for (ushort pair = 0; pair < 8; ++pair) {
    const uchar byte = row_packed[packed_base + uint(pair)];
    dot += muser_e2m1_q1(byte & 15) * int(q8[quant_base + uint(pair) * 2u]);
    dot += muser_e2m1_q1(byte >> 4) * int(q8[quant_base + uint(pair) * 2u + 1u]);
}
weighted = long(dot) * long(muser_e4m3_q9(row_scales[scale_group]));
```

The helpers are the trick from §7.2 turned algebraic: `muser_e2m1_q1`
decodes each weight as an integer in units of 2⁻¹, `muser_e4m3_q9` each
scale as an integer in units of 2⁻⁹ (`nvfp4.metal:28-42`; the CPU oracle's
`e2m1_q1`/`e4m3fn_q9` at `[crates/muser-engine/src/quant/nvfp4.rs:17-51]`
prove the equivalence exhaustively). Their products share one fixed
denominator, so the entire block contraction is an order-free i64 sum.
Only the epilogue is floating point — and it is pinned to four scalar
operations:

```metal
// crates/muser-engine/src/shaders/nvfp4.metal:691
float contribution = fma(float(integer_total), 0x1p-10f, 0.0f);   // 2^-10 = Q1 × Q9
const float q8_scale = 1.0f / block_iscale;
contribution = fma(contribution, q8_scale, 0.0f);                 // activation scale
contribution = fma(contribution, weight_scale2, 0.0f);            // tensor scale
total = fma(1.0f, contribution, total);                           // sequential accumulation
```

— then one final `float(half(...))` at the output, because the producer's
linear layers write F16 results and "otherwise Q/K RMS normalization
amplifies hidden low bits" (`[crates/muser-engine/src/quant/nvfp4.rs:331-337]`).
The CPU oracle `dot_nvfp4_a16_q8_f32` performs the identical sequence
(`[crates/muser-engine/src/quant/nvfp4.rs:349-379]`). Why this fanaticism
about integers? Because this lane must match a *CUDA producer* whose
reduction topology no Metal `simd_sum` can reproduce — so the design makes
the parallel part exactly associative and confines every rounding decision
to four named scalar FMAs. Chapter 32 tells the full trust story; file the
technique now.

### 7.4.3 The W4A4 contraction and the F16 tail

When `input_scale_inv` is present, the kernel quantizes each 16-value
activation group to its own E2M1 + E4M3FN pair (the CPU oracle at
`[crates/muser-engine/src/quant/nvfp4.rs:174-238]`), then contracts
weight-Q1 × activation-Q1 × weight-scale-Q9 × activation-scale-Q9 as one
i64 integer sum per group — denominator 2⁻²⁰ — with the epilogue `×
2^-20`, `× weight_scale2`, `× (1/input_scale_inv)`, and the same F16
boundary (`[crates/muser-engine/src/shaders/nvfp4.metal:295-308]`). The
16-column batch form (`muser_nvfp4_w4a4_prequant_m16_n32`) splits the
activation quantization into its own pass so each group is quantized once
and reused across every N=32 output tile
(`[crates/muser-engine/src/metal/encode/qkv.rs:8-65]`).

The lane's **F16 tail** is real code, not a footnote: the unquantized F16
LM head runs on `muser_f16_matvec_c*` — plain half4 dot products
(`[crates/muser-engine/src/shaders/nvfp4.metal:704-752]`) — and the
embedding on `muser_embedding_f16` (`nvfp4.metal:755`), both dispatched by wrappers that pick the F16 route
by byte length or dtype (`[crates/muser-engine/src/metal/encode/qkv.rs:348-360]`).
"Checkpoint mandates an unquantized F16 language head" is a property of
this artifact, recorded in the ledger (§7.5).

## 7.5 The measured lane: parity within noise — never "faster"

Here is the lane's headline measurement, quoted from the ledger's P1.3
decode gate — five-rep cell, same 66-token prefix, 32 teacher-forced
tokens, **F16 KV**, flash attention, release binary, adjacent lease window
(`[ledger §P1.3]`):

```
 lane          mean ns / 32 tokens      CV          tok/s
 ───────────   ─────────────────────   ─────────   ─────────────
 native NVFP4  901,644,358.4           0.13 %      35.490711722
 kquant ctrl   902,946,575.0           0.037 %     35.439527527

 native = 1.001444269× the adjacent kquant control (+0.144427 %)
```
*Figure 7.4: The P1.3 paired decode cells
(`[docs/goal-parity-ledger-2026-08.md]`; receipt SHAs in the entry).*

Read the discipline in the numbers. A 0.14% difference between cells whose
CVs are 0.03–0.13% is parity within noise, and the claims register locks
the wording: "plain Mac NVFP4 35.491 tok/s versus adjacent kquant 35.440
tok/s remain valid at their original scopes" — with the standing
instruction **"Never call decode faster"** `[claims #11]`. The book will
not phrase it loosely either: the native lane's decode is
*parity-within-noise*, full stop.

Why is the gate so narrow — shouldn't 4.5-bit float weights with an
integer-exact kernel fly? The ledger's retained diagnostics answer:
"the NVFP4 layer stack is faster, but this checkpoint mandates an
unquantized F16 language head (about 3.46 ms/token versus the kquant
head's 1.75 ms)" `[ledger §P1.3]`. The F16 head alone eats most of what
the 52-layer FP4 stack saves. Two further anchors for the lane's quality
side (the gates, not the speed):

- The **standard 2,048-token fixture is deterministic and token-identical**
  versus the exact anchor, with bounded nonzero logit drift — max/mean
  absolute error 7.270581/1.040619 at 32 tokens, 10.884401/1.233789 over
  the five-rep 2,048/256 comparator `[docs/nvfp4-fast-lane-evidence-20260817.md §Determinism]`.
  "Zero drift" is explicitly *prohibited* wording `[claims #10]`.
- At depth, one content class (documentation/digest text at 65,536 tokens)
  exceeds its calibrated top-token band — 15.134% vs a 13.339% gate —
  published as a **content-local sensitivity**, not replicated
  cross-document, with the kquant lane selectable as the reference route
  `[claims #10]`. The cost of this quantization is real, measured, and
  localized — Chapter 5's §5.8 promise, kept.

## 7.6 Speculative NVFP4: measured, rejected, fail-closed

The lane's one forbidden fruit is speculation. The measurement that
settled it: a native NVFP4 W4A4 batched-verification diagnostic ran at
**6.805 tok/s** against the kquant speculative bar of 107.9 tok/s — "one
diagnostic, explicitly unqualified" (`[docs/nvfp4-fast-lane-evidence-20260817.md
§Measured product numbers]`) — with verification consuming 35.915 s
of a 37.619 s decode span `[ledger §F-series remediation]`. The disposition is
recorded as **Fallback B**: "speculative serving stays on the qualified
kquant lane; native NVFP4 plus DFlash is rejected by the receiver
configuration rather than silently serving the measured 6.81 tok/s route"
`[docs/nvfp4-fast-lane-evidence-20260817.md §Product route]`, and the
claims register repeats it: "Native NVFP4 speculative decode has no launch
claim and remains fail-closed" `[claims #4]`.

And it is *code*, not policy prose. A receiver configuration declaring the
native producer mode cannot even enroll a DFlash identity
(`[crates/muser-cluster/src/config.rs:128-131]`), and the server refuses
the combination at startup:

```rust
// crates/muser-server/src/state.rs:1667
fn validate_remote_dflash_policy(
    producer_mode: Option<Nvfp4ProducerMode>,
    dflash_configured: bool,
) -> Result<(), InferenceLoadError> {
    if producer_mode == Some(Nvfp4ProducerMode::Native) && dflash_configured {
        return Err(InferenceLoadError::Remote(
            "native NVFP4 fast-lane speculative decode is unqualified; omit --dflash and use plain NVFP4 decode, or route speculative serving to the kquant lane"
                .into(),
        ));
    }
    Ok(())
}
```

The operator sees that message and the process refuses to serve. [Ch 33](33-speculation-and-the-distributed-verdict.md)
owns the full postmortem — including why a 240/240 all-accept target+DFlash
diagnostic still proved nothing about throughput.

## 7.7 Exact vs Native: the producer-mode preview

The `Nvfp4ProducerMode` enum you just saw is this chapter's bridge to Part
VI, so meet it on its own terms (`[crates/muser-cluster/src/config.rs:10-18]`):

```rust
/// Numeric contract selected by the Spark NVFP4 producer. Legacy receiver
/// configurations predate the split and therefore deserialize as `None`;
/// newly generated F-series configurations must name the mode explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Nvfp4ProducerMode {
    Exact,
    Native,
}
```

- **Native** is the product: the GX10's vLLM producer runs the
  deterministic Blackwell W4A4 path, prefilling KV that the Mac receives
  over Handoff V2 and decodes with the weights of this chapter.
- **Exact** is the verification anchor: an integer-dot producer mode whose
  KV exists to be *compared against*, not served. It is selected on the
  **producer side** — `MUSER_NVFP4_EXACT` is a Python environment flag
  for the producer container (`scripts/gx10/vllm/benchmark_native_prefill.py:99-102`
  even *refuses* to benchmark native with it set; the resident daemon pins
  `MUSER_NVFP4_EXACT=0` at `scripts/gx10/vllm/muser_native_prefilld.py:446`).
  It does not exist anywhere in Rust — a reviewer checking the Mac tree
  for it will find nothing, and that is by design.

The two modes use **mode-separated target-cache identities**, "so exact and
native KV entries cannot alias"
`[docs/nvfp4-fast-lane-evidence-20260817.md §Product route]`. What "good
enough" must mean when someone else's GPU computed your prefill — bounded
logit envelopes, exact-token policies, the wizard's gates — is
[Ch 32](32-precision-across-the-handoff.md), the trust chapter.

## 7.8 Tradeoffs

**Float codebook vs integer codebook at the same 4.5 bits.** Chapter 5's
two families meet head-on here. E2M1 spends its 16 codes as 15 magnitudes
plus a signed zero — one fewer *value* than Q4_K's uniform grid — but buys
*relative* spacing (Figure 7.1: each octave halves the resolution instead
of losing it absolutely) and a **float block scale** (E4M3FN spans
2⁻⁹·mantissa subnormals up to 448) where kquant's 6-bit sub-scale is an
integer under an f16 super-scale. The measured consequence of the whole
package is Figure 7.4: parity-within-noise in decode, with quality
sensitized by content class rather than collapsed
`[claims #10]` `[claims #11]`. A clean "float beats int at 4 bits" claim
is **not** what this program's evidence supports — both 4.5-bit formats
land at the same throughput, and the int lane is the reference lock.

**Group 16 vs super-block 256.** NVFP4's 0.5 header bits buy a scale
refresh every 16 weights — 16× finer than kquant's 32-element sub-blocks —
but the scale is a full byte each. Q4_K's two-level packing spends its 0.5
bits on *more integers* under two f16s. The two formats agree on the
budget and disagree on everything else, which is the best evidence that
4.5 bits is a genuine economic point and not an artifact of one design.

**Integer-exact contraction vs fast floats.** §7.4.2's a16-q8 kernel gives
up vectorized-float convenience to make the block dot associativity-free.
The alternative — matching CUDA's reduction by luck — was measured out in
this program's own wizard campaign, where a one-ULP F16 divergence in
layer-1 V snowballed into 51.7 M differing logits before the arithmetic
ABI was pinned `[ledger §2b, 2026-08-24]`. The cost of the integer path is
kernel complexity; the measured benefit is that "deterministic, bounded
drift" became a *property* rather than a hope
`[docs/nvfp4-fast-lane-evidence-20260817.md §Determinism]`.

**W4A4 on the Mac: supported, qualified for batch parity, forbidden for
speculation.** The same `input_scale_inv` that enables
producer-parity M16 verification (§7.4.3) is the mode whose serving use
measured 6.805 tok/s (§7.6). The lane keeps the kernels, gates the
deployment — Fallback B as a *design pattern*: fail-closed beats
silently-slow.

## 7.9 Where the gap lives

For the native lane, "the gap" in the dispatch-gap sense barely exists —
plain decode sits at parity-within-noise with both the kquant lane and the
comparator (§7.5), and the format is not the deficit. Where this lane's
*own* numbers diverge from its hopes is quantization-shaped but not
format-shaped: the mandated F16 LM head (3.46 vs 1.75 ms/token) narrowing
the decode gate `[ledger §P1.3]`, and the W4A4 batched verify collapsing
to 6.81 tok/s where kquant's verify was engineered to 107.9-class
`[ledger §F-series remediation]`. Both are *batch-shape* stories — the
cost of precision hides in batch shapes and content classes, and the gates
exist to localize it.

## 7.10 What comes next

Two weight lanes now stand complete: kquant for the reference and
speculative lanes, NVFP4 for the native product lane. The next chapter
follows the one component that runs on exactly one of them — the DFlash
draft model, a five-layer kquant assistant that reads the target's hidden
states and proposes tokens for exact verification. It is the smallest
model in this book, the cheapest thing in memory-footprint.md's manifest
after nothing, and — thanks to one wrong constant that survived an entire
campaign — the best teacher of what a draft must guarantee.

---

## References

- `[crates/muser-engine/src/quant/nvfp4.rs:1-6]` — the pinned format
  contract (§7.1); `:7-51` E2M1 LUT and the Q1/Q9 integer encodings;
  `:109-171` E4M3FN decode and the ModelOpt rounding boundaries;
  `:241-255` `dequant_nvfp4_row`; `:349-379` `dot_nvfp4_a16_q8_f32`.
- `[crates/muser-engine/src/gguf/types.rs:33-39]` — `NVFP4_E2M1` /
  `F8_E4M3FN` registration and the "exactly 4.5 bits/weight" comment.
- `[crates/muser-engine/src/loader.rs:72-91]` — fail-closed lane pairing.
- `[crates/muser-engine/src/weights.rs:25-27]` — companion-tensor suffixes;
  `:220-290` `nvfp4_aux` validation (shape, dtype, NaN rejection, positive
  scalar scale2, activation-precision pairing); `:519-536` the fixture
  test this chapter's worked example mirrors.
- `[crates/muser-engine/src/shaders/nvfp4.metal:1-5]` — the GPU format doc;
  `:17-42` LUT and integer twins; `:155-158` the MLX half-bit embedding;
  `:183-242` the A16 matvec family; `:613-702` `muser_nvfp4_a16_q8_matvec`;
  `:704-770` the F16 matvec/embedding kernels; `:775-788` the dequant
  fixture kernel.
- `[crates/muser-engine/src/metal/encode/qkv.rs:128-227]` —
  `encode_nvfp4_matmul` dispatch and geometries; `:8-65` the two-pass M16
  prequant route; `:348-360` F16-layout detection;
  `:674-699` the bit-exact E4M3FN sweep test.
- `[crates/muser-engine/src/decode.rs:6044-6091]` — projection routing;
  `:3254-3291` the verify-route banner (mode names as the engine prints
  them).
- `[crates/muser-cluster/src/config.rs:10-18]`, `:128-131` —
  `Nvfp4ProducerMode`; native mode cannot enroll DFlash geometry.
- `[crates/muser-server/src/state.rs:1667-1678]` —
  `validate_remote_dflash_policy`, the fail-closed Fallback B refusal.
- `[ledger §P1.3]`, `[ledger §P1.4]`, `[ledger §F-series remediation]`,
  `[ledger §2b 2026-08-24]` — `docs/goal-parity-ledger-2026-08.md`: the
  paired decode cells, the weight-only artifact correction, the 6.81
  no-go, the one-ULP wizard chase.
- `[docs/nvfp4-fast-lane-evidence-20260817.md]` — Fallback B disposition,
  measured product table (incl. 6.805), determinism/drift envelopes.
- `[claims #4]`, `[claims #10]`, `[claims #11]` — `docs/launch-claims.md`:
  native spec fail-closed; quality gates and prohibited wording; the
  35.491/35.440 parity scope.
- `[scripts/gx10/vllm/benchmark_native_prefill.py:99-102]`,
  `[scripts/gx10/vllm/muser_native_prefilld.py:446]` — `MUSER_NVFP4_EXACT`
  is producer-side Python only.
- [Ch 5](05-quantization-from-scratch.md), [Ch 6](06-the-kquant-family.md)
  — the codebook/block template and the kquant counterpart.
- [Ch 32](32-precision-across-the-handoff.md),
  [Ch 33](33-speculation-and-the-distributed-verdict.md) — the trust and
  speculation chapters this one forward-points to.
