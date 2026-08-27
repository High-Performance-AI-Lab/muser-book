# Chapter 6 — The kquant family on the reference lane
> **status:** polished  ·  **path:** Muse Glimmer, pinned Muser tree
>
> *Prerequisites: Chapter 5. You have hand-packed an 8-element 4-bit block,
> dequantized it with every multiply shown, and measured the error. This
> chapter runs the same procedure on the real bytes Muser ships — no more
> schematic numbers. Two Metal kernels get quoted; the rest is byte layout
> and arithmetic.*

Chapter 5 ended on a promise: the concrete formats. Here is the first one.
The [kquant](../glossary.md#kquant) family — llama.cpp's "K-quant" block formats — is what fills
the 16,756,681,056-byte reference artifact that Muser's kquant lane decodes
from, the lane measured at 35.440 tok/s `[claims #11]` and used as the
program's reference lock. We will read its byte layouts straight from
Muser's own dequantizers, dequantize real-format elements by hand, map
which tensor class carries which member of the family, and then watch the
dispatch code choose a kernel for each batch shape.

---

## 6.1 One model, several dtypes — decided by the GGUF

A GGUF file carries a dtype **per tensor**, not per model. Muser's parser
enumerates the types it is willing to meet:

```rust
// crates/muser-engine/src/gguf/types.rs:9
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    // …
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    // …
    /// Muser-native NVFP4 E2M1 payload, two logical values per byte.
    /// Per-16 E4M3FN scales and the per-tensor f32 scale2 live in bound
    /// companion tensors; this keeps the serving representation at exactly
    /// 4.5 bits/weight without relying on llama.cpp's experimental format.
    NVFP4_E2M1 = 1000,
    /// Raw E4M3FN bytes used only by NVFP4 companion scale tensors.
    F8_E4M3FN = 1001,
}
```

Every type knows its geometry — bytes per block and elements per block —
and those two numbers define the format's
[bitrate](../glossary.md#bitrate) `[crates/muser-engine/src/gguf/types.rs:75-127]`:

```
 format   block bytes   elements   bits/element   min+offset?   codebook
 ───────  ───────────   ────────   ────────────   ───────────   ────────
 Q4_0        18            32         4.50         no (sym)      4-bit int
 Q8_0        34            32         8.50         no (sym)      8-bit int
 Q4_K       144           256         4.50         yes           4-bit int
 Q5_K       176           256         5.50         yes           5-bit int
 Q6_K       210           256         6.5625       no (signed)   6-bit int
 F16          2             1        16.0          exact         float
```
*Figure 6.1: The kquant family as registered in `GgmlType::block_size` /
`block_elements`. Bits/element = block_bytes × 8 ÷ elements — derive each
one yourself: Q4_K is 144×8/256 = 4.5; Q6_K is 210×8/256 = 6.5625.*

The CPU reference path can dequant any of these (`quant/dispatch.rs`
fans out per dtype); the live Metal decode path is narrower — a projection
tensor must be one of `Q4_K | Q5_K | Q6_K | NVFP4_E2M1 | F16`
(`[crates/muser-engine/src/decode.rs:136-139]`), and the embedding table
must be `Q4_K` or `F16` (`[crates/muser-engine/src/decode.rs:1209]`).

The lane itself is chosen fail-closed at load: the GGUF must declare
`muser.weight_precision`, and a kquant artifact is the default only when no
native NVFP4 tensors exist (`[crates/muser-engine/src/loader.rs:72-91]`).
Chapter 7 covers the `nvfp4` pairing; this chapter stays on `q4_k_xl`.

## 6.2 Q4_K: the 144-byte super-block

The unit of Q4_K storage is a **[super-block](../glossary.md)**: 256
weights in 144 bytes, which is the 4.5 bits/weight of Figure 6.1. As in
Chapter 5, we read the layout off the dequantizer — this is Muser's
complete, real function:

```rust
// crates/muser-engine/src/quant/k_block.rs:12
pub fn dequant_q4_k(block: &[u8], out: &mut [f32]) {
    debug_assert!(block.len() >= 144);
    debug_assert!(out.len() >= 256);

    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let scales = &block[4..16];
    let qs = &block[16..144];

    // get_scale_min_k4: extract 6-bit sc and m for sub-block j (0..7)
    let get_scale_min = |j: usize| -> (f32, f32) {
        let (sc, m) = if j < 4 {
            (scales[j] & 0x3F, scales[j + 4] & 0x3F)
        } else {
            let sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
            let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
            (sc, m)
        };
        (d * sc as f32, dmin * m as f32)
    };

    // 4 outer groups of 64 elements, 2 sub-blocks per group, 32 qs bytes per group.
    let mut q_off = 0usize;
    let mut is = 0usize;
    let mut base = 0usize;
    while base < 256 {
        let (d1, m1) = get_scale_min(is);
        let (d2, m2) = get_scale_min(is + 1);
        for l in 0..32 {
            let q = qs[q_off + l];
            out[base + l] = d1 * (q & 0x0F) as f32 - m1;
            out[base + l + 32] = d2 * (q >> 4) as f32 - m2;
        }
        q_off += 32;
        is += 2;
        base += 64;
    }
}
```

The byte map (Figure 6.2), in the ASCII style Chapter 5 promised:

```
 Q4_K super-block — 144 bytes for 256 elements

 offset   size   field      meaning
 ─────────────────────────────────────────────────────────────────
  0x00     2     d          f16  super-scale        (shared by 8 sub-scales)
  0x02     2     dmin       f16  super-min-scale    (shared by 8 sub-mins)
  0x04    12     scales     8× 6-bit sc + 8× 6-bit m, packed (Fig 6.3)
  0x10   128     qs         256 nibbles (2 per byte) — the weights
  0x90     —    (end)      2+2+12+128 = 144 bytes
 ─────────────────────────────────────────────────────────────────
  0x10 = 16,  0x90 = 144
```
*Figure 6.2: The Q4_K super-block. Four regions: two f16 headers, a
12-byte packed scale/min strip, 128 bytes of nibbles. Compare the 8-byte
toy block of Chapter 5 — same idea, one more level of hierarchy.*

The structure is Chapter 5's min+offset scheme with the two-level header
that keeps it at 4.5 bits: the 256 elements split into **eight
[sub-blocks](../glossary.md) of 32**, each with its own 6-bit scale
`sc` (0–63) and 6-bit min `m` (0–63), all multiplied by the shared f16 `d`
and `dmin`. The value of element *i* in sub-block *j* is:

```
y = d × sc_j × nibble  −  dmin × m_j
```

Exactly `scale × index + min` from Chapter 5 — as `d·sc_j` (effective
scale) and `−dmin·m_j` (effective min) with a sign flip folded in. The
header budget: 4 bytes of f16s + 12 bytes of packed 6-bit fields = 16 bytes
per 256 weights = 0.5 bits/weight of [overhead](../glossary.md#overhead).
Chapter 5's dial table charged 0.125 bits for a naive one-f16-pair header
per 256; Q4_K spends 0.5 — and buys *eight* independent local scales/mins
instead of one.

The 96 bits of scale/min strip are packed with zero padding. Sub-blocks
0–3 live wholly in bytes 0–7; sub-blocks 4–7 split their values between the
low/high halves of bytes 8–11 and the top two bits of bytes 0–7:

```
 byte          b7 b6 │ b5 b4 b3 b2 b1 b0      contents
 ──────────────────────────────────────────────────────────────────────
 scales[0..4]   sc4+ │ sc0 (6 bits)            top2 of scales[j] = sc_{j+4} hi
 scales[4..8]   m4+  │ m0  (6 bits)            top2 of scales[4+j] = m_{j+4} hi
 scales[8..12]  m_lo │ sc_lo                   lo4|hi4 reassembles sc/m of 4..7
```
*Figure 6.3: The 6-bit packing, compressed. The `get_scale_min` closure
above is the authoritative spec: sub-blocks ≥4 reassemble `sc` from the low
nibble of `scales[j+4]` plus the top 2 bits of `scales[j-4]`, and `m` from
the high nibble plus the top 2 bits of `scales[j]`.*

Note also the **interleaving** in the dequant loop: within each 64-element
group, the *low* nibble of byte `qs[q_off+l]` feeds element `base+l`
(even sub-block) and the *high* nibble feeds element `base+l+32` (odd
sub-block). One byte, two sub-blocks, two different scales — the property
Chapter 5's toy deliberately did not have.

## 6.3 A worked dequant of real-format bytes

Hand-build one super-block (values schematic in magnitude — real weights
dequant near ±0.05; the mechanics are identical):

**Headers.** Pick `d` = 0.03125 = 2⁻⁵ and `dmin` = 0.0078125 = 2⁻⁷, both
exact in f16. f16 bits: exponent = power+15, mantissa 0 → `d` = `0x2800`,
`dmin` = `0x2000`; little-endian bytes `00 28` and `00 20`.

**Sub-block scales.** Choose (all 0–63):

```
 sc = [20, 12, 44,  8, 33, 25, 17, 9]      m  = [ 4, 16,  8, 24, 12, 20, 28, 36]
```

Pack per Figure 6.3 — each line is the formula with the arithmetic shown:

```
 scales[0]  = sc0 | ((sc4 >> 4) << 6) = 20 | 128 = 148 = 0x94
 scales[1]  = sc1 | ((sc5 >> 4) << 6) = 12 |  64 =  76 = 0x4C
 scales[2]  = sc2 | ((sc6 >> 4) << 6) = 44 |  64 = 108 = 0x6C
 scales[3]  = sc3 | ((sc7 >> 4) << 6) =  8 |   0 =   8 = 0x08
 scales[4]  = m0  | ((m4  >> 4) << 6) =  4 |   0 =   4 = 0x04
 scales[5]  = m1  | ((m5  >> 4) << 6) = 16 |  64 =  80 = 0x50
 scales[6]  = m2  | ((m6  >> 4) << 6) =  8 |  64 =  72 = 0x48
 scales[7]  = m3  | ((m7  >> 4) << 6) = 24 | 128 = 152 = 0x98
 scales[8]  = (sc4 & 0x0F) | ((m4 & 0x0F) << 4) = 1 | 192 = 193 = 0xC1
 scales[9]  = (sc5 & 0x0F) | ((m5 & 0x0F) << 4) = 9 |  64 =  73 = 0x49
 scales[10] = (sc6 & 0x0F) | ((m6 & 0x0F) << 4) = 1 | 192 = 193 = 0xC1
 scales[11] = (sc7 & 0x0F) | ((m7 & 0x0F) << 4) = 9 |  64 =  73 = 0x49
```

Round-trip check for sub-block 5, substituting into `get_scale_min` exactly
as the code does:

```
 sc5 = (scales[9] & 0x0F) | ((scales[1] >> 6) << 4) = 9 | (1 << 4) = 25  ✓
 m5  = (scales[9] >> 4)    | ((scales[5] >> 6) << 4) = 4 | (1 << 4) = 20  ✓
```

**Nibbles.** Set `qs[0]` = `0x5E` (offset 16) and `qs[32]` = `0x3C`
(offset 48); all other qs bytes zero.

**Dequantize three elements.** Precompute the effective scale/min per
sub-block (`d·sc`, `dmin·m`):

| sub-block | sc | m | eff. scale = d·sc | eff. min = dmin·m |
|---:|---:|---:|---:|---:|
| 0 | 20 | 4 | 0.03125×20 = 0.625 | 0.0078125×4 = 0.03125 |
| 1 | 12 | 16 | 0.03125×12 = 0.375 | 0.0078125×16 = 0.125 |
| 2 | 44 | 8 | 0.03125×44 = 1.375 | 0.0078125×8 = 0.0625 |

```
Element 0   (sub-block 0, low nibble of qs[0]):
 q = 0x5E → nibble = q & 0x0F = 14
 y0  = 0.625 × 14 − 0.03125 = 8.75 − 0.03125 = 8.71875

Element 32  (sub-block 1, high nibble of the SAME byte):
 q = 0x5E → nibble = q >> 4 = 5
 y32 = 0.375 × 5 − 0.125     = 1.875 − 0.125  = 1.75

Element 64  (sub-block 2, low nibble of qs[32]):
 q = 0x3C → nibble = q & 0x0F = 12
 y64 = 1.375 × 12 − 0.0625   = 16.5 − 0.0625  = 16.4375
```

One byte, two sub-blocks, two different effective scales — Chapter 5's
"zoom into the local band," now twice per byte. You have dequantized Q4_K
by hand against the shipping code.

## 6.4 Q5_K and Q6_K: the siblings

**Q5_K** is Q4_K plus one extra bit per element, stored in a separate
plane. Its 176 bytes (Figure 6.4): the same `d`/`dmin`/12-byte scale strip, then 32
bytes `qh` holding one high bit per element (256 bits), then the same 128
bytes of low nibbles `[crates/muser-engine/src/quant/k_block.rs:51-60]`:

```
 Q5_K — 176 bytes / 256 elements      y = d·sc·(nibble | qh_bit<<4) − dmin·m
 ────────────────────────────────────────────────────────────────────────
  0x00   2   d          0x02   2   dmin       0x04  12   scales
  0x10  32   qh         0x30 128   qs         0xB0   —  (end = 176)
```
*Figure 6.4: Q5_K layout (offsets hex: 0x30 = 48, 0xB0 = 176). The 5-bit
codebook lifts the index range from 0–15 to 0–31; 176×8/256 = 5.5 bits/element
(`[crates/muser-engine/src/quant/k_block.rs:61-105]` for the dequant).*

**Q6_K** changes shape more: a *signed* 6-bit codebook split across two
planes, with the per-sub-block scales as plain int8 and the single f16
super-scale at the **end** of the block (Figure 6.5):

```
 Q6_K — 210 bytes / 256 elements      y = d · sc_j · q ,  q = 6-bit signed
 ────────────────────────────────────────────────────────────────────────
  0x00 128   ql   low 4 bits of each code, packed 2/byte
  0x80  64   qh   high 2 bits of four codes per byte
  0xC0  16   sc   16 × int8 sub-block scales (sub-blocks of 16 elements)
  0xD0   2   d    f16 super-scale (LAST field — bytes 208..210)
```
*Figure 6.5: Q6_K layout (0x80 = 128, 0xC0 = 192, 0xD0 = 208). The Muser
kernel reads these offsets verbatim: `ql = &block[0..128]; qh =
&block[128..192]; sc = &block[192..208]; d = f16(block[208..210])`
`[crates/muser-engine/src/quant/k_block/q6.rs:21-26]`.*

Each of the 16 sub-blocks (16 elements each) reconstructs its codes as
`((ql_nibble) | (qh_2bits << 4)) − 32` — an unsigned 0–63 payload shifted
to signed −32…+31 — then scales once: 210×8/256 = 6.5625 bits/element.
No min term: a signed symmetric grid, but with 64 levels the zero-waste
argument of Chapter 5 matters less. The code-extraction quadruple
(`q1..q4` pairing `ql` low/high nibbles with two-bit `qh` fields) is at
`[crates/muser-engine/src/quant/k_block/q6.rs:45-52]`, in the llama-pinned
deferred-scaling order (§6.7).

## 6.5 Which tensor carries which format

The release artifact is a *mix*, like llama.cpp's "Q4_K_M" recipes. The
authoritative in-repo map is the shape table of the M=16 microbenchmark
harness, which enumerates the exact verify/draft projection shapes with
their dtypes and how often each fires per speculative cycle
(`[crates/muser-bench/src/m16.rs:137-226]`):

```
 label                    dtype   shape (n_in→n_out)   per-cycle mult
 ───────────────────────  ─────   ──────────────────   ──────────────
 attn_q/gate              Q4_K    6656→4096            104  (= 2 × 52 layers)
 attn_k/v                 Q4_K    6656→256              78
 attn_v                   Q6_K    6656→256              26
 attn_output              Q4_K    4096→6656             52
 ffn_gate/up              Q4_K    6656→19968           104
 ffn_down                 Q4_K    19968→6656            26
 ffn_down                 Q6_K    19968→6656            26
 lm_head                  Q5_K    6656→202048            1
 draft.k                  Q4_K    6656→1024              5  (draft layers)
 draft.v                  Q6_K    6656→1024              5
 draft.fc                 Q4_K    33280→6656             1
```
*Figure 6.6: Dtypes and shapes on the release path, from the M16 bench
`SHAPES` table. The multiplicities are the harness's own counts of
dispatches per verify cycle / draft block.*

Read the mix off the counts. Attention k/v projections: 78 + 26 = 104
tensors = two per layer, so **26 of the 104 k/v tensors are Q6_K** and the
rest Q4_K. FFN down-projections: 26 + 26 = 52, so **ffn_down alternates
Q4_K/Q6_K by layer**. Everything bandwidth-dominant — q, gate, output,
ffn_gate/up — is Q4_K; the FFN gate/up tensors are pinned Q4_K by the fused
kernel's own guard (`[crates/muser-engine/src/decode.rs:5819-5821]`). The
**lm_head is Q5_K** — one tensor, but a 6656×202,048 one, worth
924,571,648 B by the arithmetic: 26 blocks/row × 176 B × 202,048 rows. The
**embedding table rides Q4_K** on this artifact through the dedicated
`muser_embedding_q4k` kernel (§6.7). This is Chapter 5's bitrate analysis
made flesh: bulk at 4.5 bits, promoted tensors at 5.5/6.5625, averaging the
whole-artifact 4.81 bits/weight computed in Chapter 5.

The draft rows (`draft.*`) preview [Ch 8](08-the-dflash-draft.md): the
DFlash assistant is itself kquant — its loader *requires* Q4_K/Q5_K/Q6_K
(`[crates/muser-engine/src/dflash/weights.rs:148-158]`).

## 6.6 The bytes-per-token tie-back to Chapter 1

Chapter 1's whole thesis was "one token ≈ stream the model." Now you can
compute that stream precisely for one projection. A
[matvec](../glossary.md#matvec) — a matrix-by-vector multiply, one dot product
of the weight row against the input vector per output element; the shape
every projection takes when exactly one token is in flight ([Ch 13](13-the-qkv-gate-matvec-family.md)
derives it from zero) — reads `n_in / 256` super-blocks per row; each row
of the q projection (n_in = 6656) is:

```
26 blocks × 144 B = 3,744 B per row   × 4,096 rows = 15,335,424 B per q matrix
```

Do the same for every tensor class in Figure 6.6 and sum: you converge on
the artifact's own 16,756,681,056 B (≈ 16.76 GB decimal, ≈ 15.6 GiB — this
book follows `docs/memory-footprint.md` in using decimal GB). That file
*is* Chapter 1's per-token weight read, and its size is not an accident:
it is 27.9 B parameters at the mixed 4.5/5.5/6.5625-bit rates of Figure
6.6. The [block](../glossary.md#block) layout is row-major — each output row
is a contiguous run of super-blocks — which is exactly why a matvec kernel
can stream it (§6.7) and why prefill of T tokens costs roughly the same
DRAM traffic as one token (`[crates/muser-engine/src/weights.rs:4-7]`).

## 6.7 The kernels that eat these bytes

Muser deliberately runs kquant matmuls through **three sources**
(recall Chapter 4). For a single-token decode projection
(`encode_quantized_matmul`, tokens = 1), the first choice is the pinned
llama.cpp metallib:

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
        // …
```

`ggml_matvec` resolves to `kernel_mul_mv_q4_K_f32`, `kernel_mul_mv_q5_K_f32`,
or `kernel_mul_mv_q6_K_f32` from the metallib loaded via
`MUSER_GGML_METALLIB`
(`[crates/muser-engine/src/metal/encode.rs:278-280]`). The launch geometry
is named in prose as the contract requires: grid = `n_out ÷
(rows_per_group × 2)` threadgroups, threadgroup size `(32, 2)` — 64
threads, two [SIMD groups](../glossary.md#simd-group) (Chapter 2's 32-lane
hardware execution unit), Q4_K/Q6_K computing two rows per group and Q5_K
one (`[crates/muser-engine/src/metal/encode/qkv.rs:444-448]`). Note what
the `unreachable!` arm implies: **without the metallib there is no
Muser-authored single-token Q6_K matvec** — the fallback switch panics on
Q6_K (`[crates/muser-engine/src/metal/encode/qkv.rs:451-459]`); only Q4_K
(`muser_matvec_q4k_4r2s`) and Q5_K (`muser_matvec_q5k_4sg`) have
hand-written siblings.

When the ferrite-lineage fallback does run, here is the Q4_K kernel's core
— read it against §6.3:

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
    // …
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
```

*(Lines elided: the row-pointer setup and the final `simd_sum` reduction —
see the file.)* Every offset you computed by hand is there: bytes 0–3 read
as one `uint` and split into the two f16s, bytes 4–15 as three `uint`s
feeding the 8-way scale decode (`muser_decode_all_q4k_scales`,
`muse_reference.metal:41`), nibbles from byte 16 onward with the low/high
sub-block pairing of §6.3. The lane geometry (one lane per byte, 32 lanes
covering one 64-element group per iteration) is Part IV material —
[Ch 13](13-the-qkv-gate-matvec-family.md) walks it properly.

**What a matvec must read per block, then:** 4 bytes of headers, 12 bytes
of packed scales, and 128 bytes of nibbles, per 144-byte super-block, per
row — plus the activation vector it is dotted against. Nothing else exists
to read; the block is entirely self-describing. That is the
[access pattern](../glossary.md#access-pattern) in one sentence, and it is why
the bytes-per-token accounting of §6.6 is exact.

**Batch shapes get their own kernels**, all selected in
`encode_quantized_matmul` — the dispatch ladder of Figure 6.7
(`[crates/muser-engine/src/metal/encode/qkv.rs:414-641]`):

```
 token count   route (kquant)                                    source
 ───────────   ──────────────────────────────────────────────   ──────────
 1             kernel_mul_mv_q{4,5,6}_K_f32                      llama metallib
 2–3           same kernel, one launch per token                 llama metallib
 4–8           mul_mv_ext family (rows-per-TG 2/3/4/5)           llama metallib
 16            m16_q{4,5,6}k_n32 weight-stationary tile          muser (ferrite)
 Q4_K, ≥16,    matmul_q4k_batch_sgm_aligned                      muser (ferrite)
 aligned
 any (else)    kernel_mul_mm_q{4,5,6}_K_f32 aligned/bounds       llama metallib
```
*Figure 6.7: The kquant dispatch ladder. `MUSER_CROSS_VENDOR_QK=1` swaps
any rung for the strict-f32 `muser_cross_vendor_q*` kernels (Chapter 4's
second source), and `MUSER_MULTI_COL_VERIFY` gates an exact multi-column
verify route (`[crates/muser-engine/src/metal/encode/multicol.rs:12-14]`).*


Why keep llama's batch boundaries at all? The comment in the dispatch says
it exactly — and it is the chapter's most important citation:

```rust
// crates/muser-engine/src/metal/encode/qkv.rs:476
// Match the source-pinned llama.cpp Metal dispatch boundary exactly:
// K-quant projections with four through eight activation rows use
// `mul_mv_ext`, with a token-count-specific number of rows per
// threadgroup. This changes the floating-point reduction order, so
// substituting repeated decode GEMVs here breaks embedding/logprob
// numerical parity even when every other layer is identical.
```

The embedding kernel completes the single-token picture. `muser_embedding_q4k`
(`[crates/muser-engine/src/shaders/muse_reference.metal:961]`) is one
thread per output element: it resolves `row_bytes = (hidden_dim/256) × 144`
— the same 3,744-byte arithmetic as §6.6 — and calls the shared
`muser_q4k_value` scalar dequant (`muse_reference.metal:946`), which is
§6.3's formula indexed by element rather than streamed by block.

## 6.8 Tradeoffs

**The n32 tile: measured occupancy over intuition.** The 16-row batch (the
DFlash verify shape, Chapter 8) had a real measured problem: the retained
kernels ran those projections at ~50–180 GB/s on an ~800 GB/s-class M3
Ultra because the `n_out/64` K-serial threadgroup shape carried 6–20 KiB of
threadgroup memory and could not fill a core `[ledger §Stage B L0]`. The
microbenchmark-first L-series iterated t128/t64/cross-sg designs and landed
on the weight-stationary `m16_q*k_n32` (32 output rows per threadgroup,
64-K stages, 6 KiB): verify-cycle matmul estimate **148.3 → 82.8–85.0 ms**
against the retained SGM tile (pinned `mul_mm` measured ~177)
`[ledger §Stage B L0]`, and the integrated verify forward fell 202.3 →
128.4 ms `[ledger §Stage B L1]`. Exactness was gated, not assumed: the tile
landed "within the accepted half-staged error envelope (max-abs identical
to the pinned kernels on the same data, zero argmax flips on every shape)"
`[ledger §Stage B L0]`. This is the house style: hypothesis (occupancy),
apparatus (`muser-m16-bench` at the exact Figure 6.6 shapes), then a gate.

**Q5_K at 5.5 bits on the lm_head — and the transcode temptation.** The
5-bit lm_head costs 924,571,648 B (§6.5 arithmetic) — 1.22× a Q4_K
equivalent (756,467,712 B). The Q5_0 sibling has an instructive in-tree
trade: `transcode_q5_to_q8` expands 5-bit blocks to Q8_0 at GPU upload
time, "trading 30 % more memory bandwidth for 3× fewer GPU decode
instructions" `[crates/muser-engine/src/quant/blocks.rs:100-103]` — the
same class of trade the n32 tile makes in reverse. On the live path the
lm_head stays Q5_K and rides `m16_q5k_n32` at 16 rows: 5.87 → 4.11 ms per
dispatch against `mul_mm` `[ledger §Stage B L0]`.

**Why not Q6_K everywhere?** By Figure 6.1, Q6_K is 6.5625/4.5 = 1.458×
the bytes of Q4_K. Every weight byte is read per token (§6.6), so
Q6_K-everywhere would scale the per-token stream by ~1.46× and, in the
bandwidth-bound regime Chapter 1 proved, cut throughput by the same
factor. The artifact spends 6.5625 bits only where the recipe says it pays
(ffn_down on half the layers, 26 k/v tensors — Figure 6.6); whether that
*specific* allocation is the quality optimum is the recipe designer's
claim, not a Muser measurement [unverified].

## 6.9 Where the gap lives

The quant format is not the decode gap. When the engine's one-token graphs
were reconciled closure-by-closure (the +196 dispatch-gap diagnosis), the
families were **104 norm-boundary groups, 39 SWA staging groups, 52
KV-publication splits, one copy** — not matvec arithmetic
`[docs/decode-dispatch-gap-20260815.md]`. The kquant matvecs themselves run
on llama.cpp's own pinned kernels for numerical parity
(`[crates/muser-engine/src/metal/encode.rs:278-280]`), so on this lane the
weight-format question was settled before it could become a gap: same
bytes, same kernels, same arithmetic order as the comparator. The lane's
measured decode ratios — 1.0274–1.0504× across the six-depth plain matrix,
five exact-token reps per depth `[docs/benchmarks.md §1]` — carry the
paradox this book keeps meeting: parity engines can still differ in
*dispatch*, and the gap lives there, not here.

## 6.10 What comes next

The kquant family is integer codebooks: uniform grids, local scales, a
min where the format wants one. Chapter 7 meets the other family — NVFP4,
where the 4-bit payload is itself a tiny *float*, the block scale is
itself a tiny float, and the whole lane was built for the same
bytes-per-token argument with a different arithmetic inside. You will pack
one of its groups by hand (using the repo's own fixture bytes), watch the
fail-closed loader bind its scale tensors, and read the Metal kernel that
keeps the whole contraction in integers — and you will see why this lane's
35.491 tok/s must never be called "faster" than kquant's 35.440.

---

## References

- `[crates/muser-engine/src/gguf/types.rs:9-127]` — `GgmlType`, block
  sizes/elements (Figure 6.1's source), the NVFP4_E2M1/F8_E4M3FN doc.
- `[crates/muser-engine/src/quant/k_block.rs:12-49]` — `dequant_q4_k`, this
  chapter's primary source (§6.2, §6.3).
- `[crates/muser-engine/src/quant/k_block.rs:51-105]` — `dequant_q5_k` and
  the 176-byte layout doc.
- `[crates/muser-engine/src/quant/k_block/q6.rs:15-75]` —
  `dot_q6_k_f32_llama`: the 210-byte layout (ql/qh/sc/d offsets),
  signed-code extraction, deferred-scaling order.
- `[crates/muser-engine/src/loader.rs:72-91]` — `weight_precision`:
  fail-closed lane selection (`q4_k_xl` default).
- `[crates/muser-engine/src/decode.rs:136-139]`, `:1209` — Metal-path
  dtype admissions for projections and the embedding.
- `[crates/muser-bench/src/m16.rs:137-226]` — the `SHAPES` table: dtypes,
  real dimensions, per-cycle multiplicities (Figure 6.6).
- `[crates/muser-engine/src/metal/encode/qkv.rs:414-641]` —
  `encode_quantized_matmul`: the full dispatch ladder (Figure 6.7),
  including the source-pinned boundary comment at `:476`.
- `[crates/muser-engine/src/metal/encode.rs:278-280]` — the pinned
  `kernel_mul_mv_q{4,5,6}_K_f32` metallib PSOs.
- `[crates/muser-engine/src/shaders/muse_reference.metal:735-788]` —
  `muser_matvec_q4k_4r2s`; `:41-62` `muser_decode_all_q4k_scales`;
  `:961-977` `muser_embedding_q4k`.
- `[crates/muser-engine/src/quant/blocks.rs:100-103]` — the Q5→Q8 transcode
  trade comment.
- `[crates/muser-engine/src/weights.rs:4-7]` — row-contiguity / prefill
  DRAM-amortization note.
- `[docs/decode-dispatch-gap-20260815.md]` — the +196-closure
  reconciliation (§6.9).
- `[ledger §Stage B L0]`, `[ledger §Stage B L1]` —
  `docs/goal-parity-ledger-2026-08.md`, the M16 microbenchmark and
  integration entries (§6.8's ms figures).
- `[docs/benchmarks.md §1]` — the six-depth plain matrix ratios.
- `[claims #11]` — kquant 35.440 / NVFP4 35.491 tok/s scope (§6.1, §6.9).
- [Ch 5](05-quantization-from-scratch.md) — the format-agnostic template
  this chapter instantiated.
- [ferrite-book Ch 5] — the ancestor Q4_K chapter; Muser's `dequant_q4_k`
  is the same extraction lineage (`NOTICE`, `docs/extraction-manifest.md`).
