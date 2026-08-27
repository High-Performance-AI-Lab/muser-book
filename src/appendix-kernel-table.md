# Appendix B — The kernel dispatch table

> **status:** draft  ·  **path:** Muse Glimmer, pinned Muser tree
>
> The decode path on one page: every kernel the engine dispatches for one
> token, in execution order, with the shader it comes from, the Rust wrapper
> that binds it, what it reads and writes, and the chapter that explains it.
> All `file:line` tags were verified against the pinned tree `6d0807da`
> (see [PINNED.md](PINNED.md)).

## B.1 The three kernel sources (legend)

Muser deliberately runs compute from **three libraries** (Ch 4,
`crates/muser-engine/src/metal/context.rs:59-131`). Every table row below
says which source its kernel comes from:

| Tag | Source | Built | Code |
|---|---|---|---|
| **S** | Serving concat library — 27 `.metal` files concatenated with `include_str!`, compiled at engine init with **fast-math ON** (MSL 3.1). Muser-authored `muse_reference.metal` + `nvfp4.metal` plus the ferrite-lineage extraction | runtime `new_library_with_source` | `context.rs:59-110`; 66-name `PIPELINES` registry `metal/encode.rs:21-88` |
| **X** | Strict-f32 cross-vendor library — the *same two* Muser source files (`muse_reference` + `nvfp4`), recompiled with **fast-math OFF** to match CUDA's explicit scalar boundaries for the exact/verification lanes | runtime `new_library_with_source` | `context.rs:111-121`; PSO fields `metal/encode.rs:205-277` |
| **L** | llama.cpp pinned metallib — prebuilt `.metallib` from llama.cpp commit `89e0aa6f…`, loaded from the `MUSER_GGML_METALLIB` path; supplies the ggml matvec/matmul/norm/rope/unary and `flash_attn_ext` kernels Muser refuses to re-express | `new_library_with_file` | `context.rs:122-131`; PSO picks `metal/encode.rs:278-293`; flash PSO table `LlamaFlashAttnPipelines` `metal/encode.rs:150-167` |

Rows tagged **X** run only under `MUSER_CROSS_VENDOR_QK` (Appendix C);
they are the strict arithmetic-ABI mirror, not the serving default.

## B.2 Per-layer decode chain (52 layers, execution order)

From `encode_token` (`crates/muser-engine/src/decode.rs:5515-5906`), the
teacher-forced single-token graph. The serving route packs 1..=4 resident
sequences through `forward_decode_group` → `encode_decode_group`
(`decode.rs:4869`, `:4954`), which mirrors this op sequence row for row
(every wrapper below appears there too, with `rows` 1..=4); where the
serving route differs, the row says so. Everything from embedding to
softcap is one Metal command buffer per token (concurrent dispatch type,
explicit barriers, `decode.rs:5448-5460`).

| # | Stage | Kernel / function | Src | Shader (file:line) | Rust dispatch (file:line) | Reads → writes (one line) | Ch |
|---|---|---|---|---|---|---|---|
| 1 | Embedding lookup | `muser_embedding_q4k` (F16 table: `muser_embedding_f16`) | S | `shaders/muse_reference.metal:961` (`nvfp4.metal:755`) | `encode_embedding_q4k` `metal/encode/qkv.rs:338` (bind `:365`); walk `decode.rs:5524` | one u32 token id + one Q4_K embedding row → the `[hidden_dim]` f32 residual stream | [11](chapters/11-token-embedding-lookup.md) |
| 2 | Entry norm (weight = ones) | llama `kernel_rms_norm_mul_f32_4`; fallback `rms_norm_batch` | L / S | llama metallib; `shaders/ferrite/rmsnorm_batch_tail.metal:1` | `encode_rms_norm_mul` `metal/encode/norm.rs:244` (bind `:270`); `decode.rs:5535` | hidden row → `normed` (RMS, eps 1e-5, × ones) | [12](chapters/12-rmsnorm-and-the-dual-eps-sandwich.md) |
| 3 | Attention pre-norm — **layer 0 only** (later layers receive it fused from the previous tail) | same as row 2 | L / S | same as row 2 | `encode_rms_norm_mul` `norm.rs:244`; `decode.rs:5553-5564` | `normed` → `post_norm` with `layer.attn_norm` | [12](chapters/12-rmsnorm-and-the-dual-eps-sandwich.md) |
| 4 | Q, K, V, gate projections — one concurrent set of 4 matvecs sharing the input row | llama `kernel_mul_mv_q4_K_f32` (Q5_K `q5_K`, Q6_K `q6_K`); fallbacks `muser_matvec_q4k_4r2s`, `muser_matvec_q5k_4sg`; NVFP4 `muser_nvfp4_w4a4_matvec_c1`; F16 `muser_f16_matvec_c1` | L / S | llama metallib (pick `metal/encode.rs:278-280`); `muse_reference.metal:735` / `:790`; `nvfp4.metal:312` / `:738` | `encode_projection` `decode.rs:6044` → `encode_quantized_matmul` `qkv.rs:414` (ggml bind `:439`, fallback bind `:464`) / `encode_nvfp4_matmul` `qkv.rs:128` / `encode_f16_matmul` `qkv.rs:68`; call `decode.rs:5569-5598` | `post_norm` row + the four weight matrices → `q`, `k`, `v`, `gate` activations | [13](chapters/13-the-qkv-gate-matvec-family.md) |
| 5 | Per-head QK-norm (parameterless; the ≈3.87 scale is folded into the norm weights) | same ggml/`rms_norm_batch` path as row 2; DFlash's own route uses `rms_norm_per_head` (B.6) | L / S | llama metallib; `rmsnorm_batch_tail.metal:1`; `ferrite/rms_norm_per_head.metal:15` (registry `encode.rs:58`) | `encode_qk_norm` `norm.rs:286` → `encode_rms_norm_mul` `norm.rs:244`; Q call `decode.rs:5600-5608`, K call `:5609-5617` | 128-wide head slices of `q` and `k`, normalized in place | [14](chapters/14-qk-norm-and-rope.md) |
| 6 | RoPE — **SWA layers only** (`uses_rope()`, `config.rs:68-70`); interleaved GPT-J pairs; NoPE full layers skip the dispatch entirely | `rope_norm_batch_cached` (or llama `rope_norm_f32` pinned PSO) | S / L | `shaders/ferrite/rope.metal:624` (plain `rope_batch_cached` `:566`) | `encode_rope_norm_batch_cached` `metal/encode/rope.rs:45` (ggml pick `:88-137`, bind `:139`); `decode.rs:5621-5640` | cached frequency table + `q`,`k` → rotated `q`,`k` in place | [14](chapters/14-qk-norm-and-rope.md) |
| 7a | KV store (token-major ring) + attention — **SWA, vec-eligible** | `muser_kv_store_f16`, then memory barrier, then llama `kernel_flash_attn_ext_vec_f16_dk128_dv128` (+ `flash_attn_ext_pad` when `visible % 32 ≠ 0`, + `flash_attn_ext_vec_reduce`) | S + L | `muse_reference.metal:979`; llama metallib (`LlamaFlashAttnPipelines` `encode.rs:150-167`) | `encode_kv_store_f16` `attn.rs:635` (bind `:647`); `encode_llama_flash_attn_decode_vec_f16` `attn.rs:437`; `decode.rs:5660-5693` | K,V rows → ring slot `write_physical`; then q + whole ring → `attention` | [15](chapters/15-kv-store-and-the-ring.md), [16](chapters/16-attention-decode-kernels.md) |
| 7b | Attention — **SWA fallback** (window not 32-aligned, or no metallib) | `muser_attention_decode_splitk_f16` + `muser_attention_decode_splitk_reduce_f32` | S | `muse_reference.metal:1052` + `:1169` | `encode_attention_decode_splitk_f16` `attn.rs:708` (binds `:748`, `:776`); `decode.rs:5695-5723` | q + ring (split-K partials per workgroup) → `attention`; geometry `splitk_geometry` `attn.rs:888-896` | [16](chapters/16-attention-decode-kernels.md) |
| 7c | KV store (head-major plane) + attention — **NoPE, vec-eligible** | `muser_kv_store_batch_f16` + barrier + llama vec kernel (ns10 = 128) | S + L | `muse_reference.metal:1203`; llama metallib | `encode_kv_store_batch_f16` `attn.rs:787` (bind `:812`); `attn.rs:437`; `decode.rs:5728-5770` | K,V rows → growing plane at `position`; q + plane → `attention` | [15](chapters/15-kv-store-and-the-ring.md), [16](chapters/16-attention-decode-kernels.md) |
| 7d | Attention — **NoPE fallback** | `flash_attn_decode_vec_f16_gqa_interleaved` + `flash_attn_decode_reduce_v2` | S | `shaders/ferrite/flash_attn_decode_vec_contiguous_f16.metal:494`; `flash_attn_decode_reduce_v2.metal:4` | `encode_ferrite_attention_decode_interleaved_f16` `attn.rs:189` (PSOs `encode.rs:298-313`); `decode.rs:5772-5790` | q + head-major plane (partials) → `attention`; also re-reads current `k`,`v` to dodge a store-load race | [16](chapters/16-attention-decode-kernels.md) |
| — | Route ladder predicates | `llama_vec_rows = (strict ‖ has_llama_flash_attn_vec) && len>0 && capacity≥32 && (origin_physical==0 ‖ len==capacity)`; `llama_swa = llama_vec_rows && len % 32 == 0` | — | — | `decode.rs:5645-5657` | — | [16](chapters/16-attention-decode-kernels.md) |
| 8 | Sigmoid attention-output gate | `sigmoid_gate_inplace` | S | `shaders/ferrite/sigmoid_gate.metal:7` | `encode_sigmoid_gate` `metal/encode/gate.rs:7` (bind `:17`); `decode.rs:5793-5799` | `attention` × sigmoid(`gate`) → `attention` in place | [17](chapters/17-sigmoid-gate-and-oproj.md) |
| 9 | o_proj matvec | same stack as row 4 | L / S | same as row 4 | `project` `decode.rs:5800` → `project_tokens` `:5909-5930` → `encode_projection` `:6044` | gated `attention` + `output` weights → `projected` | [17](chapters/17-sigmoid-gate-and-oproj.md) |
| 10 | Fused post-attention tail: residual add + post-norm (eps **1e-8**) + FFN-norm (eps 1e-5) | `muser_fused_norm_residual_rms_norm_32sg` | S | `shaders/ferrite/rmsnorm_batch_tail.metal:147` | `encode_fused_norm_residual_rms_norm_32sg` `norm.rs:163` (via `…_32sg_batch` `:190`, bind `:227`); `decode.rs:5806-5818` | `normed` (residual) + `projected` → `post_norm`; 32 SIMD groups, 1,024 threads, 144 B threadgroup | [12](chapters/12-rmsnorm-and-the-dual-eps-sandwich.md), [17](chapters/17-sigmoid-gate-and-oproj.md) |
| 11 | FFN gate+up — fused dual-read when `MUSER_FERRITE_FFN_GATE_UP` is set **and** both tensors are Q4_K (the release artifact is); else exact control | `ffn_q4k_gate_up_silu_4r2s`; control path: two row-4 matvecs + `muser_silu_mul_inplace` | S | `shaders/ferrite/ffn_fused_tail.metal:496`; `muse_reference.metal:4` | `encode_ffn_q4k_gate_up_silu_4r2s` `ffn.rs:10` (bind `:25`) / `encode_silu_mul` `ffn.rs:38` (bind `:49`); `decode.rs:5819-5862` | `post_norm` read once → SiLU(gate)·up written to `ffn_gate` | [18](chapters/18-swiglu-ffn.md) |
| 12 | ffn_down matvec | same stack as row 4 | L / S | same as row 4 | `decode.rs:5863-5868` → `:6044` | `ffn_gate` + `ffn_down` weights → `projected` | [19](chapters/19-downproj-and-residual.md) |
| 13 | Fused post-FFN tail: residual + post-FFN-norm (1e-8) + **next layer's** attn-norm (last layer: final norm) | `muser_fused_norm_residual_rms_norm_32sg` | S | `rmsnorm_batch_tail.metal:147` | `norm.rs:163`; next_norm selected `decode.rs:5869-5876`, dispatch `:5877-5889` | `normed` + `projected` → next layer's input (or `hidden` on layer 51) | [12](chapters/12-rmsnorm-and-the-dual-eps-sandwich.md), [19](chapters/19-downproj-and-residual.md) |

Serving-route variants of rows 2/10/13: the packed decode group uses the
same `…_32sg` kernel with `rows` up to 4 (`encode_fused_norm_residual_rms_norm_32sg_batch`, `norm.rs:190`, call `decode.rs:5128`, `:5202`);
the batch-prefill graph uses the rows-general
`muser_fused_norm_residual_rms_norm_batch_dual_eps`
(`rmsnorm_batch_tail.metal:250`, calls `decode.rs:4513`, `:4656`).
Under `MUSER_CROSS_VENDOR_QK` every fused tail decomposes into strict
cross-vendor kernels (**X**) with barriers at each model-dtype boundary
(`norm.rs:208-225`).

## B.3 Once-per-token tail

| # | Stage | Kernel / function | Src | Shader (file:line) | Rust dispatch (file:line) | Reads → writes | Ch |
|---|---|---|---|---|---|---|---|
| 14 | Final RMSNorm | fused into row 13's tail on the single-token graph (last layer's `next_norm` = `output_norm`, `decode.rs:5874-5875`); the decode-group path emits a separate norm | llama `kernel_rms_norm_mul_f32_4` / `rms_norm_batch` | L / S | fused: `decode.rs:5869-5888`; separate: `encode_rms_norm_mul` `decode.rs:5229-5241` | `hidden` → normed logits input | [20](chapters/20-final-norm-lm-head-softcap.md) |
| 15 | LM head matvec | same matvec stack as row 4 (kquant); NVFP4 lane: unquantized F16 head `muser_f16_matvec_c1` — the ~3.46 ms/token cost that keeps NVFP4 decode at parity, not faster | L / S | llama metallib; `nvfp4.metal:738` | `project` `decode.rs:5892-5897` → `:6044` / `qkv.rs:68` | `hidden` + vocab projection → `logits[vocab]` | [20](chapters/20-final-norm-lm-head-softcap.md) |
| 16 | Logit scale + soft cap | `muser_scale_softcap_inplace`; or, to match llama's graph literally, four ggml unary nodes (scale → tanh → scale) | S / L | `shaders/muse_reference.metal:15`; llama metallib unary (PSO pick `encode.rs:289-290`) | `encode_scale_softcap` `lmhead.rs:163` → `…_count` `:188` (ggml quartet `:230-259`); `decode.rs:5898-5905` | logits × `logit_scale` (= 1/√26 ≈ 0.196116, GGUF metadata, `config.rs:190-192`) then tanh at softcap 20, in place | [20](chapters/20-final-norm-lm-head-softcap.md) |
| 17 | Sampling read-back | argmax / MT19937 sampling on **CPU** over the read-back distribution; a GPU argmax pair exists for the no-readback benchmark lanes: `argmax_f32_phase{1,2}` + `greedy_argmax_f32_phase{1,2}` | S (GPU lanes) | `shaders/ferrite/argmax_f32.metal:7`, `:41`, `:77`, `:125` | CPU: `Session::decode` `api.rs:696-741` + `sampling.rs` (distribution buffer retained in place, `api.rs:700-703`); GPU: `encode_argmax_f32_rows` `lmhead.rs:83`, `encode_greedy_argmax_f32` `lmhead.rs:123` | full-vocab f32 row read back once per token (4 bytes out on the greedy GPU lane) | [21](chapters/21-sampling-argmax-and-grammar.md) |

## B.4 Prefill batch kernels (the second graph)

Prefill is *not* decode with more rows: it is a separate batch graph,
`Session::prefill` (`api.rs:634`) → `forward_batch` (`decode.rs:2857`) →
`forward_batch_hidden` (`:3788`) → `encode_batch_hidden_range`
(`:3858-4365`), chunked at `PREFILL_BATCH_TOKENS = 512` idle / 64 once a
decode waits (`decode.rs:53-54`, `:2095-2113`). Projections go through
`encode_batch_projection` (`decode.rs:5946-5980`); attention routes at
`decode.rs:4090-4365` (Ch 36):

| Stage | Kernel / function | Src | Shader (file:line) | Rust dispatch (file:line) | Reads → writes | Ch |
|---|---|---|---|---|---|---|
| Embedding (per chunk row) | `muser_embedding_q4k` | S | `muse_reference.metal:961` | `encode_embedding_q4k_from_u32_buffer` `qkv.rs:376`; `decode.rs:1756` | u32 token buffer row → batch hidden | [36](chapters/36-prefill-vs-decode-paths.md) |
| Projections, NVFP4 M16 route (16 rows, n_in % 64 == 0) | `muser_nvfp4_w4a4_quantize_m16` + `muser_nvfp4_w4a4_prequant_m16_n32` (B.5) | X | `nvfp4.metal:468` + `:504` | `encode_nvfp4_w4a4_prequant_m16` `qkv.rs:13`; picked `decode.rs:5955-5977` | activations quantized once, then one weight-stationary 32-row tile per projection | [36](chapters/36-prefill-vs-decode-paths.md) |
| Projections, small batch 4..=8 rows | llama `kernel_mul_mv_ext_{q4,q5,q6}_K_f32_r1_{2..5}` | L | llama metallib (`LlamaMulMvExtPipelines` `encode.rs:174-179`) | `encode_quantized_matmul` `qkv.rs:482-508` | the llama-pinned dispatch boundary; changing it breaks logprob parity (`qkv.rs:476-481`) | [36](chapters/36-prefill-vs-decode-paths.md) |
| Projections, 16-row K-quant blocks | `m16_q4k_n32` / `m16_q5k_n32` / `m16_q6k_n32` | S | `shaders/ferrite/batch_m16_n32.metal:59` / `:266` / `:163` | `encode_quantized_matmul` `qkv.rs:556-579` | DFlash verify/draft blocks; weight-stationary n32 tile, 6 KiB threadgroup | [36](chapters/36-prefill-vs-decode-paths.md), [33](chapters/33-speculation-and-the-distributed-verdict.md) |
| Projections, aligned Q4_K batches | `matmul_q4k_batch_sgm_aligned` | S | `shaders/ferrite/batch_sgm_q4_aligned.metal:69` | `encode_quantized_matmul` `qkv.rs:584-601` | Ferrite's accepted high-occupancy SIMD-group-matrix GEMM | [36](chapters/36-prefill-vs-decode-paths.md) |
| Projections, general batch | llama `kernel_mul_mm_q{4,5,6}_K_f32` (aligned/bounds); fallback `muser_matmul_q4k`/`_q5k` | L / S | llama metallib; `muse_reference.metal:912` / `:929` | `qkv.rs:604-640` | the roofline-flipped batch GEMM regime | [36](chapters/36-prefill-vs-decode-paths.md) |
| KV store, contiguous route | `muser_kv_store_batch_f16` | S | `muse_reference.metal:1203` | `attn.rs:787`; `decode.rs:4135`, `:4388` | chunk K,V rows → plane/ring before attention reads them back | [36](chapters/36-prefill-vs-decode-paths.md) |
| Attention (a): short chunks, < 20 queries | llama vec kernel, one unmasked launch per query row | L | llama metallib (`attn.rs:437` family) | `llama_vec_prefill_route_available` `decode.rs:65`; per-row launches `decode.rs:4153-4179` | q row + visible cache → one attention row | [36](chapters/36-prefill-vs-decode-paths.md) |
| Attention (b): NoPE at llama chunk bounds | `muser_fa_causal_mask_f16` + llama `flash_attn_ext_blk` (once per chunk) + llama `kernel_flash_attn_ext_f16_dk128_dv128` | S + L | `muse_reference.metal:1514`; llama metallib | `encode_llama_fa_prefill_mask_blk` `attn.rs:266` (binds `:283`, `:307`); `encode_llama_flash_attn_prefill_f16` `attn.rs:328`; `decode.rs:4197-4208` | causal f16 mask + skip/partial/dense block bytes, then the masked causal prefill kernel | [36](chapters/36-prefill-vs-decode-paths.md) |
| Attention (c): local FlashAttention-2 | `flash_attn_v2`; one-query GQA specialization `muser_flash_attn_decode_gqa_fa2` | S | `shaders/ferrite/flash_attn_v2.metal:59`; `flash_attn_decode_gqa_fa2.metal:39` | `encode_flash_attention_v2` `attn.rs:13` (specialization `:61-79`); `decode.rs:4227`, `:4329` | q chunk + f16 KV cache → attention chunk | [36](chapters/36-prefill-vs-decode-paths.md) |
| SWA ring wrap: staging shadow | `muser_stage_swa_prefill_f16` (chunked) / `muser_stage_swa_llama_decode_f16` (single-row, llama's 256-row-padded indices) | S | `muse_reference.metal:1240` / `:1281` | `encode_stage_swa_prefill_f16` `attn.rs:103`; `encode_stage_swa_llama_decode_f16` `attn.rs:145`; `decode.rs:4264`, `:4281` | old ring rows + new chunk → detached F16 shadow; ring metadata committed after (`append_batch`, `decode.rs:4348`) | [36](chapters/36-prefill-vs-decode-paths.md), [23](chapters/23-the-swa-ring-and-the-growing-cache.md) |
| NoPE prefill fallback | `muser_attention_prefill_flash_f16` | S | `muse_reference.metal:1409` | `encode_attention_prefill_f32` `attn.rs:830` (bind `:861`); `decode.rs:4359` | q chunk + current K,V + cache planes → attention chunk (one threadgroup per (head, token)) | [36](chapters/36-prefill-vs-decode-paths.md) |

Note the last row: the Rust wrapper is named
`encode_attention_prefill_f32`, but at the pin it binds
`muser_attention_prefill_flash_f16` (`attn.rs:861`). A sibling kernel
`muser_attention_prefill_f32` (`muse_reference.metal:1326`) exists in the
registry but is not dispatched by this route (see B.8, conflict 3).

## B.5 NVFP4 lane kernels (native 4-bit weights)

All in `shaders/nvfp4.metal` (Muser-authored); the W4A4 integer
contraction and its two scalar epilogue multiplies compile in the
**no-fast-math** library (**X**) so they match the ModelOpt/MLX integer
order (`qkv.rs:203-205`, Ch 7). Dispatch: `encode_nvfp4_matmul`
(`qkv.rs:128`) picks by activation scale presence and column count.

| Kernel | Src | Shader (file:line) | Role | Ch |
|---|---|---|---|---|
| `muser_nvfp4_matvec_c{1,2,4,8,16}` | X | `nvfp4.metal:226` | plain dequantizing NVFP4 matvec; width-1 is the decode kernel, wider calls cover DFlash verification and bounded prefill | [7](chapters/07-nvfp4-native-lane.md) |
| `muser_nvfp4_a16_q8_matvec` | X | `nvfp4.metal:622` | weight-only W4A16 route (no `input_scale_inv`): activations dynamically quantized to Q8 per 16-block, n_in % 256 | [7](chapters/07-nvfp4-native-lane.md) |
| `muser_nvfp4_w4a4_matvec_c{1,2,4,8,16}` | X | `nvfp4.metal:312` | W4A4 integer-dot matvec family (weight-stationary 2/4-column specializations) | [7](chapters/07-nvfp4-native-lane.md) |
| `muser_nvfp4_w4a4_m16_n32` | X | `nvfp4.metal:337` | 16-column weight-stationary tile, N=32 output rows | [7](chapters/07-nvfp4-native-lane.md), [36](chapters/36-prefill-vs-decode-paths.md) |
| `muser_nvfp4_w4a4_quantize_m16` + `muser_nvfp4_w4a4_prequant_m16_n32` | X | `nvfp4.metal:468` + `:504` | exact two-pass M=16 route: quantize activations once, then tile — the prefill/verify pair (disabled by `MUSER_NO_M16_N32`) | [36](chapters/36-prefill-vs-decode-paths.md) |
| `muser_f16_matvec_c{1,2,4,8,16}` | X | `nvfp4.metal:738` | F16 weights on the NVFP4 lane, incl. the unquantized LM head | [7](chapters/07-nvfp4-native-lane.md) |
| `muser_embedding_f16` | X | `nvfp4.metal:755` | F16 embedding table lookup (row 1's F16 branch) | [7](chapters/07-nvfp4-native-lane.md) |
| `muser_nvfp4_dequant_fixture` | X | `nvfp4.metal:775` | test fixture: bit-exact dequant of every finite E4M3FN scale | [7](chapters/07-nvfp4-native-lane.md) |

## B.6 DFlash draft kernels (speculative lane)

The five-layer draft runs its own graph in `metal/dflash.rs` (Ch 8).
Prepared-greedy layer loop `metal/dflash.rs:1061-1180` (per layer ×5):

| Stage | Kernel | Src | Shader (file:line) | Rust dispatch (file:line) | Ch |
|---|---|---|---|---|---|
| Input norm | `rms_norm_batch` (ggml rms_norm when present) | L / S | `rmsnorm_batch_tail.metal:1` | `encode_rms_norm_mul` `norm.rs:244`; `metal/dflash.rs:1063` | [8](chapters/08-the-dflash-draft.md) |
| q/k/v projections | dense f32 `matmul_f32_batch_tiled` (batch ≥ 4) / `matmul_f32_batch`; kquant sidecar → the B.4 batch stack incl. `m16_*_n32` blocks | S | `batch_f32_support.metal:45` / `:7` | `encode_projection` `metal/dflash.rs:382` → `encode_dense_f32_batch` `encode.rs:443` / `encode_quantized_matmul` `qkv.rs:408` | [8](chapters/08-the-dflash-draft.md) |
| QK-norm (draft is Qwen-style: real weights, not folded scales) | `rms_norm_per_head` | S | `ferrite/rms_norm_per_head.metal:15` | `encode_rms_norm_per_head` `encode.rs:589` (bind `:605`); `metal/dflash.rs:1085-1101` | [8](chapters/08-the-dflash-draft.md) |
| RoPE — NeoX pairing, *not* the target's interleaved pairs | `rope_batch_cached` | S | `ferrite/rope.metal:566` | `encode_rope_neox_batch_cached` `encode.rs:638` (bind `:674`); `metal/dflash.rs:1107-1114` | [8](chapters/08-the-dflash-draft.md) |
| Dual-context attention (64-row sink + sliding window) | `dflash_dual_attention_f32` | S | `ferrite/attention_dflash_dual.metal:15` | `encode_dflash_dual_attention` `encode.rs:690`; `metal/dflash.rs:1119` | [8](chapters/08-the-dflash-draft.md) |
| o_proj → fused residual+norm | `fused_residual_rms_norm_batch` | S | `rmsnorm_batch_tail.metal:72` | `encode_fused_residual_norm` `encode.rs:554` (bind `:578`); `metal/dflash.rs:1143` | [8](chapters/08-the-dflash-draft.md) |
| gate/up projections → SwiGLU | `silu_hadamard_batch` | S | `batch_ffn_activation_tail.metal:14` | `encode_silu_hadamard_batch` `encode.rs:482`; `metal/dflash.rs:1165` | [8](chapters/08-the-dflash-draft.md) |
| down projection → residual add | `residual_add_batch` | S | `batch_f32_support.metal:155` | `encode_residual_add_batch` `encode.rs:500`; `metal/dflash.rs:1178` | [8](chapters/08-the-dflash-draft.md) |
| Final norm | `rms_norm_batch_inplace` | S | `rmsnorm_batch_tail.metal:42` | `encode_rms_norm_inplace` `encode.rs:525` (bind `:531`); `metal/dflash.rs:1185` | [8](chapters/08-the-dflash-draft.md) |
| Capture pack (target hidden states → draft input) | `pack_dflash_layer_major_f32`; `copy_f32_buffer` | S | `copy_f32_buffer.metal:18`; `:3` | `encode_pack_dflash_layer_major` `encode.rs:769`; `metal/dflash.rs:744` | [8](chapters/08-the-dflash-draft.md) |
| Verify side (target, 16-row blocks) | `m16_q{4,5,6}k_n32` (B.4) — the L-series tile that took the 16-row verify matmul from ~148 to ~83 ms/cycle | S | `batch_m16_n32.metal:59/163/266` | `encode_quantized_matmul` `qkv.rs:556-579` via the mirror-SD suffix `begin_dflash_verify_suffix` `decode.rs:3298` | [33](chapters/33-speculation-and-the-distributed-verdict.md) |
| Verify side, opt-in multi-column matvec | `muser_matvec_multicol_{q4k,q5k,q6k}_c{1,2,4}` (macro) | S | `ferrite/matvec_multicol.metal:400` | `MultiColPipelines` `multicol.rs:84-103`; gate `MUSER_MULTI_COL_VERIFY` `multicol.rs:70-82`; also the decode-group route `encode_quantized_decode_group` `qkv.rs:234` | [33](chapters/33-speculation-and-the-distributed-verdict.md) |

## B.7 Strict cross-vendor kernels (X, `MUSER_CROSS_VENDOR_QK`)

For reference: `muser_cross_vendor_{q4k,q5k,q6k}` (projections,
`muse_reference.metal:73/146/221`), `muser_cross_vendor_rms_per_head`
(`:301`), `_rms_unweighted` (`:332`) + `_mul_weight` (`:358`),
`_swiglu` (`:414`), `_scale` (`:433`), `_tanh` (`:442`), `_rope`
(`:457`) / `_rope_neox` (`:497`), `_attention_decode` (`:555`),
`_attention_prefill` (`:617`), `_sigmoid_gate` (`:680`),
`_dual_norm_residual` (`:690`), `_residual_add` (`:724`) — all in
`muse_reference.metal`, all compiled fast-math OFF (`context.rs:111-121`).
These replace their **S**-tagged serving counterparts row-by-row when the
flag is set (e.g. `decode.rs:5645`, `norm.rs:208-225`, `gate.rs:14`).

## B.8 Conflicts found while building this table

Where the research map, a chapter, and the pinned tree disagreed, the tree
won; the disagreements:

1. **QK-norm kernel (row 5).** The research map lists `rms_norm_per_head`
   as the decode-path QK-norm kernel. At the pin, the target route
   (`encode_qk_norm`, `norm.rs:286`) delegates to the *same* ggml
   rms-norm path as row 2 (or the cross-vendor decomposition);
   `rms_norm_per_head` lives in the registry (`encode.rs:58`) and is the
   **DFlash** QK-norm kernel (`encode.rs:589`). Ch 14 already words this
   correctly ("exists in the pipeline registry"); the table records the
   tree's behavior.
2. **Sigmoid gate (row 8).** Ch 17 quotes `muse_reference.metal:680`,
   which at the pin is `muser_cross_vendor_sigmoid_gate` (the strict
   variant). The live serving kernel is `sigmoid_gate_inplace`
   (`ferrite/sigmoid_gate.metal:7`, dispatched `gate.rs:17`); the
   cross-vendor kernel is the **X** mirror of the same math. Both are
   recorded.
3. **NoPE prefill fallback (B.4 last row).** The research map says the
   fallback dispatches `muser_attention_prefill_f32`
   (`muse_reference.metal:1326`). At the pin the wrapper
   `encode_attention_prefill_f32` binds
   `muser_attention_prefill_flash_f16` (`attn.rs:861`,
   `muse_reference.metal:1409`); `muser_attention_prefill_f32` is in the
   66-name registry (`encode.rs:42`) but has no dispatch on this route.
4. **Chapter in-body line tags.** Several chapters cite line numbers
   *inside* a kernel body rather than its `kernel void` line (Ch 11
   `muse_reference.metal:973` for a kernel that starts `:961`; Ch 15
   `:1224` vs kernel start `:1203`; Ch 16
   `flash_attn_decode_vec_contiguous_f16.metal:519` vs `:494`). All
   resolve within the quoted kernel; this table cites the `kernel void`
   lines.

---

*What comes next: the lane matrix and every `MUSER_*` flag an operator
can actually meet — [Appendix C](appendix-env-flags.md).*
