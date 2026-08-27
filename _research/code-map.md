# Muser engine code map

Pinned commit `6d0807da975d3628f874df6b36ac9cc2af3723f2` (clean tree). All
paths relative to `<muser-checkout>/`. Line numbers verified at this
commit. Where the book outline disagreed with code, code wins and the
discrepancy is called out inline.

---

## 1. Workspace overview

`Cargo.toml:4-10` — five crates, `third_party/kvpack` and `third_party/metal`
excluded from the workspace. Workspace deps pin kvpack crates by path +
exact version `=0.1.0-alpha.2` (`Cargo.toml:19-21`).

- **muser-engine** (`crates/muser-engine/src/lib.rs:1-47`) — the Muse Glimmer
  52-layer ~30B forward path: CPU f32 oracle + Metal decode/prefill drivers,
  GGUF loader/tokenizer, DFlash speculative assistant, sampling. Key modules:
  `config.rs` (architecture constants/layer kinds), `reference.rs` (CPU oracle
  = "the correctness gate", lib.rs:99-109), `decode.rs` (7638 lines: Metal
  single-token decode, batched decode, batched prefill, KV planes, scheduler),
  `metal/` (context, buffers, per-op encoders), `quant/` (K-quant dot kernels
  + `nvfp4.rs` CPU oracle), `dflash/` (draft model + speculative engine),
  `sampling.rs`, `gguf/`, `tokenizer/`, `cache.rs` (kvpack interchange),
  `weights.rs`, `loader.rs`, `vision.rs` (50-block mtmd port, lib.rs:133-135),
  `coreml.rs`/`dflash_ane.rs`/`target_ane.rs` (feature `ane-coreml`).
- **muser-server** (`crates/muser-server/src/main.rs`) — Axum HTTP server,
  OpenAI/Ollama-compatible chat + completions, session store, slots, grammar,
  metrics, dashboard, TLS, remote-prefill client. Biggest files:
  `openai.rs` (6845), `axum_httpd.rs` (5344), `state.rs` (2014).
- **muser-kvpack** (`crates/muser-kvpack/src/lib.rs:1-40`) — thin wrapper over
  the vendored kvpack snapshot adding Muse K1/K3 layout glue (`layout.rs`),
  durable session save/restore (`session.rs`), resident radix reuse
  (`resident.rs`, `reuse.rs`), remote prefix acquisition (`remote.rs`), and
  cache-economics accounting (`economics.rs`).
- **muser-cluster** (`crates/muser-cluster/src/lib.rs:1-18`) — authenticated
  GX10 disaggregated prefill: mTLS transport, Handoff V2 receiver, transfer
  amortization schedule, control plane, replay ledger, remote speculative
  verifiers. "CUDA prefill -> authenticated Handoff V2 tiles -> Metal
  scatter-on-arrival -> atomic commit -> Metal decode."
- **muser-bench** (`crates/muser-bench/Cargo.toml`) — 19 `[[bin]] binaries
  (§12), fail-closed benchmark executors; `default = []` — Metal binaries need
  `--features metal` or fail at runtime (AGENTS.md).

---

## 2. The decode path, step by step

Model facts asserted in `lib.rs:7-15`: 52 layers = 39 SWA (window 2048) + 13
NoPE full layers `{3,7,...,51}`; GQA 32:2, head_dim 128; parameterless QK-RMSNorm;
sigmoid attention-output gate; Gemma-2-style sandwich norms with dual eps
(1e-5 rms / **1e-8 post**, `config.rs:28`); softcap 20 logits; dual EOS.

### 2.1 Entry chain (server → one token)

1. HTTP handler → `InferenceRuntime` slot session (`state.rs:221-251`).
2. `Session::decode` — `api.rs:696-741`: validates token, calls
   `Session::forward_into` (`api.rs:1823-1829`) which dispatches on backend:
   CPU `MuseModel::forward` (`reference.rs:222`) or Metal
   `MetalMuseModel::forward_into`.
3. `MetalMuseModel::forward_into` — `decode.rs:2077-2116`. **Single-token
   decode is routed through the one-row *batch* graph** (`forward_batch`,
   `decode.rs:2092`) because the legacy fused one-token graph diverges from
   the pinned llama Metal graph at public-logprob tolerance (comment at
   `decode.rs:2085-2091`). Multi-token (prefill) input is chunked at
   `PREFILL_BATCH_TOKENS = 512` (`decode.rs:53`), shrinking to
   `MAX_TEACHER_FORCED_TOKENS = 64` (`decode.rs:54`) when a decode is queued
   (`decode.rs:2097-2113`).
4. Packed multi-sequence decode: `forward_decode_group` (`decode.rs:4869`)
   accepts 1..=4 models + tokens sharing one `MetalShared` executor, encodes
   with `encode_decode_group` (`decode.rs:4954`), one concurrent encoder +
   one commit + one wait (`decode.rs:4920-4937`).
5. The legacy single-token graph survives as `forward_token` →
   `encode_token` (`decode.rs:5432`, `5515`) — still the teacher-forced and
   phase-profile route; kernel sequence below is from `encode_token`, which
   `encode_decode_group` mirrors batch-row-for-batch-row.
6. Streaming: `forward_greedy_streaming` (`decode.rs:1626`).

Scheduler acquire happens before every graph:
`AcceleratorScheduler::acquire(seq, AcceleratorWork::Decode)` (§10).

### 2.2 Per-token GPU kernel sequence (exact order, `encode_token`, decode.rs:5515-5906)

| # | Op | Rust wrapper (file:line) | Kernel (shader file:line) |
|---|-----|--------------------------|---------------------------|
| 1 | Embedding lookup (Q4_K table) | `encode_embedding_q4k` `metal/encode/qkv.rs:338` | `muser_embedding_q4k` `shaders/muse_reference.metal:961` |
| 2 | Entry norm (weight = ones) | `encode_rms_norm_mul` `metal/encode/norm.rs:244` → binds `rms_norm_batch` or ggml pinned PSO | `rms_norm_batch` `shaders/ferrite/rmsnorm_batch_tail.metal:1`; llama-pinned `rms_norm` via metallib (norm.rs:253-262) |
| — | Per layer 0..52: | | |
| 3 | Attention pre-norm (layer 0 only; later layers come fused from previous tail) | `encode_rms_norm_mul` norm.rs:244 | same as #2 |
| 4 | Q/K/V/gate projections (one concurrent set of 4 matvecs) | `encode_projection` decode.rs:6044 → `encode_quantized_matmul` qkv.rs:414 / `encode_f16_matmul` qkv.rs:68 / `encode_nvfp4_matmul` qkv.rs:128 | ggml metallib `kernel_mul_mv_q{4,5,6}_K_f32` (encode.rs:278-280) or `muser_matvec_q4k_4r2s` muse_reference.metal:735 / `muser_matvec_q5k_4sg` :790; F16 `muser_f16_matvec_c*` nvfp4.metal:738 |
| 5 | Per-head QK-norm (parameterless; scale folded) | `encode_qk_norm` norm.rs:286 (delegates to #2's path or cross-vendor) | `rms_norm_per_head` rms_norm_per_head.metal:15 (fused variant :59) |
| 6 | RoPE — **only on SWA layers** (`uses_rope()`, config.rs:68-70); interleaved GPT-J pair style; NoPE full layers skip entirely | `encode_rope_norm_batch_cached` `metal/encode/rope.rs:45` | `rope_norm_batch_cached` `shaders/ferrite/rope.metal:624` (plain `rope_batch_cached` :566) |
| 7 | KV store + attention (see 2.3) | see 2.3 | see 2.3 |
| 8 | Sigmoid output gate: `attn_out *= sigmoid(gate_proj)` | `encode_sigmoid_gate` `metal/encode/gate.rs:7` | `sigmoid_gate_inplace` `shaders/ferrite/sigmoid_gate.metal:7` |
| 9 | o_proj matvec | `project`/`project_tokens` decode.rs:5909-5930 → #4 stack | same as #4 |
| 10 | Fused post-attn: residual add + post-norm(eps 1e-8) + ffn-norm(eps 1e-5) | `encode_fused_norm_residual_rms_norm_32sg` norm.rs:163 | `muser_fused_norm_residual_rms_norm_32sg` `shaders/ferrite/rmsnorm_batch_tail.metal:147` |
| 11 | FFN gate+up: fused Q4_K dual read when enabled (`MUSER_FERRITE_FFN_GATE_UP`, decode.rs:5819-5836), else two matvecs + `encode_silu_mul` ffn.rs:38 → `muser_silu_mul_inplace` muse_reference.metal:4 | `encode_ffn_q4k_gate_up_silu_4r2s` ffn.rs:10 | `ffn_q4k_gate_up_silu_4r2s` `shaders/ferrite/ffn_fused_tail.metal:496` |
| 12 | ffn_down matvec | #4 stack | same as #4 |
| 13 | Fused post-FFN tail: residual + post-ffn-norm(1e-8) + **next layer's** attn-norm | `encode_fused_norm_residual_rms_norm_32sg` norm.rs:163 (next_norm selected at decode.rs:5869-5876) | same kernel as #10 |
| — | After layer 51: | | |
| 14 | Final norm | fused into last tail (decode.rs:5875) or `encode_rms_norm_mul` decode.rs:5229-5241 | same as #2 |
| 15 | lm_head matvec | `project` decode.rs:5892-5897 | #4 stack |
| 16 | Logit scale × `logit_scale` (= 1/√26 ≈ 0.196116, from GGUF `muse-glimmer.logit_scale`, config.rs:190-192; provenance docs/release-provenance.md:823) then softcap tanh@20 | `encode_scale_softcap` `metal/encode/lmhead.rs:163` | `muser_scale_softcap_inplace` `shaders/muse_reference.metal:15` (CPU oracle `reference.rs:562-568`; fixed test value `metal.rs:68`) |

Argmax / sampling run **on CPU** on the read-back logits
(`api.rs:696-741`, `sampling.rs`); a GPU argmax pair exists for
benchmark/no-readback routes (`encode_argmax_f32_rows` lmhead.rs:83,
`encode_greedy_argmax_f32` lmhead.rs:123 → `argmax_f32_phase{1,2}` +
`greedy_argmax_f32_phase{1,2}` `shaders/ferrite/argmax_f32.metal:7,41,77,125`).
Everything from embedding to softcap is one Metal command buffer per token
(concurrent dispatch type, explicit barriers, decode.rs:5448-5460).

### 2.3 Attention routing per layer kind (decode.rs:5643-5792)

Route selection: `llama_vec_rows = (strict || has_llama_flash_attn_vec()) &&
len>0 && capacity>=32 && (origin_physical==0 || len==capacity)`;
`llama_swa = llama_vec_rows && len % 32 == 0` (decode.rs:5646-5657).

- **SWA layer (39), vec-eligible**: `encode_kv_store_f16` (attn.rs:635 →
  `muser_kv_store_f16` muse_reference.metal:979), memory barrier, then
  `encode_llama_flash_attn_decode_vec_f16` (attn.rs:437 → llama.cpp pinned
  metallib `flash_attn_ext_vec` family, PSO table `LlamaFlashAttnPipelines`
  encode.rs:150-167).
- **SWA layer, fallback**: kv_store then
  `encode_attention_decode_splitk_f16` (attn.rs:708 →
  `muser_attention_decode_splitk_f16` muse_reference.metal:1052 + reduce
  `muser_attention_decode_splitk_reduce_f32` :1169), token-major ring.
- **NoPE full layer (13), vec-eligible**: head-major plane;
  `encode_kv_store_batch_f16` (attn.rs:787 → `muser_kv_store_batch_f16`
  muse_reference.metal:1203) + llama vec kernel.
- **NoPE fallback**: `encode_ferrite_attention_decode_interleaved_f16`
  (attn.rs:189 → `flash_attn_decode_vec_f16_gqa_interleaved`
  `shaders/ferrite/flash_attn_decode_vec_contiguous_f16.metal:494`; reduce
  `flash_attn_decode_reduce_v2.metal:4`).

Attention scale is `1/sqrt(128)` (config.rs:279-281), independent of the
qk_scale_factor ≈ 3.87 folded into the Q-norm weights (config.rs:139-141,
test config.rs:426-430).

### 2.4 CPU vs GPU

GPU: entire token graph (embedding → softcap). CPU: tokenization, route
selection, scheduler, sampling/argmax over one vocab read-back per token
(`Session::decode` retains and refills the distribution buffer in place,
api.rs:700-703), speculative acceptance, detokenization. The CPU f32 oracle
(`reference.rs:222 forward`, kept verbatim per lib.rs:99-109) is the parity
spec; `tests/muse_golden.rs` diffs node-by-node via `capture.rs`.

---

## 3. Prefill path

`prefill.rs` is a 17-line module doc — the implementation lives in decode.rs
("The implementation lives beside decode" prefill.rs:4-5).

- `Session::prefill` (api.rs:634) → `forward_into` chunking (decode.rs:2095-2113):
  512-row chunks idle, 64-row once a decode waits.
- `forward_batch` (decode.rs:2857) → `forward_batch_hidden` (3788) →
  `encode_batch_hidden_range` (3858-4365+). Embedding is done by
  `encode_embedding_q4k_from_u32_buffer` (qkv.rs:376).
- Per layer, projections run through `encode_batch_projection`
  (decode.rs:5946-5980): token_count==16 + NVFP4 + n_in%64==0 dispatches
  `encode_nvfp4_w4a4_prequant_m16` (qkv.rs:13 →
  `muser_nvfp4_w4a4_prequant_m16_n32` nvfp4.metal:504 / `batch_m16_n32.metal`).
- Attention routes (decode.rs:4090-4365):
  - `flash_contiguous` (origin 0, no wrap, fits): store via
    `encode_kv_store_batch_f16`, then (a) short batches (<20 queries,
    `llama_vec_prefill_route_available` decode.rs:65): one unmasked llama vec
    launch per query row (decode.rs:4153-4179); (b) NoPE at llama chunk
    bounds: `encode_llama_fa_prefill_mask_blk` (attn.rs:266 → metallib
    `flash_attn_ext_blk`) once + `encode_llama_flash_attn_prefill_f16`
    (attn.rs:328 → metallib `flash_attn_ext_f16_dk128_dv128`, constants
    LLAMA_FA_PREFILL_* encode.rs:1066-1072); (c) else local
    `encode_flash_attention_v2` (attn.rs:13 → `flash_attn_v2`
    `shaders/ferrite/flash_attn_v2.metal:59`; one-query GQA specialization
    `muser_flash_attn_decode_gqa_fa2` flash_attn_decode_gqa_fa2.metal:39).
  - SWA after ring wrap: stage old ring rows + chunk into a detached F16
    shadow (`encode_stage_swa_prefill_f16` attn.rs:103 →
    `muser_stage_swa_prefill_f16` muse_reference.metal:1240; single-row llama
    variant `encode_stage_swa_llama_decode_f16` attn.rs:145 →
    `muser_stage_swa_llama_decode_f16` :1281), attend from the shadow, then
    commit ring metadata (`append_batch`, decode.rs:4348).
  - NoPE fallback: `encode_attention_prefill_f32` (attn.rs:830 →
    `muser_attention_prefill_f32` muse_reference.metal:1326).
- Mac-local prefill is also the fallback when GX10 disaggregated prefill
  isn't available (prefill.rs:10-12). Remote install entry:
  `Session::begin_remote_kv_install` api.rs:999 → decode.rs:1852/1908,
  commit decode.rs:1981/1990.

---

## 4. Shader inventory (all 30 .metal files under crates/)

27 under `crates/muser-engine/src/shaders/ferrite/` (ferrite-lineage unless
noted), 2 Muser-authored at `shaders/`, 1 bench-only. Runtime compilation:
`MetalContext::new` concatenates sources with `include_str!` and
`newLibraryWith_source` (context.rs:59-110, fast-math on, language 3.1); a
second strict-f32 `cross_vendor_library` compiles only muse_reference + nvfp4
(context.rs:111-121); llama-pinned PSOs load from a prebuilt metallib via
`MUSER_GGML_METALLIB` (context.rs:122-131). Kernel registry:
`PIPELINES: [&str; 66]` encode.rs:21-88.

| Path (crates/muser-engine/src/shaders/…) | Kernels (line) | Purpose |
|---|---|---|
| `muse_reference.metal` (Muser-authored, ~1530 lines) | `muser_silu_mul_inplace` 4; `muser_scale_softcap_inplace` 15; `muser_cross_vendor_q4k/q5k/q6k` 73/146/221; `muser_cross_vendor_rms_per_head` 301; `rms_unweighted` 332; `mul_weight` 358; `swiglu` 414; `scale` 433; `tanh` 442; `rope` 457; `rope_neox` 497; `attention_decode` 555; `attention_prefill` 617; `sigmoid_gate` 680; `dual_norm_residual` 690; `residual_add` 724; `muser_matvec_q4k_4r2s` 735; `muser_matvec_q5k_4sg` 790; `muser_matmul_q4k` 912; `q5k` 929; `muser_embedding_q4k` 961; `muser_kv_store_f16` 979; `muser_attention_decode_f32` 994; `muser_attention_decode_splitk_f16` 1052; `splitk_reduce_f32` 1169; `muser_kv_store_batch_f16` 1203; `muser_stage_swa_prefill_f16` 1240; `muser_stage_swa_llama_decode_f16` 1281; `muser_attention_prefill_f32` 1326; `muser_attention_prefill_flash_f16` 1409; `muser_copy_row_f32` 1499; `muser_fa_causal_mask_f16` 1514 | The serving-graph kernels Muser wrote for the fixed Muse driver + strict-f32 cross-vendor oracle kernels (CUDA-parity integer/scalar routes). |
| `nvfp4.metal` (Muser-authored) | `muser_nvfp4_matvec_c{1,2,4,8,16}` 226; `muser_nvfp4_w4a4_matvec_c*` 312; `muser_nvfp4_w4a4_m16_n32` 337; `muser_nvfp4_w4a4_quantize_m16` 468; `muser_nvfp4_w4a4_prequant_m16_n32` 504; `muser_nvfp4_a16_q8_matvec` 622; `muser_f16_matvec_c*` 738; `muser_embedding_f16` 755; `muser_nvfp4_dequant_fixture` 775 | Native NVFP4 E2M1 lane: weight W4 + activation A4/A16-Q8 paths, E4M3FN block scales, integer contraction matching ModelOpt/MLX order. |
| `ferrite/_q4k_helpers.metal` | (no kernels; `q4k_v4_dual_row_mac` 34, `q4k_v4_single_row_mac` 108) | Shared static-inline Q4_K dequant+MAC helpers; must precede callers in the concat. |
| `ferrite/argmax_f32.metal` | `argmax_f32_phase1/2` 7/41; `greedy_argmax_f32_phase1/2` 77/125 | Two-phase GPU argmax for no-readback benchmark lanes. |
| `ferrite/attention_dflash_dual.metal` | `dflash_dual_attention_f32` 15 | DFlash dual-context (sink+window) attention. |
| `ferrite/batch_f32_support.metal` | `matmul_f32_batch` 7; `_tiled` 45; `_tiled_8x8` 99; `residual_add_batch` 155 | Dense f32 batch matmuls (DFlash/growth paths). |
| `ferrite/batch_ffn_activation_tail.metal` | `silu_hadamard_batch` 14 | Batched SwiGLU tail. |
| `ferrite/batch_m16_n32.metal` | `m16_q4k_n32` 59; `m16_q6k_n32` 163; `m16_q5k_n32` 266 | 16-token-batch K-quant GEMMs (nvfp4-lane M16 route shares these shapes). |
| `ferrite/batch_sgm_q4_aligned.metal` | `matmul_q4k_batch_sgm_aligned` 69; `matmul_q4k_batch_sgm_b16_aligned` 215 | SIMD-group-matrix Q4_K batch matmul. |
| `ferrite/copy_f32_buffer.metal` | `copy_f32_buffer` 3; `pack_dflash_layer_major_f32` 18 | Exact Ferrite copies; layer-major pack for DFlash capture. |
| `ferrite/ffn_fused.metal` | `ffn_gate_up_silu` 45; `ffn_q8_gate_up_silu` 90; `_tiled` 144; `_normed` 256 | Fused gate+up+SiLU FFN family (Q8 era). |
| `ferrite/ffn_fused_normed_quant.metal` | `ffn_q5_gate_up_silu_normed` 1; `q5_scalar` 106; `ffn_q5_1_...` 190; `ffn_q4k_gate_up_silu_normed` 296 | Normed-input fused FFN variants. |
| `ferrite/ffn_fused_q4k_hidden.metal` | `ffn_q4k_gate_up_silu_normed_hidden_4row` 1; `_4row_v4` 111; `hidden_v5` 245; `hidden_4row_v4` 421; `hidden_4row_v4_4sg` 540 | Hidden-input Q4_K fused FFN experiments. |
| `ferrite/ffn_fused_tail.metal` | `ffn_q4k_gate_up_silu_normed_4row` 1; `ffn_q5k_gate_up_silu_normed` 113; `ffn_q4k_gate_up_silu` 213; `_4sg` 295; `_4sg_tgcache` 392; **`ffn_q4k_gate_up_silu_4r2s` 496**; `ffn_q5k_gate_up_silu` 568 | 4r2s = the live decode FFN kernel (opt-in via `MUSER_FERRITE_FFN_GATE_UP`, decode.rs:5819-5836). |
| `ferrite/flash_attn_decode_gqa_fa2.metal` | `muser_flash_attn_decode_gqa_fa2` 39 | One-query FA2 specialization: 8 Q-heads/thread, 16 Q per KV head. |
| `ferrite/flash_attn_decode_prelude.metal` | (no kernels; function-constant ABI + `DecodeParams` struct) | Shared prelude copied from Ferrite a85048a90 for the vec kernel family. |
| `ferrite/flash_attn_decode_reduce_v2.metal` | `flash_attn_decode_reduce_v2` 4 | Partial-combine reducer for the vec decode kernel. |
| `ferrite/flash_attn_decode_vec.metal` | `flash_attn_decode_vec_q8` 230; `flash_attn_decode_vec_geoprecision_dense` 424 | Q8/geo-precision vec decode attention (research lanes). |
| `ferrite/flash_attn_decode_vec_contiguous_f16.metal` | `flash_attn_decode_vec_f16_v2` 1; `_gqa` 161; `_gqa_v2` 320; **`_gqa_interleaved` 494**; `_gqa_ilp4` 680 | F16 contiguous/interleaved NoPE decode attention; interleaved is the live fallback. |
| `ferrite/flash_attn_prefill_q4.metal` | `flash_attn_prefill_q4` 134 | Prefill attention straight off Q4_K weights (legacy research route). |
| `ferrite/flash_attn_v2.metal` | `flash_attn_v2` 59 | General chunked flash attention over the F16 KV cache (live prefill kernel). |
| `ferrite/fused_residual_rms_norm_llamacpp.metal` | `fused_residual_rms_norm_llamacpp` 31 | llama accumulation-order fused residual+norm (single eps). |
| `ferrite/matmul.metal` | (no `kernel void` bodies of its own; function constants 10-12 + `block_q4_K_llama` layout doc 66+ and shared GEMM scaffolding; Q4_K 144-byte layout documented at lines 42-63) | Concat prefix defining Q4_K llama-style layouts and thin-GEMM constants consumed by later files. |
| `ferrite/matvec_multicol.metal` | macro `muser_matvec_multicol_##dtype##_c##nc` 400 (q4k/q5k/q6k × c4/c2/c1) | Multi-column matvec sharing one weight load across verify columns. **Compiled lazily, not in the main concat** (`MultiColPipelines::new`, encode/multicol.rs:93-103); default-off via `MUSER_MULTI_COL_VERIFY` (multicol.rs:12-14). |
| `ferrite/rms_norm_llamacpp.metal` | `rms_norm_llamacpp_f32` 49 | llama-exact 32-lane RMSNorm reduction. |
| `ferrite/rms_norm_per_head.metal` | `rms_norm_per_head` 15; `rms_norm_per_head_qkv_fused` 59 | Per-head QK-norm kernels. |
| `ferrite/rmsnorm_batch_tail.metal` | `rms_norm_batch` 1; `rms_norm_batch_inplace` 42; `fused_residual_rms_norm_batch` 72; `fused_rms_norm_residual_add_batch` 108; **`muser_fused_norm_residual_rms_norm_32sg` 147**; `fused_inplace_norm_residual_add_batch` 211; `muser_fused_norm_residual_rms_norm_batch_dual_eps` 250 | The sandwich-norm stack; 32sg is the live dual-eps fused tail. |
| `ferrite/rope.metal` | `rope_inplace` 20; `rope_batch` 65; `fused_bias_rope_batch` 130; `fused_bias_q_rope_batch` 202; `fused_bias_rope_store_kv_batch` 266; `rope_store_kv_batch_cached` 350; `fused_bias_q_rope_store_kv_batch` 429; `rope_batch_cached` 566; **`rope_norm_batch_cached` 624** | RoPE family; norm_batch_cached is the live SWA-layer RoPE. |
| `ferrite/sigmoid_gate.metal` | `sigmoid_gate_inplace` 7 | Muse sigmoid attention gate. |
| `crates/muser-bench/shaders/m16_candidates.metal` | `m16_q6k_r2` 178; `m16_q5k_r2` 235; `m16_q4k_t128` 313; `m16_q4k_t128_nobar` 434; `m16_dbg_mac` 559; `m16_dbg_stage` 617 | Bench-only M16 kernel candidates (`muser-m16-bench`). |

---

## 5. Quantization on live paths

Three weight lanes:

1. **kquant reference/serving lane (`q4_k_xl`)** — declared by GGUF
   `muser.weight_precision` (loader.rs:72-91; absent or `q4_k_xl` ⇒ kquant).
   CPU dequant/dot kernels for Q4_K, Q5_K, Q6_K, Q8_0, Q4_0, F16, F32 in
   `quant/` (`quant.rs` module doc lib.rs:90-97; dispatch `quant/dispatch.rs`,
   blocks `quant/blocks.rs`, Q6_K `quant/k_block/q6.rs`; `dot_row` dispatch
   weights.rs:123-126). The pinned artifact is 16,756,681,056 bytes
   (lib.rs:14; asserted chat_template.rs:246). Per-tensor dtypes are whatever
   the GGUF carries; FFN gate/up are Q4_K on the release artifact (decode.rs:5820-5821).
2. **Native NVFP4 lane (`nvfp4`)** — GGUF dtype `NVFP4_E2M1` with
   `muser.weight_precision=nvfp4`, fail-closed pairing (loader.rs:73-90).
   Format: E2M1 packed two per byte, one E4M3FN scale per 16 values, one f32
   `scale2` per tensor; order pinned `(e2m1 * e4m3fn) * scale2`
   (quant/nvfp4.rs:1-6). Scale plumbing: `nvfp4_scale2`,
   `nvfp4_input_scale_inv`, `nvfp4_scale_view` on projections (decode.rs:5960-5976,
   6064-6079). GPU: nvfp4.metal kernels (§4). Producer mode Exact vs Native is
   a receiver config enum `Nvfp4ProducerMode` (cluster config.rs:10-15).
   **`MUSER_NVFP4_EXACT` does not exist in Rust** — it is a *producer-side
   Python* env flag (scripts/gx10/vllm/benchmark_native_prefill.py:99-101,
   muser_native_prefilld.py:446).
3. **DFlash draft model** — a **five-layer** assistant ("Five-layer DFlash
   assistant", dflash.rs:1-5), loaded from a development SafeTensors export or
   the official llama.cpp-compatible **k-quant GGUF sidecar** (one validated
   loader, dflash.rs:8-10; `DFlashWeights::load` dflash/weights.rs:35,
   GGUF shell weights.rs:68-76). It reads hidden states from pinned target
   layers (`DFlashConfig.target_layer_ids`, dflash/config.rs:66) and has its
   own 64-row-sink + sliding-window context cache
   (`DFLASH_CONTEXT_SINK_SIZE = 64` dflash/config.rs:61; window from
   `dflash.attention.sliding_window` — wrong window collapsed acceptance
   72.5%→2.2%, config.rs:87-91). CPU oracle: dflash/forward.rs; Metal
   projection backends via `metal/dflash.rs` + `dflash/attention.rs`.

---

## 6. KV cache implementation

- `MetalKvPlane` — decode.rs:182-190: one `GpuHalfBuffer` key + value pair,
  `capacity`, `len`, `origin_logical`, `origin_physical`, `head_major`.
  **Element type is f16 on all Metal lanes** (`GpuHalfBuffer`, decode.rs:236-237);
  F32 planes exist only in the kvpack interchange (`PlaneEncoding::{F16Le,F32Le}`
  cache.rs:11-16).
- Allocation per layer kind — `from_shared` decode.rs:1344-1358: SWA layers
  get `capacity = min(max_context, sliding_window=2048).max(32)` **token-major**
  (`head_major=false`); NoPE full layers get `max_context.max(32)`
  **head-major** (`head_major=true`, decode.rs:1356). So: a 2048-row **ring**
  for the 39 SWA layers, a **growing contiguous plane** for the 13 NoPE layers.
  Zero-filled so wrapped rows never leak uninitialized storage (decode.rs:1350-1352).
- Ring mechanics — `append` decode.rs:265-284 (single) and `append_batch`
  decode.rs:286-314 (chunk): fail-closed `CacheDiscontinuity` when position ≠
  `origin_logical + len`; physical placement is never derived from absolute
  token position (prefill.rs:15-17).
- Store kernels: `muser_kv_store_f16` (muse_reference.metal:979, token-major),
  `muser_kv_store_batch_f16` (:1203, both layouts); NoPE "relocate = memcpy"
  is the kvpack free lunch (lib.rs:9-10).
- Snapshots/restore: `MetalKvSnapshot` decode.rs:193; restore preserves ring
  rotation so float accumulation order replays bitwise (decode.rs:376-383).
  Engine interchange for kvpack: `SessionCacheSnapshot` cache.rs:33-40 — 39
  SWA planes carry the logical tail, 13 NoPE planes `[0, position)`.
- Speculative transactional checkpoint: `MetalSpeculativeCheckpoint`
  decode.rs:213-226 — NoPE planes rewind metadata only; SWA planes retain the
  ≤16 rows a block may overwrite; commit/rollback decode.rs:1386-1496.
- **Context shift**: there is no engine-level "shift" op; the server owns the
  policy (`ContextPolicy::{Shift,Error}` state.rs:216-219) and rebuilds
  context via the full-capacity `staging` session + atomic swap
  (state.rs:240-243); chat-unit shift logic in openai.rs:5374
  (`shift_chat_units`).

---

## 7. Speculative decoding (DFlash)

- Draft/verify engine: `dflash/spec.rs` (`DFlashAssistant`). Draft:
  `draft_greedy` spec.rs:730 / `draft_greedy_with_session_projection` :778 /
  `draft_sampled` :910. Greedy chains: `generate_greedy` :1000,
  `prepare_greedy` :1022, `generate_prepared_greedy_streaming` :1180.
- **Acceptance happens on the CPU against full target distributions**:
  `verify_full_speculative_mt_ordered` sampling.rs:1033 (also
  `verify_full_speculative` :827, `_mt` :1008) produces a
  `SpeculativeDecision`; `Session::verify_batch` (api.rs:913) +
  `greedy_verification_decision` (api.rs:1870) drive the engine-level
  batch-verify (`decode.rs:3298 begin_dflash_verify_suffix` /
  `decode.rs:3635 finish_dflash_verify_suffix` — the Metal mirror-SD route
  that splits the target graph at a capture layer and overlaps draft work).
- Telemetry & gating: `DFlashSpecStats` spec.rs:12-101 (acceptance rate,
  window gate `should_disable_speculation` :184, re-qualification backoff);
  per-request adaptive fallback `fallback_round` :220.
- Target distribution verification is exact: the qualifier compares 256
  greedy tokens plus every full target-logit row (muser-bench remote.rs:8-10;
  DFLASH_ACCEPTANCE_MINIMUM 0.95 remote.rs:37).
- ANE variant (Core ML, feature `ane-coreml`): `dflash_ane.rs` +
  `target_ane.rs`; composite overlap profiled by
  `muser-composite-dflash-qualify`.

---

## 8. kvpack

### 8.1 Adapter crate `crates/muser-kvpack`

- Purpose per lib.rs:1-40: kvpack is Muser's **one** shared external dep,
  pinned to the audited in-tree snapshot; this crate re-exports the pinned
  API and adds `layout` (Muse K1/K3 layout glue), `session` (save/restore +
  relocation-as-memcpy), `economics` (dashboard accounting, no Ferrite source).
- Muse layout accounting: `layout.rs` — `MuseIdentity` :21 (digest :37),
  `descriptor` :69 builds the qualified layout table; `validate_geometry`
  :138 cross-checks cached vs live geometry. Upstream Muse keys: K1 (NoPE
  theta=0 fail-closed), K3 (2-class 39-SWA/13-NoPE), K4 scalar-math identity,
  K5 session artifact (lib.rs:22-29).
- Durable sessions: `session.rs` — `save` :144, `save_snapshot` :157,
  `find_deepest` :222, `restore_deepest` :248.
- Reuse order (muser-original): `reuse.rs` — current session → resident →
  durable → remote (reuse.rs:1-7). Resident compressed token radix:
  `resident.rs` (content-interned Muse plane chunks scoped to the identity
  digest). Remote authenticated prefix import: `remote.rs` (streams kvpack
  manifest + chunks into a private local import before any engine install).

### 8.2 Vendored tree `third_party/kvpack`

- Own workspace with three crates (provenance `workspace_members`):
  **kvpack-core** (format: `canonical.rs` canonical JSON, `chunk.rs`
  content-addressed chunks, `manifest.rs`, `pack.rs` append-only packs,
  `identity.rs` keyed cache identities, `keys.rs` longest-committed-prefix
  lookup, `quant.rs`, `rotation.rs`, `half.rs`, `validator.rs`),
  **kvpack** (engine-facing store: `store/`, `restore/`, `writer/`,
  `export/`, `gguf_layout/`, `artifact.rs`, `intent.rs`, `sink.rs`),
  **kvpack-handoff** (the sealed V2 wire format: `handoff_v2.rs` with
  `SegmentDescriptorV2` :59, `BeginManifestV2` :73 carrying
  `HmacIdentityV2`, `SealManifestV2` :243 whose `hmac_sha256` tags the
  canonical-JSON core :285-286, streaming verify :447-449; `manifest.rs`
  (`ExactIdentityV1` :58, `LayoutClassV2` :92, `LayerHeaderV1` :349);
  `receiver/`, `coordinator.rs`, `mac.rs`).
- README highlights (third_party/kvpack/README.md): replay = restoring exact
  engine bytes; fail-closed compatibility identities (model/revision/quant/
  tokenizer/template/layout/ABI/dtype); crash-safe append-only publication
  with terminal commit seal; layered integrity (record hashes, Merkle commit,
  whole-pack digest); content-addressed prefix lookup.
- `provenance.json`: schema `muser.vendored-source.v1`; upstream
  `https://github.com/High-Performance-AI-Lab/kvpack` at commit
  `70c34c7d790dbfc9c1271727dd34ea0e863404d2`, tag
  `kvpack-v0.1.0-alpha.2-rc1`, tree `7d56417c...`; per-file SHA-256 map; one
  recorded patch (canonical-json sorted-keys, independent of serde
  preserve_order). Audit script: `scripts/audit_vendored_kvpack.py`.
- Format note: **compression/quant live in kvpack-core (`quant.rs`,
  lossless encodings); auth is HMAC-SHA256 over canonical manifests (seal)
  plus the mTLS channel in muser-cluster** — there is no payload encryption
  inside a pack.

---

## 9. Handoff V2 / cluster lane

- `muser-cluster/src/lib.rs:1-18`: 1× M3 Ultra decode + 1× NVIDIA GB10 GX10
  prefill; one producer at a time (one control endpoint, one HMAC key id,
  per-key-id replay ledger); mTLS-TCP with a release floor of 3.0 Gbps median
  installed-payload throughput; ships the 13 NoPE tiles *during* prefill.
- **mTLS**: `security.rs` — TLS 1.3-only, exact ALPN `muser-kvpack-v2`
  (security.rs:16), leaf-pin sets, `accept_mtls`/`connect_mtls_with_alpn`,
  `load_mac_key` :55, `ReplayLedger` :356 (`load` :363, `reserve` :412,
  `record` :431). **Durable reservation dance**: write temp + `sync_all` +
  rename + parent-dir `sync_all` (security.rs:483-485).
- **Receiver-side ledger-volume gate**: `check_ledger_volume` probes the
  reserve-pattern tail latency and refuses a slow volume before any handoff
  (receiver.rs:108-150, `probe_ledger_reserve` :150) — the 2026-08-18
  bimodal-~1s lesson operationalized.
- **Wire framing**: `transport.rs` — frames `Begin/Segment/...` over the TLS
  stream, magic `KVPKV2\0\0`, 20-byte preamble (transport.rs:11-12). HMAC
  seal + per-segment verify come from kvpack-handoff (§8.2); the Mac sink
  (`muse_sink.rs`) unpacks each authenticated tile into a **detached Metal
  generation** and swaps live decode state only after the seal.
- **Transfer schedule**: `schedule.rs` — 13 NoPE layers
  `[3,7,...,51]` (schedule.rs:20), one HMAC/TLS frame per 512-token NoPE tile
  (~6.5 MiB) + three pipe-safe SWA groups, overlapping CUDA ubatches; gate
  `hidden_pct >= 0.95` (schedule.rs:1-11).
- **Mac-side receiver binding**: `ReceiverConfigV2` (cluster config.rs:22-50):
  listen addr, cert chain/private key/peer CA/leaf SHA-256 pins, hmac key
  file + id + minimum epoch, **`replay_ledger` path**, timeouts, optional
  `producer_control` address, `producer_mode` (exact|native),
  `ExactIdentityV1`, `target_cache_identity_sha256`, optional
  `dflash_identity_sha256` + `dflash_context_geometry` (stamped at
  enrollment). Loaded from the node config at
  `~/.muser/nodes/<name>/cluster.json` (AGENTS.md; config surface only —
  secrets are referenced by path, never read).
- **Control plane**: `control.rs` — small canonical-JSON request channel to
  the resident GX10 `muser-prefilld`, ALPN `muser-prefill-control-v1`
  (control.rs:13); cache bytes never ride this channel.
- **Phase evidence**: `phase.rs` — per-segment `read_ns`/verify+install/seal/
  commit nanosecond splits (N-series), structural no-phase-exceeds-span check.
- **Remote speculative verifiers**: `verifier.rs` (authenticated round log),
  `verifier_v2.rs` (durable carried-frontier protocol: commit fsynced before
  renderer activation), `verifier_gateway*.rs` (durably reserve → execute →
  sign → stage invisible render → commit/activate → reply).
- **Producer side** (Python, on the node): `scripts/gx10/vllm/`
  (`muser_native_prefilld.py` — requires `payload_pacing_bps >= 4 Gbps` in
  receipts, :569-570; resident vLLM NVFP4 producer in docker; fail-closed
  exit 75). **Wire pacing is a producer-side pin**; Mac code only measures.
- **Qualification**: `muser-remote-qualify` (muser-bench remote.rs) uses the
  exact serving `RemoteReceiver`, cold-recompute vs remote install, 256
  greedy tokens + all logit rows, `LINK_GBPS_MINIMUM = 3.0` and
  `DFLASH_ACCEPTANCE_MINIMUM = 0.95` (remote.rs:36-37). Enrollment recipe:
  native identity at `scripts/gx10/vllm/native_onboarding_identity_v1.json`.

---

## 10. Scheduler & slots

- **One-scheduler-one-accelerator**: `AcceleratorScheduler`
  (decode.rs:1023-1026) — a single Mutex+Condvar owner of the shared Metal
  queue; `active` flag, `decode_waiting: BTreeSet<usize>`,
  `last_decode` for cyclic rotation.
- **Decode-over-prefill priority**: `acquire` (decode.rs:1040-1074) — decode
  work runs only when selected as next decode; prefill acquires only when
  *no* decode is waiting (`selected_decode.is_none()`, decode.rs:1058).
  Fairness: ascending cyclic order via `next_decode_sequence`
  (decode.rs:1142-1152). Chunk shrinking so decode doesn't queue behind a
  512-row batch: decode.rs:2098-2107.
- **1..=4 slots**: `forward_decode_group` rejects anything outside 1..=4
  (decode.rs:4874); server enforces `--parallel` in 1..=4
  (state.rs:1054-1056) and OpenAI `n` in 1..=4 (openai.rs:3480-3481).
  Server-side `SlotPool` (state.rs:479-483) is bounded admission with an
  unhealthy latch; the full-capacity `staging` session is deliberately
  outside the pool (state.rs:240-243).
- **Rendezvous batching**: `DecodeBatcher` (state.rs:231-233) — request
  threads keep slot ownership while one elected runner packs up to four
  ready Metal rows; coalesce window `DECODE_COALESCE = 250 µs`
  (state.rs:254); admission bound `MAX_QUEUED_REQUESTS = 64` (state.rs:253),
  Axum `ConcurrencyLimitLayer(256)` (axum_httpd.rs:544), transfer payload
  lane limited to 4 (axum_httpd.rs:552-556).

---

## 11. Server surface

- Routes (axum_httpd.rs:489-541): `/` + `/dashboard`, `/snapshot`,
  `/metrics` (Prometheus, metrics.rs), `/telemetry`, `/health`+`/v1/health`+
  `/healthz`, `/models`+`/v1/models`, `/props`, `/slots` (+`/{id}`),
  `/tokenize`, `/detokenize`, `/apply-template`, `/embedding(s)`+
  `/v1/embeddings`, `/completion(s)`+`/v1/completions`, `/api/generate`
  (Ollama), `/v1/chat/completions` (+`/control`), `/v1/streams/lookup`
  (resumable SSE, resumable_stream.rs), `/v1/dashboard/login`,
  `/v1/ws-tickets`, `/v1/sessions` CRUD + `/save` `/restore` `/migrate`,
  `/stream` (WebSocket), `/v1/nodes` (+`/{name}/progress`),
  `/__muser/benchmark/shutdown`, and a separate unlimited-body
  `/__muser/v1/session-transfers/{id}/payload` router.
- Chat orchestration: `openai.rs` (request validation, n≤4, dry-run mode,
  context-shift lineage checks openai.rs:6717-6817).
- Sampler: engine `sampling.rs` — scalar temperature/top-k/top-p
  (`distribution_ordered` :399), deterministic MT19937 (`Mt19937` :58 with
  snapshot/restore :88/:95 — same RNG stream across local/remote lanes),
  shared-Gumbel :700.
- Grammar: `grammar.rs:1-9` — pinned llama-style **GBNF**, Earley recognizer
  over code points, partial-UTF-8 acceptance, EOS only at completed root.
- Detokenizer: engine `tokenizer/streaming.rs` (`StreamingDetokenizer`,
  api.rs:348); template engine `chat_template.rs` (fancy-regex over the GGUF
  template).
- Sessions: durable store `session_store.rs` (revision CAS, encrypted
  bundles, template_sha256 binding :26).

---

## 12. Benchmarks & tooling

**muser-bench binaries** (Cargo.toml:8-140): `muser-bench` (main), `muser-forward-evidence`,
`muser-capture-evidence`, `muser-token-fixture`, `muser-greedy-evidence`,
`muser-kvpack-qualify`, `muser-dflash-qualify`, `muser-dflash-inspect`,
`muser-dflash-extract`, `muser-vision-qualify`, `muser-vision-inspect`,
`muser-ane-qualify`, `muser-target-ane-qualify` (ane-coreml),
`muser-remote-qualify` (metal), `muser-metal-phase-diagnostic` (metal),
`muser-m16-bench` (metal), `muser-dflash-m0` (metal),
`muser-composite-dflash-qualify` (metal).

**scripts/ highlights**: `accelerator_safe.py` (mandatory wrapper holding
`/tmp/ferrite.gpu.lock`, dry-run default); `scripts/gx10/` diagnostics —
`tcp_probe.py` (raw ceiling; ~9.4 Gbps healthy point-to-point reference),
`durable_fsync_probe.py` (ledger-volume tail check, exit 1 past
`--max-tail-ms`), `handoff_report.py` (per-rep phase table from retained
receipts), `restart_resident_producer.py` / `supervise_resident_producer.py`
(fail-closed producer rituals); tests `scripts/tests/test_gx10_diagnostics.py`.
Campaign/audit harnesses: `campaign.py`, `correctness_campaign.py`,
`atomic_seal_campaign.py`, `audit_vendored_kvpack.py`,
`compile_llama_metallib.sh` (builds the pinned ggml metallib for
`MUSER_GGML_METALLIB`), `run_kvpack_ladder_session.py`.

**Where results go**: append-only evidence volume
`muser-receipt://` (AGENTS.md); repo `results/` holds stage
artifacts (e.g. `results/stage2-accepted-matvec/...`). Operational state
(replay ledger, sockets, locks) must stay on the internal disk.

---

## 13. GGUF / model identity

- Startup: `main.rs:129-160` reconciles configured vs verified model SHA-256
  (`ServerState::new_with_verified_sha256` state.rs:962; mismatch refuses,
  state.rs:1168-1175). Download path verifies with SHA-256 (main.rs:22).
- Pinned release artifact facts (test `release_gguf`,
  muser-server/src/chat_template.rs:237-261): model SHA-256
  `7e9b74b7c8875e9e265695df9613bf6290f2392e479ce740495a129019c488d8`,
  byte size **16,756,681,056**, chat template exactly **7,167 bytes** with
  SHA-256 `114f55ebdc1804c1af371197b9fdf2d6bb925966c9dfe46b73782a71bc07965e`,
  tokenizer-metadata SHA-256
  `61e73226502f8f54455555990c0000852247bbec32b107730ec544bc0b738055`.
  The same 7,167-byte fact appears on the producer side in
  `scripts/gx10/vllm/native_onboarding_identity_v1.json:15`.
- Template/tokenizer identity hashing at load: `loader.rs:48-63`
  (`chat_template_sha256`, `tokenizer_metadata_sha256`); identity is bound
  into session envelopes (axum_httpd.rs:1335) and durable bundles
  (session_store.rs:26). Fail-closed QK-norm probe at load: loader.rs:28-37,
  probe_qk_norms loader.rs:98-138 (converter-synthesized constant broadcasts;
  a learned norm aborts).
- Remote lane identity: `target_cache_identity_sha256` +
  `ExactIdentityV1` in cluster config (§9); DFlash draft identity
  `dflash_identity_sha256` bound into bundles (state.rs:225-227).

---

## 14. Env flags (`MUSER_*`)

84 unique names in crates (`grep -rhoE 'MUSER_[A-Z0-9_]+' crates --include='*.rs'`).
The ones that change live-path behavior (the rest are test/bench/fixtures):

| Flag | Meaning | file:line (representative) |
|---|---|---|
| `MUSER_GGML_METALLIB` | Path to pinned llama.cpp metallib; enables ggml matvec/norm/rope/FA PSOs | metal/context.rs:122; pipelines encode.rs:278-280 |
| `MUSER_GGML_METALLIB_RECEIPT` | Provenance receipt for the metallib | encode.rs:83 |
| `MUSER_CROSS_VENDOR_QK` | Strict-f32 cross-vendor routes (CUDA parity) for QK norm + attention | decode.rs:5645; norm.rs:536; context.rs:111-121 |
| `MUSER_CROSS_VENDOR_ROPE_CACHE` | Retained RoPE table file (regular-file check, exact byte length) | decode.rs:1217-1248 |
| `MUSER_CROSS_VENDOR_ROPE_BYPASS` | Skip RoPE in cross-vendor comparisons | (muse_reference routes) |
| `MUSER_FERRITE_FFN_GATE_UP` | Use fused Ferrite `ffn_q4k_gate_up_silu_4r2s` FFN | decode.rs:5819-5836, 1334 |
| `MUSER_NO_FUSED_PREFILL_DUAL_NORM` | Diagnostic: split the fused dual-eps tails | decode.rs:1331 |
| `MUSER_SERIAL_PREFILL_DISPATCH` | Serial (non-concurrent) prefill encoder | decode.rs:1332 |
| `MUSER_NO_LLAMA_FA_PREFILL` | Force local FA2 over llama pinned prefill kernel | decode.rs:56 (route helpers) |
| `MUSER_MULTI_COL_VERIFY` | `1` = bitwise-exact dtypes only; `all` = add Q6_K multi-column verify | encode/multicol.rs:70-82 |
| `MUSER_NO_M16_N32` | Disable the M16 NVFP4 batch route | decode.rs:5958 |
| `MUSER_METAL_PHASE_PROFILE` | Per-phase token graph timing report | decode.rs:5440 |
| `MUSER_METAL_BATCH_PHASE_PROFILE` / `MUSER_STREAM_DECODE_PROFILE` | Batch/stream decode profiling | decode.rs:33; decode.rs:29 |
| `MUSER_DFLASH_*` (`GATE`, `WINDOW`, `SINK`, `VERIFY_LEN`, `CYCLE_TRACE`, `MIRROR_OVERLAP`, `SAMPLED_REPLAY`, `CAPTURE_FC_PIPELINE`, …) | Spec-decoding knobs/diagnostics | dflash/spec.rs:95,153; decode.rs (verify routes) |
| `MUSER_ACCELERATOR_LEASE` | Accelerator lease diagnostic | decode.rs (scheduler) |
| `MUSER_NVFP4_QKV_*` / `MUSER_GX10_*` / `MUSER_LLAMA_*` | Fixture plumbing for NVFP4/llama parity comparators | muser-bench fixtures |
| `MUSER_MODEL`, `MUSER_MODEL_SHA256` | Release-real-model test identity | chat_template.rs:241-246 |
| `MUSER_HOME`, `MUSER_HOST`, `MUSER_PORT` | Server/node runtime locations | muser-server |
| `MUSER_REMOTE_*` (`QUALIFY`, `QUALIFY_SERIAL`, `CACHE_PROBE`, `CACHE_DIFF`, `FIRST_DIVERGENCE`, `TOKEN_FIXTURE`) | Remote qualify/probe behavior | muser-bench/src/remote.rs |
| `MUSER_CACHE_ABI` | KV-cache ABI variant selection for tests | decode.rs:84 area |

Not in Rust despite the outline: `MUSER_NVFP4_EXACT` (Python producer only, §5).
`MUSER_MC_NSG` / `MUSER_MC_NR0` are shader-side constants documented at
multicol.rs:41-44, not runtime flags.

---

## 15. Glossary seeds (30 terms)

1. **MuseConfig** — fully-resolved hyperparameters parsed fail-closed from GGUF; `config.rs:106`.
2. **MuseLayerKind** (`SlidingRope`/`FullNoPe`) — per-layer attention kind; SWA layers carry RoPE, full layers are NoPE; `config.rs:55-60`.
3. **layer_kind / sliding_window_pattern** — `il % 4 == 3` ⇒ full layer; `config.rs:84-93`, resolver `config.rs:347`.
4. **QkNormProbe** — load-time proof that q/k norms are converter-synthesized constant broadcasts (qk_scale_factor ≈ 3.87); `config.rs:390`, loader.rs:98.
5. **post_norm_eps = 1e-8** — llama.cpp hard-coded eps for the two sandwich post-norms (vs 1e-5 elsewhere); `config.rs:28`, muse-glimmer.cpp:67.
6. **logit_scale = 1/√26 ≈ 0.196116** — `output_multiplier` applied before softcap; `config.rs:190`, metal.rs:68.
7. **final_logit_softcap = 20** — tanh softcap on scaled logits; `config.rs:195`, reference.rs:565.
8. **MetalKvPlane** — per-layer f16 K/V buffers + explicit ring metadata; `decode.rs:182`.
9. **origin_logical / origin_physical** — logical token origin vs physical ring slot; never derived from absolute position; `decode.rs:187-188`.
10. **head_major** — NoPE plane layout `[kv_head][capacity][head_dim]` (vs token-major SWA); `decode.rs:189`, snapshot walk decode.rs:329-334.
11. **append/append_batch** — fail-closed ring reservation; `decode.rs:265`, `:286`.
12. **AcceleratorScheduler / AcceleratorPermit** — the one queue owner; decode-first, cyclic fairness; `decode.rs:1023`, `:1028`.
13. **forward_decode_group** — packed 1..=4-row decode graph; `decode.rs:4869`.
14. **encode_token** — the single-token 52-layer Metal graph (the book's "one decode token" walkthrough); `decode.rs:5515`.
15. **MetalShared** — one executor: context, kernels, mmap'd weights, layers, scheduler, workspaces; `decode.rs:958`.
16. **PREFILL_BATCH_TOKENS=512 / MAX_TEACHER_FORCED_TOKENS=64** — chunk sizes; decode-latency-aware shrinking; `decode.rs:53-54`, 2098-2107.
17. **SWA staging shadow** — detached logical tail staged before ring wrap commits; `encode_stage_swa_prefill_f16` attn.rs:103.
18. **llama_vec_rows / llama_swa** — route predicates for the pinned llama vec FA kernel (32-row rounding constraints); decode.rs:5646-5657.
19. **ggml_library / MUSER_GGML_METALLIB** — pinned llama.cpp PSOs for numerical parity; context.rs:122-131.
20. **cross_vendor_library** — strict-f32 recompile of muse_reference+nvfp4 matching CUDA scalar boundaries; context.rs:36-39, 111-121.
21. **q4_k_xl / nvfp4 (weight_precision)** — the two live GGUF weight lanes; loader.rs:72-91.
22. **NVFP4_E2M1 + E4M3FN scale + scale2** — native 4-bit format, `(e2m1*e4m3fn)*scale2`; quant/nvfp4.rs:1-6.
23. **DFlash** — five-layer speculative draft assistant (kquant GGUF sidecar or SafeTensors); dflash.rs:1-10.
24. **DFlashContextGeometry / DFLASH_CONTEXT_SINK_SIZE=64** — draft context ABI (sink+window), enrollment-stamped; dflash/config.rs:17, 61.
25. **verify_full_speculative_mt_ordered** — CPU acceptance against the exact target distribution; sampling.rs:1033.
26. **Mirror-SD / begin_dflash_verify_suffix** — split-graph speculative verification overlapping draft on Metal; decode.rs:3298.
27. **Handoff V2** — mTLS-TCP + HMAC-sealed manifest tile protocol (kvpack-handoff); transport.rs, handoff_v2.rs:73/243.
28. **ReplayLedger + durable reserve** — HMAC-epoch/generation admission persisted with fsync+rename+dir-fsync; security.rs:356, 483-485.
29. **NoPE-tiles-during-prefill schedule** — 13 position-free tiles streamed during CUDA prefill (relocate=memcpy); schedule.rs:1-11, lib.rs:9-10.
30. **SlotPool / DecodeBatcher / staging** — server admission (1..=4), decode rendezvous, and the out-of-pool rebuild generation; state.rs:479, 233, 243.

---

## Outline corrections (code vs book outline)

1. **"logit soft-cap tanh@20 with 1/sqrt(26)"** — correct, but 1/√26 is not a
   code constant: it is GGUF metadata `muse-glimmer.logit_scale` read at
   config.rs:190-192 (value 0.196116 confirmed in docs/release-provenance.md:823).
   The hard-coded `1.0/26.0f32.sqrt()` appears only in a fixed test (metal.rs:68).
2. **"RoPE interleaved-pair"** — correct for SWA layers (GPT-J pairing noted
   at lib.rs:105); full layers are **NoPE (no RoPE at all)**, not an
   alternative RoPE.
3. **`MUSER_NVFP4_EXACT=1`** — not a Rust flag; producer-side Python only.
4. **"flash attention prefill, chunking"** — prefill has three attention
   routes (llama pinned vec per-query, llama pinned non-vec masked, local
   FA2) plus an SWA staging shadow; see §3.
5. **prefill.rs as a module** — it is only a doc header; the driver lives in
   decode.rs (by design).
6. **"store kernels, context shift"** — context shift is server-level policy
   + staging rebuild, not a KV-cache op (§6).
7. The scheduler is per-MetalShared (engine-level) and separate from the
   server SlotPool/DecodeBatcher; "one-scheduler-one-accelerator" is exactly
   `AcceleratorScheduler` (§10).
