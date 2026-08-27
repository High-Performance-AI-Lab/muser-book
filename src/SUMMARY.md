# Table of Contents

> Path explained: **Muse Glimmer (52 layers, ~30B), pinned Muser tree, M3 Ultra
> / 96 GB decode + one GX10 (GB10) NVFP4 prefill node over authenticated
> Handoff V2.**
>
> All 40 chapters are `status: polished` at the pinned revision (every code
> fence byte-verified, every number receipt-tagged — see Appendix E, the pin).

[Introduction](README.md)

> A note on the decode path: serving decode routes one-token work through the
> one-row *batch* graph because the legacy fused single-token kernels breach
> public-logprob tolerance `[crates/muser-engine/src/decode.rs]`. The
> per-kernel chapters (11–21) walk the teacher-forced route, which runs the
> same op sequence and is the cleanest narration of the graph; where the
> serving route differs, the chapter says so.

---

## Part I — The Problem and the Metal Compute Model

- [**Ch 1. The problem: why inference is a memory problem**](chapters/01-why-inference-is-a-memory-problem.md)
  The bandwidth wall at 30B scale. One Muse Glimmer token, costed by hand on
  an M3 Ultra with 96 GB: parameters → bytes → per-token reads → FLOPs → GPU
  time → DRAM time. Why decode is ~99 % reading weights, and why that single
  fact dictates everything from quantization to disaggregation.

- [**Ch 2. The Metal compute model**](chapters/02-metal-compute-model.md)
  `MTLDevice`, `MTLCommandQueue`, `MTLCommandBuffer`, the compute encoder.
  Grids, threadgroups, threads, **SIMD groups** — the unit that actually
  matters on Apple Silicon. Why Metal is "record a tape, then press play."

- [**Ch 3. Unified memory and the buffer substrate**](chapters/03-unified-memory-and-buffers.md)
  96 GB of one physical memory for CPU and GPU. `StorageModeShared`,
  mmap'd GGUF, zero-copy weight views, tracking modes, and the buffer
  substrate Muser actually built.

- [**Ch 4. Pipeline state objects and the three kernel sources**](chapters/04-pso-and-three-kernel-sources.md)
  How `.metal` source becomes a runnable kernel — and why Muser deliberately
  runs kernels from **three sources**: its concatenated fast-math library, a
  strict-f32 cross-vendor library, and a **pinned llama.cpp metallib**, with
  fingerprinted selection so a silent fallback cannot hide.

## Part II — Quantization

- [**Ch 5. Quantization from scratch**](chapters/05-quantization-from-scratch.md)
  Why four bits per weight is not enough alone; blocks, scales, and the
  min+offset trick. Symmetric vs asymmetric quantization. A full worked
  example: pick amax, round, dequantize, measure the error — the template
  every later quant chapter reuses.

- [**Ch 6. The kquant family on the reference lane**](chapters/06-the-kquant-family.md)
  Q4_K, Q6_K, and friends as the reference lane uses them: block layouts,
  the dequant formulas, and the pinned ggml matvec kernels
  (`kernel_mul_mv_q*_K_f32`) that Muser dispatches for Q/K/V/gate.

- [**Ch 7. NVFP4: the native lane**](chapters/07-nvfp4-native-lane.md)
  FP4 (e2m1) with block scales: the format, the loader, `nvfp4.metal`, and
  why native NVFP4 decode is parity-within-noise with kquant — never claimed
  faster `[claims #11]` — while speculative NVFP4 stays fail-closed.

- [**Ch 8. The DFlash draft**](chapters/08-the-dflash-draft.md)
  Speculative decoding's draft side: the 5-layer kquant draft model, its
  SafeTensors/GGUF sidecar, the 64-row sink context ABI, and what a draft
  must guarantee for the target's verification to stay exact.

## Part III — The Model

- [**Ch 9. The Muse Glimmer architecture**](chapters/09-muse-glimmer-architecture.md)
  The 52-layer graph in the repeating `[sliding, sliding, sliding, full]`
  pattern: 39 SWA layers with a 2,048-token window and interleaved-pair RoPE,
  13 NoPE full-attention layers with no rotation, GQA 32Q:2KV at head_dim 128,
  the sigmoid attention gate, the dual-epsilon norm sandwich, and the
  `1/√26` + tanh(20) logit soft cap.

- [**Ch 10. The forward pass at a glance**](chapters/10-the-forward-pass-at-a-glance.md)
  The full decode loop as one diagram: the residual stream, the per-layer
  kernel chain with its named fusions, the DFlash draft/verify/accept loop,
  and the handoff-receive path that plants precomputed KV. Where every buffer
  lives. Why prefill and decode are different code paths.

## Part IV — The Decode Path, Kernel by Kernel

- [**Ch 11. Token embedding lookup**](chapters/11-token-embedding-lookup.md)
  The `[hidden_dim]` residual stream is born here; `muser_embedding_q4k`.

- [**Ch 12. RMSNorm and the dual-epsilon sandwich**](chapters/12-rmsnorm-and-the-dual-eps-sandwich.md)
  Why root-mean-square and not standard deviation; the per-head QK-norm
  cousin; the 1e-5 GGUF epsilon everywhere except the two 1e-8 post-norms —
  a llama.cpp graph constant; the fused dual-eps residual tails.

- [**Ch 13. The QKV + gate matvec family**](chapters/13-the-qkv-gate-matvec-family.md)
  **The hero chapter.** Four concurrent matvecs (Q, K, V, and the attention
  gate) through pinned ggml kernels; the matvec math; the access pattern
  where the bandwidth story lives; why serving routes one token through the
  batch graph.

- [**Ch 14. QK-norm and RoPE — rotating only the layers that rotate**](chapters/14-qk-norm-and-rope.md)
  What "position" means per layer class: interleaved-pair RoPE on the 39 SWA
  layers, **no rotation at all** on the 13 NoPE layers, and why that
  distinction makes NoPE KV tiles relocatable bytes.

- [**Ch 15. KV store and the ring**](chapters/15-kv-store-and-the-ring.md)
  The f16 KV planes; token-major storage for the SWA ring, head-major for the
  growing NoPE cache; the store kernel; ring rotation; why restore must
  preserve rotation for bitwise replay.

- [**Ch 16. Attention: the decode kernel ladder**](chapters/16-attention-decode-kernels.md)
  Q·Kᵀ softmax V from zero, the online-softmax trick, GQA 32:2 fan-in, SWA
  masking against the ring, and the route ladder (llama vec / splitk /
  ferrite interleaved) with its alignment predicates.

- [**Ch 17. The sigmoid gate and the output projection**](chapters/17-sigmoid-gate-and-oproj.md)
  Muse Glimmer's unusual attention output: a learned sigmoid gate on the
  attention result before `o_proj`; the residual add.

- [**Ch 18. The SwiGLU feed-forward block**](chapters/18-swiglu-ffn.md)
  What an FFN does; SiLU defined; the fused `ffn_q4k_gate_up_silu` kernel
  and its opt-in flag; the normed-quant tail variant.

- [**Ch 19. The down projection + residual**](chapters/19-downproj-and-residual.md)
  Closing the FFN and the layer; the second fused tail producing the next
  layer's normed input.

- [**Ch 20. Final norm, LM head, and the soft cap**](chapters/20-final-norm-lm-head-softcap.md)
  The last RMSNorm; the vocab projection; `muser_scale_softcap_inplace` —
  `1/√26` from GGUF metadata, `tanh` at 20 — and why a soft cap changes how
  you must compare logits across engines.

- [**Ch 21. Sampling, argmax, and grammar**](chapters/21-sampling-argmax-and-grammar.md)
  Greedy vs sampled decode; why the read-back is 4 bytes; exact speculative
  acceptance on the CPU; grammar-constrained sampling state.

## Part V — The KV Cache and kvpack

- [**Ch 22. The price of context**](chapters/22-the-price-of-context.md)
  KV bytes per token derived by hand for both layer classes; footprint at
  2k/32k/131k; the 131,072-position model limit; why KV — not weights —
  decides how many slots a 96 GB Mac can serve.

- [**Ch 23. The SWA ring and the growing cache**](chapters/23-the-swa-ring-and-the-growing-cache.md)
  Two regimes in one engine: the 2,048-token ring of the 39 sliding layers
  vs the growing cache of the 13 full layers; context shift as server policy
  with staging generations and atomic publication.

- [**Ch 24. kvpack: the format**](chapters/24-kvpack-the-format.md)
  The vendored kvpack tree and its provenance; the container: headers,
  layout, the Muse layout/accounting adapter in `muser-kvpack`; what the
  HMAC-sealed manifest binds; why the receiver refuses slow volumes before
  any transfer.

- [**Ch 25. Warm reuse: the cache as an asset**](chapters/25-warm-reuse.md)
  The kvpack ladder's stage-5 results: bit-identical warm hits at 65,536 and
  130,815 tokens, first token in 0.6132 s / 1.0566 s vs 68.6 s / 147.8 s
  cold, no producer drive; the shallow 64.631 ms hit at its own scope;
  admission identity discipline.

- [**Ch 26. Delta handoff and migration**](chapters/26-delta-handoff-and-migration.md)
  Moving only what's new: the 32,768-of-65,536 cell at 54.2851 % of full
  bytes with bit-exact output; two-phase decode-node migration and
  storage-tier moves; destination-commits-before-source-deletes.

## Part VI — The Disaggregated Lane

- [**Ch 27. Why disaggregate prefill and decode**](chapters/27-why-disaggregate.md)
  The economics: prefill is compute-bound and parallel, decode is
  bandwidth-bound and serial; the TTFT cliff at depth (570 s local vs 137.4 s
  remote at 130,815 tokens, 4.149×); why one Mac + one producer is the v0.1
  truth and scale-out is roadmap, not fiction.

- [**Ch 28. The GX10 node and vLLM NVFP4 prefill**](chapters/28-the-gx10-and-vllm-nvfp4-prefill.md)
  GB10 hardware in brief; the resident vLLM producer container; NVFP4 on
  tensor cores; the producer's fail-closed exit (status 75) and the restart
  ritual; why a bare `docker restart` is not enough.

- [**Ch 29. CUDA vs Metal: the differences that mattered**](chapters/29-cuda-versus-metal.md)
  Warps vs SIMD groups; tensor-core FP4 pipelines vs SIMD-group dot; UMA vs
  discrete memory (and why the wire becomes the star); stream/table
  semantics vs one queue; and why Muser pins llama.cpp's metallib kernels
  instead of re-expressing them — each difference tied to a decision in this
  codebase, not a spec-sheet tour.

- [**Ch 30. Handoff V2: the authenticated transport**](chapters/30-handoff-v2-transport.md)
  mTLS with node-local key generation; the HMAC-sealed manifest; SSH-verified
  shared-secret transfer; enrollment and the one-button wizard; the
  three-handoff qualification recipe and its exactness policies.

- [**Ch 31. The wire discipline**](chapters/31-the-wire-discipline.md)
  The pacing ladder from 3.9 of 9.4 Gbps (our own pin) to the 8 Gbps
  ceiling; EEE's retransmission blackouts and the EEE-off link invariant;
  the durability lesson: operational state on the internal disk, evidence on
  the append-only volume; the replay ledger's fsync dance.

- [**Ch 32. Precision across the handoff**](chapters/32-precision-across-the-handoff.md)
  **The trust chapter.** What "the producer's KV is good enough" must mean:
  exact-token policies vs declared bounded-logit policies; the integer-dot
  verification anchor (`MUSER_NVFP4_EXACT=1`); logit drift gates and the
  docs top-token sensitivity at 65k; separate cache identities per recipe;
  the kquant lane as the explicit reference lock.

- [**Ch 33. Speculation: the local win and the distributed verdict**](chapters/33-speculation-and-the-distributed-verdict.md)
  DFlash speculative decode at 107.9 tok/s with exact verification; why
  native NVFP4 speculation stays fail-closed (Fallback B); the measured
  rejection of the linear distributed-verifier lane (110.59 tok/s only on an
  all-accept control; real acceptance 9.23–38.07 %).

## Part VII — Orchestration and Serving

- [**Ch 34. The scheduler and the slots**](chapters/34-scheduler-and-slots.md)
  One scheduler owns one accelerator; 1..=4 resident slots with independent
  KV/sampler/grammar state; decode favored over prefill; the 250 µs
  rendezvous; bounded admission and cancellation discipline.

- [**Ch 35. Ordering, hazards, and the dispatch gap**](chapters/35-ordering-hazards-and-the-dispatch-gap.md)
  RAW/WAW/WAR in a single-queue world; fences and buffer tracking; the
  bounded one-token diagnosis — 104 norm-boundary groups, 39 SWA staging
  groups, 52 KV-publication splits, one bookkeeping copy (removed
  bit-exactly) — and why the obvious norm fusion is rejected for changing
  logprobs beyond contract.

- [**Ch 36. Prefill vs decode: the two graphs**](chapters/36-prefill-vs-decode-paths.md)
  Why prefill is batch GEMM and decode is matvec; `encode_batch_hidden_range`;
  llama's `flash_attn_ext` prefill kernels and the SWA staging-shadow route;
  chunking; the roofline flip between the two regimes.

- [**Ch 37. The server surface: sessions, migration, and the security boundary**](chapters/37-server-sessions-and-security.md)
  The frozen llama-compatible routes plus sessions and migration; stateful
  generation with revisions and idempotency keys; the deliberately
  asymmetric auth model; the unhealthy latch and 503.

## Part VIII — Measurement and Evidence

- [**Ch 38. Measuring against llama.cpp**](chapters/38-measuring-against-llama-cpp.md)
  The parity ledger; same-session interleaved A/B; exact-token five-rep
  means; synthetic vs natural workloads; `accelerator_safe.py` and the flock
  locks; the J0 anchor flip that took decode from 0.781× to above parity;
  what a ratio may and may not claim.

- [**Ch 39. The evidence culture**](chapters/39-the-evidence-culture.md)
  Fail-closed as a design philosophy; the release lock, findings, and the
  feature contract; receipts on the append-only volume; the launch-claims
  register and its ground rules; why copy is never allowed to outrun the
  receipt.

- [**Ch 40. What we measured and rejected**](chapters/40-what-we-measured-and-rejected.md)
  The falsified-hypothesis ledger: the distributed speculative lane; the ANE
  route at 0.827×; the norm-boundary fusion; send-during-prefill; what each
  rejection preserved and what it cost.

---

## Appendices

- [A. Glossary](glossary.md) — every term, defined on first use, indexed here.
- [B. The kernel dispatch table](appendix-kernel-table.md) — the per-layer kernel chain on one page.
- [C. Lanes, flags, and environment](appendix-env-flags.md) — the lane matrix and the `MUSER_*` surface.
- [D. Bibliography](bibliography.md) — master list, mirrored per-chapter.
- [E. The pin](PINNED.md) — the exact trees this book quotes, and the verification recipe.
- [F. The writing contract](STYLE.md) — the rules every chapter was written and reviewed under.
