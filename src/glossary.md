# Appendix A — Glossary

> Every term is defined in place at first use in the chapters; this is the
> index. *(introduced in Ch N)* marks the chapter that defines it; *also*
> lists later chapters that rely on it. Anchors are lowercase-hyphenated
> term names, e.g. `[SIMD group](glossary.md#simd-group)`.

### 104 norm-boundary groups

the three separated-norm families (+51 entry +52 post-attention +1 post-FFN) of the +196-closure serving gap; their fusion is rejected for breaching the 1e-4 logprob contract *([Ch 19](chapters/19-downproj-and-residual.md))*

### 131,072-position limit

`MUSE_MAX_CONTEXT`, the model's per-slot context ceiling; NoPE planes are allocated at exactly this capacity *([Ch 22](chapters/22-the-price-of-context.md))*

### 4-byte read-back

the GPU-resident greedy chain's alternative: read only the argmax result slot per token (`dflash_argmax_results`, u32::MAX = fail-closed nonfinite flag) *([Ch 21](chapters/21-sampling-argmax-and-grammar.md))*

### 95.5/4.5 payload split

at depth, the 13 NoPE planes carry ≈95.5 % of a handoff's bytes and the 39 SWA rings ≈4.5 %; why the wire economy is a NoPE economy *([Ch 22](chapters/22-the-price-of-context.md))*

### accelerator lease

the `flock` on `/tmp/ferrite.gpu.lock` held by a GPU process for its lifetime; the same file discipline on both the Mac and the GX10 *([Ch 28](chapters/28-the-gx10-and-vllm-nvfp4-prefill.md); also Ch 38)*

### AcceleratorPermit

the RAII guard returned by scheduler acquire; dropping it releases the accelerator and wakes the next waiter *([Ch 34](chapters/34-scheduler-and-slots.md))*

### AcceleratorScheduler

the one Mutex+Condvar owner of the shared Metal queue; decode selected first, cyclic fairness *([Ch 2](chapters/02-metal-compute-model.md))*

### acceptance rate

accepted proposals ÷ drafted proposals; gated per-request over a recent 8-round window with a 0.25 floor *([Ch 8](chapters/08-the-dflash-draft.md))*

### access pattern

what memory a kernel reads, in what order, how much; "where the bandwidth story lives" *([Ch 6](chapters/06-the-kquant-family.md))*

### accounting-invariant metric

a timing column defined identically across engines (wall time at 131,008 is one; short-output decode is not) *([Ch 38](chapters/38-measuring-against-llama-cpp.md))*

### Activations pool

the per-sequence GPU buffers of Table 10.2, allocated once, reused every token, zero hot-path allocation (`decode.rs:897`) *([Ch 10](chapters/10-the-forward-pass-at-a-glance.md))*

### adjacent lease window

measuring two lanes back-to-back under one held accelerator lease so drift between them is bounded by seconds *([Ch 38](chapters/38-measuring-against-llama-cpp.md))*

### all-accept control

a diagnostic run under forced 100% acceptance (the distributed lane's 110.59 tok/s standard trace) that proves the pipeline's plumbing while proving nothing about throughput; never citable as serving performance. *([Ch 33](chapters/33-speculation-and-the-distributed-verdict.md); also Ch 40)*

### ALPN

Application-Layer Protocol Negotiation: the protocol name agreed inside the TLS handshake; the data plane's is exactly `muser-kvpack-v2` *([Ch 30](chapters/30-handoff-v2-transport.md))*

### anchor flip

the J0 operator revision that retired Muser's self-referential production hash and made llama's own bytes the exactness gate *([Ch 38](chapters/38-measuring-against-llama-cpp.md))*

### append

the fail-closed ring reservation: position must equal origin_logical+len, else CacheDiscontinuity *([Ch 15](chapters/15-kv-store-and-the-ring.md))*

### append-only evidence volume

`muser-receipt://`, where receipts are immutable (created by exclusive temp+fsync+rename, never replaced) and records journal O_APPEND *([Ch 39](chapters/39-the-evidence-culture.md))*

### append_batch

the chunked ring reservation; overflow advances both origins by the evicted count and returns which source rows remain live *([Ch 23](chapters/23-the-swa-ring-and-the-growing-cache.md))*

### argmax

the reduction returning the index of the largest element; the winning index is itself the token id, so greedy decoding needs no softmax *([Ch 21](chapters/21-sampling-argmax-and-grammar.md))*

### arithmetic ABI

the pinned agreement in op order, reduction tree, and materialization dtype across vendors' math (e.g. CUDA's serial 128-dim attention reduction vs Metal's 32-lane tree) *([Ch 29](chapters/29-cuda-versus-metal.md))*

### arithmetic intensity

a workload's FLOPs per byte read; decode's is ~3.2, fixed by the model format *([Ch 1](chapters/01-why-inference-is-a-memory-problem.md))*

### asymmetric auth model

keyless loopback inference, bearer-or-dashboard management, cookie+CSRF+Origin mutations, authenticated-everything on LAN — security posture chosen by where you bind *([Ch 37](chapters/37-server-sessions-and-security.md))*

### asymmetric quantization

two numbers per block (scale + min); handles blocks not centered on zero at the cost of the extra stored min *([Ch 5](chapters/05-quantization-from-scratch.md))*

### atomic seal bundle

the final-campaign output: all lanes freshly rerun into a hidden directory, fsynced, then exposed with one rename; failure exposes nothing *([Ch 39](chapters/39-the-evidence-culture.md))*

### attention

the only cross-token mixer: each token's Query scores against past Keys (softmax-weighted) and pulls a weighted sum of Values *([Ch 9](chapters/09-muse-glimmer-architecture.md); also Ch 16)*

### attention route ladder

per-layer kernel selection: llama-pinned vec / splitk / ferrite-interleaved, gated by `llama_vec_rows`/`llama_swa` predicates (`decode.rs:5646-5657`) *([Ch 10](chapters/10-the-forward-pass-at-a-glance.md))*

### attention scale

`1/√head_dim = 1/√128 ≈ 0.0884` applied to Q·K scores, independent of the folded qk_scale_factor (`config.rs:279-281`) *([Ch 9](chapters/09-muse-glimmer-architecture.md))*

### attention-output gate

Muse Glimmer's learned per-channel multiplier: attn_out[i] ← attn_out[i] · σ(gate_proj[i]), applied in attention space before o_proj *([Ch 17](chapters/17-sigmoid-gate-and-oproj.md))*

### attn_dim / kv_dim

query space `n_heads×head_dim = 4,096` vs KV space `n_kv_heads×head_dim = 256` (`config.rs:268-273`) *([Ch 9](chapters/09-muse-glimmer-architecture.md))*

### bandwidth

how fast memory can hand bytes to the GPU, measured GB/s; the budget that governs decode *([Ch 1](chapters/01-why-inference-is-a-memory-problem.md))*

### batch-width boundary

the token-count at which the projection kernel changes; a floating-point reduction-order boundary kept llama-exact *([Ch 13](chapters/13-the-qkv-gate-matvec-family.md))*

### begin/finish_dflash_verify_suffix

the Mirror-SD split: run the verify block to a capture layer synchronously, submit the suffix layers + LM head without waiting (`decode.rs:3298, 3635`) *([Ch 10](chapters/10-the-forward-pass-at-a-glance.md))*

### bit-exactness-over-throughput

the disposition that keeps the 104 separated norm boundaries because their available fusion changes logprobs beyond contract *([Ch 35](chapters/35-ordering-hazards-and-the-dispatch-gap.md))*

### bitrate

bits per weight of a stored format: payload bits plus header bits ÷ block size *([Ch 5](chapters/05-quantization-from-scratch.md))*

### blit

a GPU block-copy executed via a blit command encoder rather than a compute kernel *([Ch 3](chapters/03-unified-memory-and-buffers.md))*

### block

a small contiguous group of weights sharing one scale (and optionally a min); the unit of local quantization *([Ch 5](chapters/05-quantization-from-scratch.md))*

### bound inventory

every queue in the server has a number (64 MiB bodies, 30 s timeouts, 256 connections, 64 admissions, 64-deep streams, 64 sessions); overflow is a status code, never a hang *([Ch 37](chapters/37-server-sessions-and-security.md))*

### boundary-token hold-back

the receiver decodes the final prompt token locally, so KV ships for prompt − 1 tokens (one NoPE token = 13,312 B less payload) *([Ch 26](chapters/26-delta-handoff-and-migration.md))*

### bounded-logit policy

a qualification contract mode (`bounded-drift`) in which greedy tokens must be bit-exact while full-logit drift must fit sealed bounds (native lane: max < 11.0, mean < 1.25), declared in the frozen identity and checked per sample and per summary. *([Ch 32](chapters/32-precision-across-the-handoff.md))*

### BPE tokenizer (merge-order aware)

GGUF byte-pair encoding respecting merge priority; vocab 202,048, identity bound by a metadata SHA-256 *([Ch 9](chapters/09-muse-glimmer-architecture.md))*

### cache hit (economics)

an authenticated restore *and*-committed install; session continuations and fresh disaggregated prefills never count *([Ch 25](chapters/25-warm-reuse.md))*

### cached frequency table

the powf-built [head_dim/2] θ table uploaded once; kernels do one multiply, no pow *([Ch 14](chapters/14-qk-norm-and-rope.md))*

### CacheDiscontinuity

the fail-closed error when a KV append position is not exactly `origin_logical + len` (`decode.rs:265-284`) *([Ch 10](chapters/10-the-forward-pass-at-a-glance.md))*

### calibrated gate

a tolerance derived from a measured second quantization (Q6-vs-kquant disagreement) rather than chosen; used for the deep-ladder content controls so no threshold can be accused of convenience. *([Ch 32](chapters/32-precision-across-the-handoff.md))*

### canonical JSON

recursively key-sorted, compact JSON encoding shared by the Rust and Python sides so both compute identical bytes to hash and MAC *([Ch 30](chapters/30-handoff-v2-transport.md))*

### carried frontier

the target-selected token held un-evaluated between speculative rounds; its explicit witness geometry prevents publishing state whose KV rows do not exist yet. *([Ch 33](chapters/33-speculation-and-the-distributed-verdict.md))*

### causal mask

the rule forbidding a query from seeing keys at later positions *([Ch 9](chapters/09-muse-glimmer-architecture.md))*

### chat shift unit

the atomic replay unit of context shift: a user message plus everything attached to its turn (assistant calls, tool results, images) *([Ch 23](chapters/23-the-swa-ring-and-the-growing-cache.md))*

### chat template

the GGUF-embedded Jinja-style prompt renderer; pinned at exactly 7,167 bytes with its own SHA-256 *([Ch 9](chapters/09-muse-glimmer-architecture.md))*

### codebook

the small table of allowed values a quantized index selects from; 4 bits select 16 entries *([Ch 5](chapters/05-quantization-from-scratch.md))*

### command buffer

the "tape": a recorded sequence of GPU instructions handed to the GPU in one shot at commit *([Ch 2](chapters/02-metal-compute-model.md))*

### command-buffer amortization

the whole 52-layer token recorded onto one concurrent encoder, committed once per token (`decode.rs:5448-5458`) *([Ch 10](chapters/10-the-forward-pass-at-a-glance.md))*

### companion tensors

the bound `.nvfp4_scale`, `.nvfp4_scale2` (and optional `.nvfp4_input_scale_inv`) tensors every NVFP4 weight matrix must carry, validated fail-closed at load *([Ch 7](chapters/07-nvfp4-native-lane.md))*

### comparator golden

the pinned llama.cpp byte artifact that gates exactness (since J0: the complete 808,192-byte f32 logit row, SHA-256 `fc37487b…`) *([Ch 38](chapters/38-measuring-against-llama-cpp.md))*

### compute command encoder

the recorder that writes dispatches onto a command buffer *([Ch 2](chapters/02-metal-compute-model.md))*

### concurrent dispatch

an encoder (`MTLDispatchType::Concurrent`) that may overlap dispatches with no dependency between them *([Ch 2](chapters/02-metal-compute-model.md))*

### concurrent dispatch set

multiple independent kernels encoded in one closure so the GPU may overlap them (Q/K/V/gate) *([Ch 13](chapters/13-the-qkv-gate-matvec-family.md))*

### consumer (receiver)

the process that accepts authenticated KV over Handoff V2 and decodes; today the Mac Metal engine *([Ch 27](chapters/27-why-disaggregate.md))*

### content-local sensitivity

a quality exceedance confined to one content class at one depth (docs@65,536: 15.134% vs 13.339% top-token gate) that did not replicate cross-document; published, not capped. *([Ch 32](chapters/32-precision-across-the-handoff.md))*

### content-sensitive envelope

the published form of the NVFP4 quality result: gates are content- and depth-local, and the docs@65,536 exceedance is part of the claim *([Ch 40](chapters/40-what-we-measured-and-rejected.md))*

### context geometry

the enrolled sink+window shape (`DFlashContextGeometry`) bound to the draft's digest; receivers must never infer it locally *([Ch 8](chapters/08-the-dflash-draft.md))*

### ContextPolicy (`Shift`/`Error`)

the server-owned context-overflow policy; the engine has no shift op *([Ch 23](chapters/23-the-swa-ring-and-the-growing-cache.md))*

### counted-warmup convention

the disagg cells' protocol of one uncounted warmup handoff before five counted reps; part of the claim, not a footnote *([Ch 27](chapters/27-why-disaggregate.md))*

### CPU-side exact acceptance

speculative tokens accepted/rejected on the CPU against the target's full distributions (`verify_full_speculative_mt_ordered`, `sampling.rs:1033`) *([Ch 10](chapters/10-the-forward-pass-at-a-glance.md))*

### cross-vendor arithmetic ABI

the pinned set of reduction orders and materialization dtypes that lets CUDA-produced logits match Metal bit-for-bit; what the wizard's one-ULP chase (attempts 10–31) had to version. *([Ch 32](chapters/32-precision-across-the-handoff.md))*

### cross-vendor library

the strict-f32 recompile of muse_reference + nvfp4 whose arithmetic matches CUDA's scalar boundaries for remote-parity routes *([Ch 4](chapters/04-pso-and-three-kernel-sources.md))*

### CSRF

cross-site request forgery: a hostile page makes the victim's browser send a cookie-authenticated request; countered by exact-Origin matching plus a constant-time token on mutations *([Ch 37](chapters/37-server-sessions-and-security.md))*

### current-token bypass

reading the current token's K/V as f32 from the activation buffers instead of the f16 plane; ferrite rung only *([Ch 16](chapters/16-attention-decode-kernels.md))*

### cyclic slot rotation

fairness by ascending sequence-ID order resuming after the last-served ID, implemented identically at both scheduler levels *([Ch 34](chapters/34-scheduler-and-slots.md))*

### D2H gather

the producer-side device-to-host copy of each computed KV layer into pinned host memory on a fenced CUDA side stream, ahead of the TLS send *([Ch 29](chapters/29-cuda-versus-metal.md))*

### DC offset

a block mean far from zero; halves a symmetric grid's effective precision, the case min+offset exists for *([Ch 5](chapters/05-quantization-from-scratch.md))*

### decode

generating tokens one by one after the prompt has been read; the bandwidth-bound regime *([Ch 1](chapters/01-why-inference-is-a-memory-problem.md); also Ch 10)*

### decode-aware chunk shrinking

prefill boundaries collapse from 512 rows to 64 the moment any decoder queues, capping decode's worst-case wait at one small interval *([Ch 34](chapters/34-scheduler-and-slots.md))*

### decode-over-prefill priority

the scheduler rule that any waiting decode outranks all prefill: prefill acquires only when no decode is queued *([Ch 34](chapters/34-scheduler-and-slots.md))*

### DecodeBatcher (250 µs rendezvous)

the server-side decode-step coalescer: request threads keep slot ownership while one elected runner waits ≤250 µs to pack up to four rows (disabled at parallel=1) *([Ch 34](chapters/34-scheduler-and-slots.md))*

### delta handoff

a handoff armed on an exact held prefix so only the suffix crosses the wire; admission requires 256-aligned cut, nonempty suffix, exact held tokens *([Ch 26](chapters/26-delta-handoff-and-migration.md))*

### delta witness

the receiver's record of the observed `(role, layer, start, count)` segment stream, re-checked against the span schedule at prepare time on deferred delta handoffs *([Ch 30](chapters/30-handoff-v2-transport.md))*

### dequantize

reconstruct an approximate value from a stored index: `scale × index + min` (or a float-LUT lookup) *([Ch 5](chapters/05-quantization-from-scratch.md))*

### detached generation

a full replacement state built alongside live decode; swapped in only after the handoff seal validates *([Ch 10](chapters/10-the-forward-pass-at-a-glance.md); also Ch 30)*

### deterministic tiebreak

strictly-greater comparison so equal logits keep the lower index, matching the CPU first-maximum convention; load-bearing for byte-identical diffs *([Ch 21](chapters/21-sampling-argmax-and-grammar.md))*

### DFlash

Muser's five-layer kquant draft assistant, fed pinned target hidden states via a 33,280→6,656 fc projection *([Ch 8](chapters/08-the-dflash-draft.md); also Ch 10)*

### disaggregated prefill

splitting inference across machines by role: a producer prefills the prompt into KV, a consumer receives it and decodes *([Ch 27](chapters/27-why-disaggregate.md))*

### dispatch

the four-line unit of GPU work: bind kernel, bind buffers, set constants, launch N threadgroups *([Ch 2](chapters/02-metal-compute-model.md))*

### dispatch gap (+196)

the reconciled 760-vs-564 closure delta (104 norm-boundary + 39 SWA staging + 52 KV-publication + 1 copy); every cheap removal changes bits *([Ch 10](chapters/10-the-forward-pass-at-a-glance.md))*

### dispatch ladder

the ordered kernel choices per (dtype, token-count) in `encode_quantized_matmul`, kept source-pinned against llama.cpp *([Ch 6](chapters/06-the-kquant-family.md))*

### divergence penalty

the serialization CUDA applies when threads of one warp take different branches *([Ch 29](chapters/29-cuda-versus-metal.md))*

### documentation truth pass

the audit genre that re-reads claim-bearing documents against implementation and retained receipts *([Ch 39](chapters/39-the-evidence-culture.md))*

### dot product

the multiply-and-add pairing of two equal-length vectors into one scalar; the atom under every weight matrix *([Ch 5](chapters/05-quantization-from-scratch.md); also Ch 13)*

### down projection

the FFN exit matvec W_down · ffn_mid ([19968→6656]); carries both Q4_K (74.76 MB) and Q6_K (109.03 MB, +45.8 %) tensors on the release artifact *([Ch 19](chapters/19-downproj-and-residual.md))*

### draft model

the small model that proposes tokens cheaply for a large target to verify; pure overhead that pays when its guesses are accepted *([Ch 8](chapters/08-the-dflash-draft.md))*

### draft trace

the deterministic per-round proposal list (`draft_token_trace`) that qualification compares exactly *([Ch 8](chapters/08-the-dflash-draft.md))*

### DRAM

dynamic RAM, the machine's main system memory (96 GB on the decode Mac) *([Ch 3](chapters/03-unified-memory-and-buffers.md))*

### drift envelope

the measured max/mean absolute full-logit (and KV) deltas between two engines' outputs on a fixed fixture (native vs exact: 7.270581/1.040619 at 32 tokens; 10.884401/1.233789 at 2,048/256); deterministic but nonzero, and published as part of the claim. *([Ch 32](chapters/32-precision-across-the-handoff.md))*

### dual EOS

two end-of-generation control tokens (EOS 200,001 + EOT 200,008) merged into one stop set *([Ch 9](chapters/09-muse-glimmer-architecture.md))*

### dual-eps fused tail

`muser_fused_norm_residual_rms_norm_32sg`: residual add + post-norm (eps 1e-8) + next norm (eps 1e-5) in one kernel; produces the next sub-block's normed input *([Ch 10](chapters/10-the-forward-pass-at-a-glance.md))*

### dual-eps tail

the fused dispatch computing post-norm (eps 1e-8) + residual add + next pre-norm (eps 1e-5) in one kernel *([Ch 12](chapters/12-rmsnorm-and-the-dual-eps-sandwich.md))*

### durable reservation

the write + fsync + rename + directory-fsync dance that persists a generation before any live engine state is published or the ACK leaves *([Ch 30](chapters/30-handoff-v2-transport.md))*

### durable reserve pattern

the crash-safe commit sequence write-temp + fsync + rename + directory-fsync used by the replay ledger; its directory-fsync tail is why operational state must live on the internal disk, never the evidence volume. *([Ch 31](chapters/31-the-wire-discipline.md))*

### E2M1

the 4-bit float codebook: 1 sign + 2 exponent + 1 mantissa; values ±{0, 0.5, 1, 1.5, 2, 3, 4, 6} with relative spacing *([Ch 7](chapters/07-nvfp4-native-lane.md))*

### E4M3FN

the 8-bit float block scale: finite-only (NaN at 0x7f/0xff), max magnitude 448, exponent bias 7 *([Ch 7](chapters/07-nvfp4-native-lane.md))*

### Earley recognizer

the chart parser behind the grammar matcher: keeps every ambiguous parse stack alive and consumes tokens byte-wise, accepting partial UTF-8 sequences *([Ch 21](chapters/21-sampling-argmax-and-grammar.md))*

### EEE (Energy-Efficient Ethernet)

the link's low-power idle mode (LPI); on this lane's burst schedule it produced retransmission blackouts quantized at 6.42 ± 0.03 s, so EEE-off is the enrolled link invariant. *([Ch 31](chapters/31-the-wire-discipline.md))*

### effective read rate

bytes-per-token × measured tokens-per-second (~594 GB/s for kquant decode); derived from measured throughput, not a spec *([Ch 1](chapters/01-why-inference-is-a-memory-problem.md))*

### embedding

a learned lookup table with one row per vocabulary token; "embedding a token" means reading its row *([Ch 11](chapters/11-token-embedding-lookup.md))*

### embedding table (`token_embd.weight`)

the `[hidden_dim × vocab_size]` GGUF tensor, Q4_K (3,744 B/row) or F16 on Muser's lanes *([Ch 11](chapters/11-token-embedding-lookup.md))*

### encode_batch_hidden_range

the batch/serving encoder over a row range; mirrors `encode_token`'s op sequence (`decode.rs:3858`) *([Ch 10](chapters/10-the-forward-pass-at-a-glance.md); also Ch 36)*

### encode_token

the legacy single-token 52-layer Metal graph; teacher-forced benchmark and phase-profile route (`decode.rs:5515`) *([Ch 10](chapters/10-the-forward-pass-at-a-glance.md))*

### encrypted session envelope

the on-disk bundle format: Postcard bytes sealed with XChaCha20Poly1305 under a MUSER-SESSION-V3 magic, 0700 directory, atomic private write *([Ch 37](chapters/37-server-sessions-and-security.md))*

### EOG exclusion

masking the request's end-of-generation tokens inside the argmax reduction only, leaving stored logits byte-identical for logprob and session uses *([Ch 21](chapters/21-sampling-argmax-and-grammar.md))*

### epsilon (ε)

the tiny constant inside the square root keeping the denominator positive; 1e-5 from the GGUF on Muse Glimmer *([Ch 12](chapters/12-rmsnorm-and-the-dual-eps-sandwich.md))*

### exact speculative acceptance

Muser's CPU-side `verify_full_speculative_mt_ordered` contract: full target distributions plus the pinned RNG stream, gated by all-logit-row comparators *([Ch 21](chapters/21-sampling-argmax-and-grammar.md))*

### exact-token gate

a performance rep counts only if the two engines produced byte-identical tokens; divergent outputs mean different work, not different speed *([Ch 38](chapters/38-measuring-against-llama-cpp.md))*

### ExactIdentityV1

the compatibility namespace binding model, revision, adapter, tokenizer, chat-template, and context-policy identities; any difference is a miss *([Ch 24](chapters/24-kvpack-the-format.md))*

### exit 75

the producer's fail-closed death code (EX_TEMPFAIL) on any engine-touched error; a dead producer is recoverable, a degraded one is not *([Ch 28](chapters/28-the-gx10-and-vllm-nvfp4-prefill.md))*

### f16

IEEE 754 16-bit float ("half"): 1 sign + 5 exponent + 10 mantissa bits, ~3 decimal digits, range 2⁻¹⁴ to 65,504 *([Ch 5](chapters/05-quantization-from-scratch.md))*

### f32

IEEE 754 32-bit float: 1 sign + 8 exponent + 23 mantissa bits, ~7 decimal digits *([Ch 5](chapters/05-quantization-from-scratch.md))*

### fail-closed

refusing to proceed when a required ingredient is missing (e.g. Q6_K without the metallib aborts load) rather than silently substituting *([Ch 4](chapters/04-pso-and-three-kernel-sources.md); also Ch 39)*

### Fallback B

the shipping disposition: native NVFP4 serves plain decode, speculation stays kquant-only, and `producer_mode: native` refuses DFlash at startup *([Ch 7](chapters/07-nvfp4-native-lane.md); also Ch 40)*

### falsification ledger

the recording device that lists each hypothesis with its evidence class and verdict (the frontier's 14-attempt disposition table; the dispatch-gap reconciliation), so rejected designs teach instead of haunting. *([Ch 33](chapters/33-speculation-and-the-distributed-verdict.md))*

### falsified-hypothesis ledger

the closing record of what was measured and rejected, each entry carrying hypothesis, experiment, receipt, verdict, and what the rejection preserves *([Ch 40](chapters/40-what-we-measured-and-rejected.md))*

### fast-math

a compiler contract allowing NaN/Inf assumptions and FP reordering for speed; ON for the serving library, OFF for cross-vendor parity *([Ch 4](chapters/04-pso-and-three-kernel-sources.md))*

### fast-math library

Muser's main runtime-compiled shader library, built with fast math on for speed at the token-boundary parity gate *([Ch 29](chapters/29-cuda-versus-metal.md))*

### feature contract

`release/feature-contract-v1.json`, the frozen scope/hardware/policy file that (with findings and the lock) constitutes the campaign identity *([Ch 39](chapters/39-the-evidence-culture.md))*

### FFN (feed-forward network)

the per-token "thinking" sub-block; Muse Glimmer's is SwiGLU at width 19,968 *([Ch 9](chapters/09-muse-glimmer-architecture.md); also Ch 18)*

### findings register

`release/findings-v1.json`, the zero-waiver defect list; release requires zero open findings *([Ch 39](chapters/39-the-evidence-culture.md))*

### fingerprint

a record of what actually ran (e.g. the metallib's SHA-256), derived from resolved state, not an env-var echo *([Ch 4](chapters/04-pso-and-three-kernel-sources.md))*

### five-rep mean

the campaign's unit of evidence: five counted repetitions after the stated warmup convention, reported as mean with coefficient of variation *([Ch 38](chapters/38-measuring-against-llama-cpp.md))*

### flash_contiguous

the prefill route predicate (origin 0, no wrap, fits capacity) that admits the direct store-then-attend routes before any staging *([Ch 36](chapters/36-prefill-vs-decode-paths.md))*

### FLOP

one floating-point operation (a multiply or an add); the unit of arithmetic budget *([Ch 1](chapters/01-why-inference-is-a-memory-problem.md))*

### fma

fused multiply-add, `a·b+c` with one rounding; the per-element dequant instruction of the 4r2s fallback *([Ch 13](chapters/13-the-qkv-gate-matvec-family.md))*

### footprint table

one-slot / four-slot KV bytes at 2k/32k/131k depth (0.191/0.518/1.827 GB per slot, decimal); topology-derived allocation, never peak RSS *([Ch 22](chapters/22-the-price-of-context.md))*

### forward_decode_group

packed decode: 1..=4 sequences' tokens in one concurrent encoder, one commit, one wait (`decode.rs:4869`) *([Ch 10](chapters/10-the-forward-pass-at-a-glance.md))*

### forward_into

the serving entry that routes one token to the batch graph and multi-token input to chunked prefill (`decode.rs:2077`) *([Ch 10](chapters/10-the-forward-pass-at-a-glance.md))*

### four unary nodes

the serving soft cap as llama.cpp builds it: ×scale, ×1/20, tanh, ×20 as separately published kernels, chosen so public bytes match the comparator *([Ch 20](chapters/20-final-norm-lm-head-softcap.md))*

### full-distribution read-back

the serving route's per-token copy of all 202,048 logits (~789 KiB) to the CPU, the price of exact sampling, grammar re-rolls, and speculative acceptance *([Ch 21](chapters/21-sampling-argmax-and-grammar.md))*

### function constant

a per-PSO compile-time value (`[[ function_constant(N) ]]`) that specializes one source into many kernels *([Ch 4](chapters/04-pso-and-three-kernel-sources.md))*

### fused gate+up route

`ffn_q4k_gate_up_silu_4r2s`, the opt-in (`MUSER_FERRITE_FFN_GATE_UP`) single-kernel SwiGLU widening; opt-in because its rounding differs from llama.cpp's node graph *([Ch 18](chapters/18-swiglu-ffn.md))*

### gamma (γ)

the learned per-channel weight applied after normalization; re-learns the shape the norm flattened *([Ch 12](chapters/12-rmsnorm-and-the-dual-eps-sandwich.md))*

### gate activation buffer

the separate 16 KiB `σ(gate)` tensor an unfused gate would materialize; Muser's in-place kernel avoids it *([Ch 17](chapters/17-sigmoid-gate-and-oproj.md))*

### gather

reading many table rows at once indexed by an id array; decode's embedding is a gather of batch width 1 *([Ch 11](chapters/11-token-embedding-lookup.md))*

### GBNF

llama-style grammar format (Backus–Naur with charsets and token terminals) used for structured-output constraints *([Ch 21](chapters/21-sampling-argmax-and-grammar.md))*

### GEMM

general matrix × matrix; the prefill-shape of every projection *([Ch 10](chapters/10-the-forward-pass-at-a-glance.md); also Ch 13)*

### generation number

the per-handoff monotonically increasing counter; admission requires strictly above the ledger's committed high-water mark, and generation 0 is refused outright *([Ch 30](chapters/30-handoff-v2-transport.md))*

### ggml kargs

llama.cpp's packed C struct of tensor extents/strides (`ne00`, `nb01`, …) its Metal kernels take instead of scalars *([Ch 13](chapters/13-the-qkv-gate-matvec-family.md))*

### GGUF

the on-disk model-file format Muser reads: a small header plus weight tensors packed end to end *([Ch 1](chapters/01-why-inference-is-a-memory-problem.md); also Ch 9)*

### GPU

a processor built for ten thousand easy things at once; the unit of work is the thread *([Ch 2](chapters/02-metal-compute-model.md))*

### GpuBuffer

Muser's f32 activation buffer type: one shared MTLBuffer plus a length, with checked CPU slice access *([Ch 3](chapters/03-unified-memory-and-buffers.md))*

### GpuBytes

Muser's raw-byte buffer type, the only one that can carry the GGUF mmap (`_mmap` keeps the mapping alive) *([Ch 3](chapters/03-unified-memory-and-buffers.md))*

### GpuByteView

a checked `(buffer, offset, len)` slice of a GpuBytes; the per-tensor weight handle kernels receive *([Ch 3](chapters/03-unified-memory-and-buffers.md))*

### GpuHalfBuffer

Muser's f16 (binary16) buffer type; kept distinct so an F16 KV plane can never be indexed as F32 *([Ch 3](chapters/03-unified-memory-and-buffers.md))*

### GQA (grouped-query attention)

32 query heads share 2 KV heads (16:1); shrinks KV bytes 16× vs full multi-head *([Ch 9](chapters/09-muse-glimmer-architecture.md))*

### GQA fan-in (32:2, 16:1)

32 query heads share 2 KV heads; kv_head = head/16; the KV read is 16× smaller than full MHA *([Ch 16](chapters/16-attention-decode-kernels.md))*

### grammar rejection sampling

run the ordinary sampler first, check the winner against the grammar, re-roll with a mask only after rejection; the rejected draw still advances every RNG *([Ch 21](chapters/21-sampling-argmax-and-grammar.md))*

### greedy decoding

always emitting the highest-scoring token; deterministic, hence the only policy the exact-token parity gate can diff *([Ch 21](chapters/21-sampling-argmax-and-grammar.md))*

### grid

the entire launch: how many threadgroups; always shaped like the output *([Ch 2](chapters/02-metal-compute-model.md))*

### growing plane

the NoPE regime where capacity is max_context, the ring never wraps, and origin_logical stays 0 *([Ch 15](chapters/15-kv-store-and-the-ring.md))*

### GX10 (GB10)

the lab's ASUS DGX Spark-class node: aarch64 host + NVIDIA GB10 GPU, prefill/storage node only, never a decode destination *([Ch 28](chapters/28-the-gx10-and-vllm-nvfp4-prefill.md))*

### Hadamard product

element-wise vector multiply (a ⊙ b)[j] = a[j]·b[j], no cross-coordinate mixing *([Ch 18](chapters/18-swiglu-ffn.md))*

### half-bit embedding

decoding eight E2M1 codes as f16 bit patterns scaled by 2⁻¹⁴ and folding 2¹⁴ into the block scale; MLX-lineage trick *([Ch 7](chapters/07-nvfp4-native-lane.md))*

### half-split (NEOX) pairing

RoPE convention pairing x[i] with x[i+head_dim/2]; Llama-family convention, wrong for this checkpoint *([Ch 14](chapters/14-qk-norm-and-rope.md))*

### Handoff V2

the authenticated mTLS + HMAC-sealed tile transport from the GX10 producer (Part VI) *([Ch 10](chapters/10-the-forward-pass-at-a-glance.md); also Ch 24, 30)*

### hardware-aware token tree

the one surviving distributed-speculation experiment: spend otherwise-idle GX batch arithmetic covering near-miss branches, admitted only through a preregistered emitted-tokens-per-node screen against the 107.9 bar. *([Ch 33](chapters/33-speculation-and-the-distributed-verdict.md))*

### hazard tracking

Metal's automatic dependency ordering between dispatches touching the same buffer; Muser keeps it on (tracked) plus explicit barriers *([Ch 3](chapters/03-unified-memory-and-buffers.md))*

### head-major layout

`[kv_head][capacity][head_dim]` storage keeping each KV head's history contiguous; the NoPE plane's layout *([Ch 15](chapters/15-kv-store-and-the-ring.md))*

### head_dim

the width of one attention head's Q/K/V vectors; 128 here, read from `attention.key_length` (not `hidden/n_heads` = 208) *([Ch 9](chapters/09-muse-glimmer-architecture.md))*

### heads_per_kv

`n_heads / n_kv_heads = 16` query heads served by each KV head (`config.rs:274-276`) *([Ch 9](chapters/09-muse-glimmer-architecture.md))*

### HMAC

keyed hash-based message authentication code: a cryptographic tag computed over a message with a shared secret key — anyone holding the key can compute it, nobody without the key can forge it *([Ch 30](chapters/30-handoff-v2-transport.md))*

### HMAC-sealed manifest

the terminal `SealManifestV2`: an HMAC-SHA256 tag over the canonical-JSON core binding the begin manifest, every segment descriptor, and the entire payload stream *([Ch 30](chapters/30-handoff-v2-transport.md))*

### hmac_epoch

the counter-space version for the HMAC key id; re-enrollment mints a new epoch so a regenerated PKI starts a disjoint ledger space *([Ch 30](chapters/30-handoff-v2-transport.md))*

### honesty tags

per-field labels on telemetry and claims: `measured` / `target` / `mock` (metrics schema) plus `[precedent-7B-ferrite]` and `[roadmap]` in the claims legend *([Ch 39](chapters/39-the-evidence-culture.md))*

### hybrid postmortem

the measured rejection record for aggressive norm fusion: logprob max error 3.197e-4 over contract, first divergence one f16 ULP in layer-1 V *([Ch 19](chapters/19-downproj-and-residual.md); also Ch 35)*

### Idempotency-Key replay

a retry of the same key+revision+request-digest returns the cached completion instead of generating again; the same key on a different request is a conflict *([Ch 37](chapters/37-server-sessions-and-security.md))*

### inert guard

the discipline (honest skips, ambiguity refusal, load-time aborts) that keeps an unset flag from masquerading as an enabled one *([Ch 4](chapters/04-pso-and-three-kernel-sources.md))*

### input_scale_inv

the optional scalar whose presence selects W4A4 arithmetic; must pair with `muser.activation_precision=nvfp4` or the load fails *([Ch 7](chapters/07-nvfp4-native-lane.md))*

### installed payload

the measured goodput of a handoff: payload bytes over the producer's `TCP_INFO.busy_time`, the campaign's only trusted wire clock. *([Ch 31](chapters/31-the-wire-discipline.md))*

### integer-dot producer

the `MUSER_NVFP4_EXACT=1` producer mode whose NVFP4 arithmetic is deterministic by integer construction; a verification anchor that is never served ("Verification only" in the lane matrix). *([Ch 32](chapters/32-precision-across-the-handoff.md))*

### integer-exact contraction

decoding E2M1/E4M3FN as Q1/Q9 integers so the block dot is an order-free i64 sum with a fixed 2⁻¹⁰/2⁻²⁰ denominator *([Ch 7](chapters/07-nvfp4-native-lane.md))*

### interleaved (GPT-J) pairing

RoPE convention pairing adjacent dimensions 2i and 2i+1 (vs half-split "NEOX"); the pinned checkpoint's convention *([Ch 9](chapters/09-muse-glimmer-architecture.md))*

### interleaved (NORM) pairing

RoPE convention rotating adjacent pairs (x[2i], x[2i+1]); Muse Glimmer's convention *([Ch 14](chapters/14-qk-norm-and-rope.md))*

### interleaved A/B

measuring both engines in the same session, rep by rep, so machine-state noise is common-mode and cancels in the ratio *([Ch 38](chapters/38-measuring-against-llama-cpp.md))*

### INVALID_WRONG_REASON

the label for a run that failed for reasons outside its hypothesis; retained, named, and not retried into a verdict *([Ch 40](chapters/40-what-we-measured-and-rejected.md))*

### JIT compilation

compiling shader source at runtime (`new_library_with_source`) rather than loading a prebuilt library *([Ch 4](chapters/04-pso-and-three-kernel-sources.md))*

### kernel

the per-thread program: one function every thread in a launch runs once over its own data slice *([Ch 2](chapters/02-metal-compute-model.md))*

### kquant

llama.cpp's K-quant family of block-quantized integer formats; Muser's reference-lane weights, mixed per tensor *([Ch 6](chapters/06-the-kquant-family.md))*

### KV cache

the store holding every visible past token's K and V per layer so attention never recomputes them *([Ch 15](chapters/15-kv-store-and-the-ring.md))*

### KV plane

one layer's f16 key+value buffers with explicit ring metadata; live planes zero-fill by design *([Ch 3](chapters/03-unified-memory-and-buffers.md))*

### KV publication

the store dispatch that makes the current token's K/V visible to attention; a split closure per layer in production *([Ch 15](chapters/15-kv-store-and-the-ring.md))*

### KV ring

the fixed 2,048-row token-major KV plane of each SWA layer; bounded at 2 MiB per layer forever *([Ch 9](chapters/09-muse-glimmer-architecture.md))*

### KV row cost (1,024 B)

2 KV heads × head_dim 128 × 2 B f16 × (K+V): what one layer pays to cache one token; identical for both layer classes *([Ch 22](chapters/22-the-price-of-context.md))*

### KV store + memory barrier

the vec-route pattern: store the K/V row, `memory_barrier_with_resources`, then attend *([Ch 10](chapters/10-the-forward-pass-at-a-glance.md))*

### kvpack

the vendored format/protocol family that saves, seals, verifies, and restores exact KV state; "exactness is the product; speed is the consequence" *([Ch 24](chapters/24-kvpack-the-format.md))*

### launch-claims register

`docs/launch-claims.md`, the table every piece of launch copy must be checked against; a number with no row does not ship *([Ch 39](chapters/39-the-evidence-culture.md))*

### layer (block)

one copy of the repeating unit: norm → attention → sandwich-norm residual → norm → FFN → sandwich-norm residual; Muse Glimmer has 52 *([Ch 9](chapters/09-muse-glimmer-architecture.md))*

### layer exit tail

the second fused dual-eps kernel per layer: residual += post_norm(delta) at ε 1e-8, then next layer's input = rms_norm(residual) at ε 1e-5 *([Ch 19](chapters/19-downproj-and-residual.md))*

### LayerNorm

normalize by subtracting the mean and dividing by the standard deviation; two reductions *([Ch 12](chapters/12-rmsnorm-and-the-dual-eps-sandwich.md))*

### LayoutClassV2

a compact layer-class declaration (`from..until` step `except`, kv_heads, dtype, `window_tokens`) in a begin's layout table *([Ch 24](chapters/24-kvpack-the-format.md))*

### leaf pin

rejecting any peer certificate whose leaf SHA-256 is not one of the enrolled digests, even when it chains to the trusted CA *([Ch 30](chapters/30-handoff-v2-transport.md))*

### ledger-volume gate

the receiver's bind-time refusal to run when its replay-ledger directory shows a >100 ms reserve-pattern tail (the 2026-08-18 evidence-volume lesson, enforced in code) *([Ch 30](chapters/30-handoff-v2-transport.md))*

### legs_valid / leg_errors

the stage-5 gate that separates infrastructure failure from correctness results; a timeout can no longer publish as a cache mismatch *([Ch 25](chapters/25-warm-reuse.md))*

### link invariant

a link-level setting a measurement or claim is enrolled under (here: EEE off on the 10GbE path), shipped as production guidance rather than re-derived per run. *([Ch 31](chapters/31-the-wire-discipline.md))*

### llama-pinned kernels

PSOs loaded from the prebuilt llama.cpp metallib (`MUSER_GGML_METALLIB`) so arithmetic matches the comparator bit-for-bit *([Ch 10](chapters/10-the-forward-pass-at-a-glance.md))*

### llama_vec_rows predicate

vec-route eligibility: metallib present, len>0, capacity≥32, and contiguous (unwrapped or full ring) *([Ch 16](chapters/16-attention-decode-kernels.md))*

### LM head

the final `output.weight [6656 × 202048]` projection to logits; a separate tensor from the embedding (untied) *([Ch 9](chapters/09-muse-glimmer-architecture.md))*

### LM head (unembedding)

the largest matvec in the engine, `output.weight` [6656→202048] Q5_K ≈ 924.6 MB read once per token; the inverse role of the embedding *([Ch 20](chapters/20-final-norm-lm-head-softcap.md))*

### local-prefill fallback

the engine's complete Mac-local prefill path, used whenever the producer is unavailable; makes the producer a TTFT SPOF, never a correctness SPOF *([Ch 27](chapters/27-why-disaggregate.md))*

### lockstep MAC

the fused-FFN pattern where one x load feeds both the gate and the up accumulator (and two rows) before touching memory again *([Ch 18](chapters/18-swiglu-ffn.md))*

### logprob

the natural-log probability the model assigns a token, read off the softmaxed distribution; the exactness contract bounds normalized-logprob error (1e-4 in the dispatch-gap rejection) *([Ch 35](chapters/35-ordering-hazards-and-the-dispatch-gap.md); also Ch 38)*

### logit

one raw f32 score per vocabulary token, before any probability or bound; Muse emits 202,048 per token *([Ch 20](chapters/20-final-norm-lm-head-softcap.md))*

### logit scale

the 0.196116 (= 1/√26) multiplier applied to logits BEFORE the soft cap; GGUF metadata `muse-glimmer.logit_scale`, not a code constant *([Ch 9](chapters/09-muse-glimmer-architecture.md))*

### logit_scale

the 0.196116 multiplier applied to logits before the cap, read fail-closed from GGUF metadata `muse-glimmer.logit_scale` (numerically 1/√26) — not a code constant *([Ch 20](chapters/20-final-norm-lm-head-softcap.md))*

### logits

raw per-vocab-entry scores, `[202,048]`, before scale/cap *([Ch 9](chapters/09-muse-glimmer-architecture.md))*

### M16 NVFP4 batch route

the W4A4 quantized-activation GEMM taken by 16-row NVFP4 chunks with 64-aligned inputs (opt-out MUSER_NO_M16_N32) *([Ch 36](chapters/36-prefill-vs-decode-paths.md))*

### marginal KV cost

the per-token bytes added past the window: 13,312 B/token (13 NoPE layers only) vs 53,248 B/token below 2,048 — exactly 13/52 *([Ch 22](chapters/22-the-price-of-context.md))*

### mask/blk block classifier

llama's `flash_attn_ext_blk` skip/partial/dense tile bytes, prepared once per chunk and shared by every full-attention layer, making the pinned causal kernel cheap on the masked triangle *([Ch 36](chapters/36-prefill-vs-decode-paths.md))*

### matvec

matrix-by-vector multiply: one dot product of a weight row against the input vector per output element; decode's every projection *([Ch 6](chapters/06-the-kquant-family.md); also Ch 10)*

### matvec / GEMV

matrix × vector multiply, the one operation decode performs over and over; ~2 FLOPs per weight element *([Ch 1](chapters/01-why-inference-is-a-memory-problem.md); also Ch 13)*

### max-subtraction

subtracting the row max before exp; algebraically exact, prevents overflow to inf/NaN *([Ch 16](chapters/16-attention-decode-kernels.md))*

### maximal coupling

the speculative acceptance rule `accept if rng ≤ min(p/q, 1)` with a residual-corrected resample on rejection, which makes the output marginal exactly the target's regardless of draft quality. *([Ch 33](chapters/33-speculation-and-the-distributed-verdict.md))*

### memoryBarrierWithScope

the explicit barrier Muser plants between dispatch groups on a concurrent encoder, delimiting real graph dependencies *([Ch 2](chapters/02-metal-compute-model.md))*

### Metal

Apple's API for programming its GPUs: the MSL shading language, a host API, and a memory model *([Ch 2](chapters/02-metal-compute-model.md))*

### MetalContext

Muser's long-lived GPU state: device, queue, and the three kernel libraries *([Ch 2](chapters/02-metal-compute-model.md))*

### MetalKvPlane

one layer's K+V f16 buffer pair plus explicit ring metadata: capacity, len, origin_logical, origin_physical, head_major (`decode.rs:182`) *([Ch 10](chapters/10-the-forward-pass-at-a-glance.md); also Ch 15)*

### metallib

an `MTLLibrary` serialized to disk; loading it skips the frontend compiler entirely *([Ch 4](chapters/04-pso-and-three-kernel-sources.md))*

### metallib pin

loading llama.cpp's own prebuilt Metal library (`MUSER_GGML_METALLIB`) so comparator kernels run bit-identically instead of being re-expressed; fingerprinted by SHA-256 in every route identity *([Ch 29](chapters/29-cuda-versus-metal.md))*

### MetalShared

one executor per accelerator: context, kernels, mmap'd weights, scheduler; shared by all slots (`decode.rs:958`) *([Ch 10](chapters/10-the-forward-pass-at-a-glance.md))*

### MetalSpeculativeCheckpoint

transactional KV protection: NoPE planes rewind metadata only; SWA planes retain the ≤16 rows a block may overwrite (`decode.rs:213-226`) *([Ch 10](chapters/10-the-forward-pass-at-a-glance.md))*

### min

the offset where an asymmetric block's codebook starts; `value = scale × index + min` *([Ch 5](chapters/05-quantization-from-scratch.md))*

### min+offset

the asymmetric-quantization trick: spend one extra stored number per block to place the grid exactly where the data lives *([Ch 5](chapters/05-quantization-from-scratch.md))*

### Mirror-SD

the split-graph speculative verify overlap: execute the target through a capture layer synchronously, submit the remaining layers and LM head without waiting, and accept no result until the pending suffix completes (`begin/finish_dflash_verify_suffix`). *([Ch 33](chapters/33-speculation-and-the-distributed-verdict.md))*

### miss control

the unrelated-prompt leg that must stay slow for a warm-hit claim to be valid; proof of reuse, not cache-forever *([Ch 25](chapters/25-warm-reuse.md))*

### mmap

mapping a file's bytes directly into the process address space, lazily, page by page *([Ch 3](chapters/03-unified-memory-and-buffers.md))*

### mock

the honesty tag for a field with no backing measurement; the dashboard renders it unavailable and historical Ferrite results are never inserted as live Muser metrics *([Ch 39](chapters/39-the-evidence-culture.md))*

### MSL

Metal Shading Language, the C++ dialect kernels are written in *([Ch 2](chapters/02-metal-compute-model.md))*

### Mt19937

the in-tree bit-for-bit reimplementation of libc++'s std::mt19937 (llama.cpp's sampler RNG), so seeded results are stable across engines and Rust releases *([Ch 21](chapters/21-sampling-argmax-and-grammar.md))*

### MTLBuffer

Metal's handle to a range of GPU-addressable memory *([Ch 3](chapters/03-unified-memory-and-buffers.md))*

### MTLCommandQueue

the queue command buffers come from; created once at startup *([Ch 2](chapters/02-metal-compute-model.md))*

### MTLDevice

the handle to the (one, system-default) GPU; allocates memory and compiles shaders *([Ch 2](chapters/02-metal-compute-model.md))*

### MTLLibrary

a bundle of compiled kernel functions addressable by name; the middle compile stage *([Ch 4](chapters/04-pso-and-three-kernel-sources.md))*

### mTLS (mutual TLS)

Transport Layer Security with both sides presenting certificates; Handoff V2 is TLS 1.3-only with an exact ALPN and pinned leaf certificates *([Ch 30](chapters/30-handoff-v2-transport.md))*

### MuseConfig

the fully-resolved hyperparameter set parsed fail-closed from GGUF metadata; every field cited, no silent defaults (`config.rs:106`) *([Ch 9](chapters/09-muse-glimmer-architecture.md))*

### MuseIdentity digest

one SHA-256 over all eight Muse identity dimensions; scopes the resident radix so a wrong identity is structurally unreachable *([Ch 24](chapters/24-kvpack-the-format.md))*

### MuseLayerKind

`SlidingRope` / `FullNoPe` enum; `uses_rope()` is true iff sliding (`config.rs:55-70`) *([Ch 9](chapters/09-muse-glimmer-architecture.md))*

### MUSER_FERRITE_FFN_GATE_UP

opt-in (default OFF) fused `ffn_q4k_gate_up_silu_4r2s` FFN; opt-in because the pinned baseline packet regressed with it (`decode.rs:980-983`) *([Ch 10](chapters/10-the-forward-pass-at-a-glance.md))*

### muser_scale_softcap_inplace

the fused tail kernel: ×logit_scale then tanh soft cap in one pass over the logits row (`muse_reference.metal:15`) *([Ch 10](chapters/10-the-forward-pass-at-a-glance.md))*

### native lane

the product lane: Mac decodes NVFP4 weights directly, FP16 KV, F16 LM head, remote NVFP4 prefill *([Ch 7](chapters/07-nvfp4-native-lane.md))*

### natural-text cell

a real-corpus measurement standing alongside the synthetic matrix since the half-window bug; cross-engine outputs may diverge, so speed stands without an exactness gate *([Ch 38](chapters/38-measuring-against-llama-cpp.md))*

### negative fixture

a known-bad implementation retained to guard a contract (the 104-group fusion whose logits changed) *([Ch 40](chapters/40-what-we-measured-and-rejected.md))*

### nibble

four bits, values 0–15; two pack into one byte, low nibble first in every format in this book *([Ch 5](chapters/05-quantization-from-scratch.md))*

### nonloopback bind gate

the server refuses to listen on a non-loopback address unless a TLS certificate, a mode-0600 key, and a mode-0600 API-key file are all supplied *([Ch 37](chapters/37-server-sessions-and-security.md))*

### NoPE

the 13 full-attention layers ({3,7,…,51}) with no positional rotation at all; `layer % 4 == 3` *([Ch 14](chapters/14-qk-norm-and-rope.md))*

### NoPE (FullNoPe)

full causal attention with NO positional rotation at all; the 13 layers at indices 3,7,…,51 *([Ch 9](chapters/09-muse-glimmer-architecture.md))*

### NoPE growing plane

the 13 full layers' head-major KV plane that grows to max_context; rows are position-free bytes *([Ch 9](chapters/09-muse-glimmer-architecture.md))*

### NoPE tiles during prefill

the transfer schedule streaming the 13 position-free NoPE tiles (~6.5 MiB per 512 tokens) while CUDA prefill runs (`schedule.rs:1-21`) *([Ch 10](chapters/10-the-forward-pass-at-a-glance.md))*

### norm epsilon (ε)

the small constant under the RMS square root; 1e-5 from the GGUF for every norm except the two post-norms *([Ch 9](chapters/09-muse-glimmer-architecture.md))*

### norm-boundary fusion

merging separated norm-boundary dispatches; rejected in Muser's campaign for breaching logprob tolerance *([Ch 12](chapters/12-rmsnorm-and-the-dual-eps-sandwich.md))*

### normalization

any op that pins a vector's magnitude to a predictable band so deep stacks of multiplies don't drift *([Ch 12](chapters/12-rmsnorm-and-the-dual-eps-sandwich.md))*

### notarial vs non-notarial

sealed, independently reproducible release evidence versus unsealed engineering evidence; everything in the 2026-08 campaign is `seal_eligible: false` *([Ch 39](chapters/39-the-evidence-culture.md))*

### nr0 / rows_per_group

output rows each threadgroup reduces in the pinned matvec kernels; 2 for Q4_K/Q6_K, 1 for Q5_K *([Ch 13](chapters/13-the-qkv-gate-matvec-family.md))*

### NVFP4

NVIDIA's 4-bit float weight format: E2M1 payloads, one E4M3FN scale per 16 values, one f32 scale2 per tensor; exactly 4.5 bits/weight *([Ch 7](chapters/07-nvfp4-native-lane.md))*

### Nvfp4ProducerMode

receiver config enum `Exact | Native` selecting the producer's numeric contract; modes keep separated cache identities *([Ch 7](chapters/07-nvfp4-native-lane.md))*

### o_proj (output projection)

the matvec mixing the 32 gated attention heads (4,096 wide) back into the 6,656-wide residual stream; Q4_K [4096→6656] per layer *([Ch 17](chapters/17-sigmoid-gate-and-oproj.md))*

### one-button wizard

`muser node add` / dashboard Add node: preflight, pinned deploy, SHA-verified model placement, TLS+HMAC enrollment, daemon start, and the three-handoff qualification recipe *([Ch 30](chapters/30-handoff-v2-transport.md))*

### one-row batch graph

the serving decode route: `forward_batch` with `token_count = 1`, chosen because it dispatches the exact pinned llama kernels (`decode.rs:2085-2091`) *([Ch 10](chapters/10-the-forward-pass-at-a-glance.md))*

### one-shot bandwidth hit

a per-token (not per-layer) weight read, like the LM head's 924.6 MB: a flat slice of the budget, never a 52× multiplier *([Ch 20](chapters/20-final-norm-lm-head-softcap.md))*

### online softmax

the running (max, sum, accumulator) formulation that never materializes the score array; rescales by exp(old−new) *([Ch 16](chapters/16-attention-decode-kernels.md))*

### operational state

replay ledgers, sockets, locks: state that must live on the internal disk because the evidence volume's fsync tail poisons TTFT *([Ch 39](chapters/39-the-evidence-culture.md))*

### OPERATOR REVIEW REQUIRED

the register tier for evidence-backed proposed wording that stays dead until the owner approves it, even if its reproduction gate passes *([Ch 39](chapters/39-the-evidence-culture.md))*

### origin_logical / origin_physical

the ring's logical start vs its physical slot; placement is never derived from absolute position *([Ch 10](chapters/10-the-forward-pass-at-a-glance.md); also Ch 15)*

### overhead

the per-block header bytes (scales, mins) amortized over the block's elements; what block size trades against local range *([Ch 5](chapters/05-quantization-from-scratch.md))*

### pacing pin

the producer-side `SO_MAX_PACING_RATE` socket cap that deliberately holds the sender under line rate (8 Gbps against a ~9.4 Gbps path) so the kernel smooths the handoff's bursts; fail-closed in both the readback and the receipt validator. *([Ch 31](chapters/31-the-wire-discipline.md))*

### pack

kvpack's append-only durable container: 4 KiB header ‖ canonical manifest ‖ 4 KiB footer, commit written last, published by atomic rename *([Ch 24](chapters/24-kvpack-the-format.md))*

### packed decode group

`forward_decode_group`: 1..=4 ready decode rows from distinct slots packed into one concurrent encoder, one commit, one wait — one weight pass *([Ch 34](chapters/34-scheduler-and-slots.md))*

### page alignment

`new_buffer_with_bytes_no_copy` requires page-aligned pointer and length; Apple Silicon pages are 16 KB *([Ch 3](chapters/03-unified-memory-and-buffers.md))*

### page fault

the OS trap that pulls a chunk of a mapped file into memory on first touch *([Ch 3](chapters/03-unified-memory-and-buffers.md))*

### parameter

one learned number inside the model, tuned during training; a 30B model holds ~30 billion *([Ch 1](chapters/01-why-inference-is-a-memory-problem.md))*

### parity ledger

the append-only campaign ledger (`docs/goal-parity-ledger-2026-08.md`) whose entries record hypothesis, change, exactness gate, verdict, and receipt paths *([Ch 38](chapters/38-measuring-against-llama-cpp.md))*

### partials `[M, S, O]`

the per-workgroup online-softmax state (max, denominator, weighted values) a reducer merges *([Ch 16](chapters/16-attention-decode-kernels.md))*

### PCIe

the bus connecting CPU and discrete GPU; the memcpy path unified memory eliminates *([Ch 3](chapters/03-unified-memory-and-buffers.md))*

### permutation-invariance

raw attention's property of producing the same scores under token reordering; why position must be injected *([Ch 14](chapters/14-qk-norm-and-rope.md))*

### PhaseProfiler closures

the one-command-buffer-plus-wait units the dispatch gap counts; profiling closures, not raw Metal dispatches *([Ch 10](chapters/10-the-forward-pass-at-a-glance.md); also Ch 2, Ch 35)*

### pinned metallib (`MUSER_GGML_METALLIB`)

the prebuilt llama.cpp kernel library loaded as binary provenance for parity *([Ch 13](chapters/13-the-qkv-gate-matvec-family.md))*

### pinned vec per-query prefill

short chunks (<20 queries) run llama's own vec flash kernel once per query row with exact visible prefixes, reusing the upstream PSO and reduction order *([Ch 36](chapters/36-prefill-vs-decode-paths.md))*

### positional encoding

any scheme baking token order into the vectors before attention, which is permutation-invariant without it *([Ch 14](chapters/14-qk-norm-and-rope.md))*

### post_norm_eps

the 1e-8 epsilon of the two sandwich post-norms; a llama.cpp graph constant, NOT GGUF metadata (`config.rs:28`) *([Ch 9](chapters/09-muse-glimmer-architecture.md))*

### post_norm_eps (1e-8)

llama.cpp's hard-coded epsilon for the two sandwich post-norms; not carried in the GGUF *([Ch 12](chapters/12-rmsnorm-and-the-dual-eps-sandwich.md))*

### pre-decoded scales

decoding all 8 sub-block scale/min pairs once per super-block so the inner loop uses constants *([Ch 13](chapters/13-the-qkv-gate-matvec-family.md))*

### precise::cos / precise::sin

Metal's per-call high-accuracy trig, used at RoPE's large angles despite fast-math compilation *([Ch 14](chapters/14-qk-norm-and-rope.md))*

### prefill

reading the prompt (many tokens at once, weight rows reused); the compute-friendly regime *([Ch 1](chapters/01-why-inference-is-a-memory-problem.md); also Ch 10)*

### PREFILL_BATCH_TOKENS / MAX_TEACHER_FORCED_TOKENS

the 512-row idle-prefill chunk that shrinks to 64 rows once a decode waits (`decode.rs:53-54`) *([Ch 10](chapters/10-the-forward-pass-at-a-glance.md))*

### prefix_cut

the delta boundary lifted from raw JSON beside the typed begin manifest (the typed protocol drops unknown keys); 0 means full transfer *([Ch 26](chapters/26-delta-handoff-and-migration.md))*

### PREPARED/staged-render/WAL/activation/ACK

the V2 research protocol's durable verification transaction (fsynced commit WAL before idempotent renderer activation); implemented and fault-tested locally, deliberately unwired from serving. *([Ch 33](chapters/33-speculation-and-the-distributed-verdict.md))*

### preregistered bar

a performance threshold fixed before the deciding runs (e.g. ≥ 99.151 % IID per-edge acceptance; the 200 ms GPU verifier gate) *([Ch 40](chapters/40-what-we-measured-and-rejected.md))*

### producer

the process that runs prefill and ships the KV; a role, not a machine (today: the resident vLLM NVFP4 process on the GX10) *([Ch 27](chapters/27-why-disaggregate.md))*

### producer mode (exact/native)

the producer's two lanes: `native` = tensor-core NVFP4 fast path; `exact` = integer-dot verification producer; selected by the Python-only `MUSER_NVFP4_EXACT` flag, recorded receiver-side as `Nvfp4ProducerMode` *([Ch 28](chapters/28-the-gx10-and-vllm-nvfp4-prefill.md))*

### provenance.json

the vendored-tree manifest (schema `muser.vendored-source.v1`): upstream commit/tag/tree, per-file SHA-256 map, and recorded patches *([Ch 24](chapters/24-kvpack-the-format.md))*

### PSO

pipeline state object: one library function lowered all the way to machine code for this specific GPU; what `set_compute_pipeline_state` binds *([Ch 4](chapters/04-pso-and-three-kernel-sources.md))*

### PsoCache

Muser's in-process name→PSO registry; panics on an unregistered name rather than falling back *([Ch 4](chapters/04-pso-and-three-kernel-sources.md))*

### public-logprob tolerance

the parity contract the legacy fused single-token graph breaches; the reason serving refuses it *([Ch 10](chapters/10-the-forward-pass-at-a-glance.md))*

### Q4_K

144 bytes per 256 elements (4.5 bits): two f16 super-scales, 8× 6-bit packed sub-scales/mins, 128 bytes of nibbles; min+offset *([Ch 6](chapters/06-the-kquant-family.md))*

### Q5_K

176 bytes per 256 elements (5.5 bits): Q4_K plus a 32-byte high-bit plane *([Ch 6](chapters/06-the-kquant-family.md))*

### Q6_K

210 bytes per 256 elements (6.5625 bits): signed 6-bit codes split across ql/qh planes, 16 int8 sub-scales, f16 super-scale last *([Ch 6](chapters/06-the-kquant-family.md))*

### QK-norm

per-head RMSNorm (over each 128-vector) applied to Q and K before RoPE; eps 1e-5 *([Ch 9](chapters/09-muse-glimmer-architecture.md); also Ch 14)*

### qk_scale_factor

≈3.87 scalar broadcast into every Q-norm weight by the converter; folded gain on top of 1/√128 *([Ch 9](chapters/09-muse-glimmer-architecture.md))*

### qk_scale_factor (≈3.87)

the uniform Q-norm gain the converter synthesized; folded into Q, independent of the 1/√128 softmax scale *([Ch 14](chapters/14-qk-norm-and-rope.md))*

### QkNormProbe

load-time verification that q/k norm tensors are the converter's constant broadcasts; aborts on a learned norm (`loader.rs:98-138`) *([Ch 9](chapters/09-muse-glimmer-architecture.md); also Ch 14)*

### qualification recipe

the identity-declared check a node must pass to become healthy: exactly three ordered 2,048/256 handoffs, exact tokens under the lane's logit policy (bounded for native/text; exact full logits plus DFlash trace for the combined lane) *([Ch 30](chapters/30-handoff-v2-transport.md); also Ch 32)*

### quantization

storing each learned number in fewer bits than a full float (four-ish bits per weight on the kquant lane) *([Ch 1](chapters/01-why-inference-is-a-memory-problem.md); also Ch 5)*

### quantization error

the gap between the original value and its reconstruction; bounded by scale/2 for nearest rounding *([Ch 5](chapters/05-quantization-from-scratch.md))*

### quantize

the inverse mapping: pick the codebook entry (index) closest to the original value *([Ch 5](chapters/05-quantization-from-scratch.md))*

### query / key / value (Q/K/V)

what this token seeks / what each past token offers for matching / the payload it returns *([Ch 16](chapters/16-attention-decode-kernels.md))*

### query head / KV head

one of 32 independent 128-wide attentions vs one of 2 shared K/V providers *([Ch 9](chapters/09-muse-glimmer-architecture.md))*

### RAR non-hazard

read-after-read: concurrent readers of immutable data need no ordering; the packed decode group's shared weight reads *([Ch 35](chapters/35-ordering-hazards-and-the-dispatch-gap.md))*

### ratio convention

all throughput ratios in this book are `llama ÷ muser`; above 1.0 means Muser wins *([Ch 38](chapters/38-measuring-against-llama-cpp.md))*

### raw ceiling

the single-stream TCP rate the physical path sustains with zero tuning (9.40 Gbps pre-rebuild direct; 9.256 Gbps GX10→Mac on the switched fabric, asymmetric in reverse); a property of a topology that must be re-proven after any change. *([Ch 31](chapters/31-the-wire-discipline.md))*

### RAW hazard

read-after-write: a consumer kernel must see a producer's bytes; the dominant decode dependency, ordered by the store→barrier→attend pattern *([Ch 35](chapters/35-ordering-hazards-and-the-dispatch-gap.md))*

### read-modify-write

the hazard class of any in-place `x = x ∘ y` GPU statement; safe only when each thread owns disjoint elements and dispatches are ordered *([Ch 17](chapters/17-sigmoid-gate-and-oproj.md))*

### readiness receipt

the single receipt issued only after all findings are closed and every unsealed lane report belongs to the exact campaign identity *([Ch 39](chapters/39-the-evidence-culture.md))*

### red-team pass

self-audit by independent reviewers (ledger forensics, receipt recompute, statistics, framing) that certifies the evidence, not the decision built on it *([Ch 39](chapters/39-the-evidence-culture.md))*

### reference lock

an explicit lane users can route to when the native lane's published sensitivity matters (the kquant lane), chosen manually because an automatic disagreement proxy would itself require a second reference computation. *([Ch 32](chapters/32-precision-across-the-handoff.md))*

### release lock

`release/release-lock.json`, the authoritative single file whose containment state disables seals, candidates, and publication regardless of evidence strength *([Ch 39](chapters/39-the-evidence-culture.md))*

### relocatable KV

NoPE cache bytes valid at any position (no rotation baked in); "relocate = memcpy" *([Ch 14](chapters/14-qk-norm-and-rope.md))*

### relocate = memcpy

the NoPE consequence: a KV tile for positions [a,b) can be planted anywhere without recomputation — "the whole kvpack free lunch" (`lib.rs:9-10`) *([Ch 9](chapters/09-muse-glimmer-architecture.md))*

### remote KV install

`begin_remote_kv_install`: build detached planes, scatter received tiles, validate, then atomically swap into live decode (`decode.rs:1852-1994`) *([Ch 10](chapters/10-the-forward-pass-at-a-glance.md))*

### replay ledger

the durable per-`key_id:epoch` record of the highest committed generation; the only defense against a validly-signed replay of an already-installed handoff *([Ch 30](chapters/30-handoff-v2-transport.md); also Ch 31)*

### residency set

an `MTLResidencySet` attaching the 16+ GiB weight arena once so Metal skips per-token residency work; fails open *([Ch 3](chapters/03-unified-memory-and-buffers.md))*

### resident producer

the pinned, long-lived producer process in a docker container on the GX10 (vLLM at a pinned commit, digest-pinned image and checkpoint) *([Ch 28](chapters/28-the-gx10-and-vllm-nvfp4-prefill.md))*

### residual relay

the two-buffer residual stream: `activations.normed` carries the residual, `activations.post_norm` the normed sub-block output; the last tail writes `activations.hidden` (`decode.rs:5546-5548`) *([Ch 10](chapters/10-the-forward-pass-at-a-glance.md))*

### residual stream (hidden state)

the per-token vector every layer only ever *adds* to; `[6656]` f32 in Muse Glimmer *([Ch 9](chapters/09-muse-glimmer-architecture.md))*

### restart ritual

the only correct producer restart: move the O_EXCL startup receipt and RoPE cache aside, remove the stale socket, check the accelerator lease, restart, and wait for the fresh startup receipt *([Ch 28](chapters/28-the-gx10-and-vllm-nvfp4-prefill.md))*

### reuse ladder

the ordered exact-prefix tiers: current session → resident → durable → remote; each stricter tier caps the one before *([Ch 25](chapters/25-warm-reuse.md))*

### revision CAS

optimistic concurrency for sessions: commit advances revision only if the record still holds the client's expected value *([Ch 37](chapters/37-server-sessions-and-security.md))*

### ring

the fixed-capacity circular store where new rows overwrite the oldest; capacity = min(max_context, 2,048) on SWA layers *([Ch 15](chapters/15-kv-store-and-the-ring.md))*

### ring rotation

where origin_physical points; must be preserved on restore because attention's float accumulation is scan-order sensitive *([Ch 15](chapters/15-kv-store-and-the-ring.md))*

### rms / mean of squares

√((1/n)Σxⱼ²); equals the standard deviation exactly when the mean is zero *([Ch 12](chapters/12-rmsnorm-and-the-dual-eps-sandwich.md))*

### RMSNorm

`x / sqrt(mean(x²)+ε) ⊙ γ`; no mean subtraction; rescales the stream to ~unit magnitude *([Ch 9](chapters/09-muse-glimmer-architecture.md); also Ch 12)*

### roofline

the compute-vs-memory crossover picture; a workload below the machine's balance point is memory-bound *([Ch 1](chapters/01-why-inference-is-a-memory-problem.md))*

### roofline flip

the ~500× jump in arithmetic intensity between decode (~3.2 FLOPs/byte) and a 512-token prefill chunk (~1,619 FLOPs/byte): the two regimes want opposite machines *([Ch 27](chapters/27-why-disaggregate.md))*

### RoPE

rotary positional embedding: rotate Q/K pairs by position-dependent angles so Q·K depends only on position difference *([Ch 14](chapters/14-qk-norm-and-rope.md))*

### RoPE (rotary positional embedding)

position injected by rotating Q/K coordinate pairs by position-dependent angles; SWA layers only, theta 500,000 *([Ch 9](chapters/09-muse-glimmer-architecture.md))*

### rope theta (freq base)

base frequency 500,000 (`rope.freq_base_swa`); larger theta ⇒ positions distinguishable over longer ranges *([Ch 9](chapters/09-muse-glimmer-architecture.md))*

### rotation matrix

the 2×2 `[cos −sin; sin cos]` that spins one coordinate pair; RoPE's only arithmetic *([Ch 14](chapters/14-qk-norm-and-rope.md))*

### route ladder

the four attention routes (llama vec / splitk / batch-store vec / ferrite interleaved) selected per layer per token *([Ch 16](chapters/16-attention-decode-kernels.md))*

### row-major

weight layout where each output row's elements are contiguous in memory; makes one row one DRAM span *([Ch 13](chapters/13-the-qkv-gate-matvec-family.md))*

### rsqrt vs 1/sqrt

fast-math reciprocal sqrt vs the IEEE-rounded pair; a few ULP per call that compound across a token *([Ch 12](chapters/12-rmsnorm-and-the-dual-eps-sandwich.md))*

### sampler state

per-request RNG streams (distribution, XTC, mirostat, adaptive) plus scalars, snapshotted with sessions so a resumed request replays its exact draw sequence *([Ch 21](chapters/21-sampling-argmax-and-grammar.md))*

### sampling

converting logits to a (temperature-scaled, top-k/top-p filtered) distribution and drawing a token; needs the full vocab row, not just the winner *([Ch 21](chapters/21-sampling-argmax-and-grammar.md))*

### sandwich norm

Gemma-2-style placement: each sub-block's *output* is normalized (post_attention_norm / post_ffw_norm) before the residual add *([Ch 9](chapters/09-muse-glimmer-architecture.md); also Ch 12)*

### scale

the local step size of a block's codebook, stored in floating point because it must span wide magnitude ranges *([Ch 5](chapters/05-quantization-from-scratch.md))*

### scale2

the per-tensor f32 multiplier applied last in NVFP4's pinned order `(e2m1 × e4m3fn) × scale2` *([Ch 7](chapters/07-nvfp4-native-lane.md))*

### score

the scaled dot product Q·K/√128 that measures a past token's relevance *([Ch 16](chapters/16-attention-decode-kernels.md))*

### segment role

the closed 9-variant plane vocabulary (NoPE/SWA × K/V, packed tiles, DFlash, Auxiliary); the two-regime economics as wire types *([Ch 24](chapters/24-kvpack-the-format.md))*

### serial prefill dispatch

`MUSER_SERIAL_PREFILL_DISPATCH`: restores the serial prefill encoder for exact A/B against the default concurrent grouping *([Ch 35](chapters/35-ordering-hazards-and-the-dispatch-gap.md))*

### session bundle

the durable unit binding model/tokenizer/template/layout-ABI/DFlash/vision identities to target KV, logits, RNG streams, sampler, grammar, detokenizer, and replay state *([Ch 37](chapters/37-server-sessions-and-security.md))*

### SessionCacheSnapshot

the engine↔kvpack interchange cut: 39 SWA planes carry the logical tail, 13 NoPE planes `[0, position)`, shape-gated fail-closed *([Ch 23](chapters/23-the-swa-ring-and-the-growing-cache.md))*

### sigmoid

the logistic function σ(x) = 1/(1 + e^(−x)), mapping any real into (0, 1); the smooth on/off curve behind both Muse gates *([Ch 17](chapters/17-sigmoid-gate-and-oproj.md))*

### sigmoid attention gate

Muse Glimmer's extra `[6656→4096]` projection; `attn_out ⊙ σ(gate)` applied between attention and o_proj *([Ch 9](chapters/09-muse-glimmer-architecture.md))*

### SiLU

activation `x·σ(x)`, also called Swish *([Ch 9](chapters/09-muse-glimmer-architecture.md))*

### SiLU (Swish)

the activation x·σ(x): ReLU-like at the extremes, smooth, and slightly negative for small negative x (dips to ≈ −0.278 near x ≈ −1.28) *([Ch 18](chapters/18-swiglu-ffn.md))*

### SIMD group

exactly 32 threads executing in lockstep on one SIMD ALU; the unit that actually matters on Apple Silicon *([Ch 2](chapters/02-metal-compute-model.md))*

### simd_sum

one-instruction 32-way reduction across a SIMD group's lanes *([Ch 2](chapters/02-metal-compute-model.md))*

### single-rep diagnostic

a one-sample cell marked `†` that never joins a matrix and cannot support a claim *([Ch 38](chapters/38-measuring-against-llama-cpp.md))*

### sink

the pinned first 64 rows of the DFlash context cache (`DFLASH_CONTEXT_SINK_SIZE`), part of the ABI, not GGUF metadata *([Ch 8](chapters/08-the-dflash-draft.md))*

### sliding_window_pattern

the GGUF period-4 key resolving `[sliding, sliding, sliding, full]`; missing key ⇒ panic, not a guess *([Ch 9](chapters/09-muse-glimmer-architecture.md))*

### slot (serving)

one of 1..=4 resident request contexts with independent KV/RNG/sampler/grammar state over shared immutable weights; KV is the per-slot, depth-scaling term *([Ch 22](chapters/22-the-price-of-context.md))*

### SlotPool

bounded server admission for the 1..=4 resident slots: at most 64 waiters, immediate overload rejection past that, and the permanent unhealthy latch on poisoned state *([Ch 34](chapters/34-scheduler-and-slots.md))*

### slow-client cancellation

a streaming writer backpressured past its 5 s grace (channel depth 64) is cancelled with 499; a slow reader can waste its own request, never the accelerator *([Ch 34](chapters/34-scheduler-and-slots.md))*

### slow-volume refusal

the receiver's bind-time probe rejecting a replay-ledger volume whose reserve pattern (write+fsync+rename+dir-fsync) tails past 100 ms *([Ch 24](chapters/24-kvpack-the-format.md))*

### SoC

system-on-chip: CPU cores and GPU on one piece of silicon pointing at the same DRAM *([Ch 3](chapters/03-unified-memory-and-buffers.md))*

### soft cap

`l → 20·tanh(l/20)`: confines logits to (−20, 20) without a cliff; changes what cross-engine logit comparison means *([Ch 9](chapters/09-muse-glimmer-architecture.md); also Ch 20)*

### softmax

`exp(xᵢ)/Σⱼexp(xⱼ)`: turns attention scores into positive weights summing to 1 *([Ch 9](chapters/09-muse-glimmer-architecture.md); also Ch 16)*

### source receipt

a provenance JSON binding a built artifact to its exact source commit, per-file hashes, and toolchain *([Ch 4](chapters/04-pso-and-three-kernel-sources.md))*

### speculative acceptance rule

accept a draft token with probability min(p/q, 1) against the full target distribution; on rejection, sample the renormalized max(p−q, 0) residual *([Ch 21](chapters/21-sampling-argmax-and-grammar.md))*

### speculative checkpoint

the per-block transactional rollback: NoPE planes rewind metadata only; SWA planes retain the ≤16 live ring rows a block overwrites *([Ch 23](chapters/23-the-swa-ring-and-the-growing-cache.md))*

### speculative decoding

draft k tokens, verify all k+1 rows in one target batch; exact because acceptance is decided against full target distributions *([Ch 8](chapters/08-the-dflash-draft.md))*

### speculative verify

the block-shaped third regime: the target scores up to `MAX_DFLASH_BLOCK = 16` drafted tokens in one pass *([Ch 10](chapters/10-the-forward-pass-at-a-glance.md))*

### splitk

splitting the KV scan across workgroups/SIMD groups, each producing partials the reducer combines *([Ch 16](chapters/16-attention-decode-kernels.md))*

### staging generation

the full-capacity rebuild context (target + DFlash) kept deliberately outside the slot pool so atomic context shift can never become a fifth serving slot *([Ch 34](chapters/34-scheduler-and-slots.md))*

### staging session

the one full-capacity generation reserved for atomic context rebuilds, deliberately outside the slot pool; publication is a pointer swap *([Ch 23](chapters/23-the-swa-ring-and-the-growing-cache.md))*

### staging shadow

the detached F16 buffer where a wrapped SWA prefill stages old ring rows + the new chunk as one contiguous logical span before ring commit *([Ch 23](chapters/23-the-swa-ring-and-the-growing-cache.md))*

### staging-shadow route

wrapped-SWA prefill: stage old ring rows plus the chunk into a detached F16 shadow (llama's padded absolute indices for the one-row variant), attend from the shadow, commit ring metadata only after *([Ch 36](chapters/36-prefill-vs-decode-paths.md))*

### stateful generation

chat requests carrying session_id + expected_revision + Idempotency-Key (all or none); the server holds the frontier and refuses concurrent writers with 409 *([Ch 37](chapters/37-server-sessions-and-security.md))*

### storage mode

Metal's declaration of where buffer bytes live: Shared, Private, or Managed *([Ch 3](chapters/03-unified-memory-and-buffers.md))*

### StorageModeShared

one copy of the bytes in unified memory, CPU- and GPU-visible; the only mode Muser uses *([Ch 3](chapters/03-unified-memory-and-buffers.md))*

### strict-f32 cross-vendor library

the second compile of `muse_reference.metal` + `nvfp4.metal` with fast math off, used by `MUSER_CROSS_VENDOR_QK` routes so Q/K match the CUDA producer's scalar boundaries exactly *([Ch 29](chapters/29-cuda-versus-metal.md))*

### sub-block

the 32-element (K4/K5) or 16-element (K6) group inside a super-block carrying its own scale/min *([Ch 6](chapters/06-the-kquant-family.md))*

### SUPERSEDED banner

the correction written on the artifact itself when a document's numbers are invalidated; history is preserved, never edited *([Ch 39](chapters/39-the-evidence-culture.md))*

### supervisor latch

`supervise_resident_producer.py`'s policy of giving up after three consecutive failed starts (with doubling backoff) rather than flapping forever *([Ch 28](chapters/28-the-gx10-and-vllm-nvfp4-prefill.md))*

### SWA (sliding-window attention)

each query sees only the previous 2,048 tokens (`p1 − p0 < n_swa`); Muse Glimmer's 39 SlidingRope layers *([Ch 9](chapters/09-muse-glimmer-architecture.md))*

### SWA amortization

the property that 39 of 52 layers stop billing at token 2,048, so a 64× longer context costs only ~16.75× more KV *([Ch 22](chapters/22-the-price-of-context.md))*

### SWA layer

one of the 39 sliding-window layers (window 2,048) that carry RoPE *([Ch 14](chapters/14-qk-norm-and-rope.md))*

### SWA staging

copying a wrapped ring into llama's padded absolute indices so the pinned kernel sees llama's exact reduction lanes *([Ch 16](chapters/16-attention-decode-kernels.md))*

### SWA window re-send

the delta's mandatory fresh 2,048-token window (≈81.79 MB): held ring rows are position-bound (RoPE) and cannot serve the new tail *([Ch 26](chapters/26-delta-handoff-and-migration.md))*

### SwiGLU

gated FFN: `down(SiLU(gate·x) ⊙ (up·x))` over width 19,968 *([Ch 9](chapters/09-muse-glimmer-architecture.md); also Ch 18)*

### symmetric quantization

one number per block (a scale), codebook spanning ±scale·max, zero maps to zero; cheapest, wastes range on offset blocks *([Ch 5](chapters/05-quantization-from-scratch.md))*

### synthetic fixture

a deterministic fixed prompt (period-8 cycle of 9 token ids) enabling exact cross-engine comparison; predictable from token identity alone *([Ch 38](chapters/38-measuring-against-llama-cpp.md))*

### target

the full model whose exact distributions decide every accepted token; the draft never decides *([Ch 8](chapters/08-the-dflash-draft.md))*

### target-cache identity

the digest a receiver pins per producer recipe so KV entries from different recipes (exact vs native) can never alias; part of the cluster config, not derivable from the model file alone. *([Ch 32](chapters/32-precision-across-the-handoff.md))*

### target-engine epoch

the named identity of *which* engine executed authoritative target transitions (Mac/Metal vs GX/vLLM); silently switching epochs mid-session is not exact, so fallbacks need a signed seam. *([Ch 33](chapters/33-speculation-and-the-distributed-verdict.md))*

### targeted resource barrier

`memory_barrier_with_resources` naming exactly the buffers a dependency flows through (K/V planes, partials, staged shadow) instead of a whole-scope stall *([Ch 35](chapters/35-ordering-hazards-and-the-dispatch-gap.md))*

### teacher-forced

a decode benchmark cell that feeds known prior tokens (e.g. 32) rather than model-generated ones *([Ch 1](chapters/01-why-inference-is-a-memory-problem.md); also Ch 38)*

### teacher-forced route

feeding known tokens through the exact one-token graph to match the comparator's no-readback policy (`decode.rs:2118-2137`) *([Ch 10](chapters/10-the-forward-pass-at-a-glance.md))*

### tensor core

a matrix-multiply unit consuming low-precision operands natively (FP4 on the GB10); the arithmetic engine of producer-side NVFP4 prefill *([Ch 27](chapters/27-why-disaggregate.md))*

### theta / rope_base_swa

the base of the per-pair frequency table θᵢ = base^(−2i/head_dim), read from `rope.freq_base_swa` *([Ch 14](chapters/14-qk-norm-and-rope.md))*

### thread

one execution lane running the kernel once, identified by `thread_position_in_grid` *([Ch 2](chapters/02-metal-compute-model.md))*

### threadgroup

a block of 32–1024 threads sharing on-chip memory and barriers; the unit of co-scheduling *([Ch 2](chapters/02-metal-compute-model.md))*

### threadgroup barrier

`threadgroup_barrier(...)`: every thread in a threadgroup waits until prior threadgroup-memory writes are visible *([Ch 2](chapters/02-metal-compute-model.md))*

### threadgroup memory

on-chip shared memory visible to all threads of one threadgroup (`threadgroup(0)`) *([Ch 2](chapters/02-metal-compute-model.md))*

### TLB

translation-lookaside buffer, the small cache of virtual→physical page translations; fewer buffers means less pressure *([Ch 3](chapters/03-unified-memory-and-buffers.md))*

### token

one unit of text, roughly a piece of a word; the model emits tokens one at a time *([Ch 1](chapters/01-why-inference-is-a-memory-problem.md); also Ch 9)*

### token-major layout

`[capacity][kv_dim]` storage keeping one token's two KV-head rows adjacent; the SWA ring's layout *([Ch 15](chapters/15-kv-store-and-the-ring.md))*

### TPOT (time per output token)

the decode-dominated per-token latency a streaming user feels; DFlash's target, as TTFT is disaggregation's *([Ch 36](chapters/36-prefill-vs-decode-paths.md))*

### tracked-by-default (b9678d4)

every Muser buffer allocates with Metal hazard tracking on; the one global-untracked experiment changed DFlash conditioning with identical greedy IDs and was reverted *([Ch 35](chapters/35-ordering-hazards-and-the-dispatch-gap.md))*

### transactional checkpoint

the pre-round KV snapshot (`MetalSpeculativeCheckpoint`): NoPE rewind is metadata-only, SWA rings retain the ≤16 rows a block may overwrite *([Ch 8](chapters/08-the-dflash-draft.md))*

### transfer status vocabulary

`starting → transferring → destination_committed → source_restored → completed`, with `ambiguous` and `source_restored_remote_retained` as crash-safe degradations *([Ch 26](chapters/26-delta-handoff-and-migration.md))*

### transformer

an attention-plus-FFN block stack that maps a token sequence to a next-token prediction; the 2017 architecture of [arxiv:1706.03762] *([Ch 9](chapters/09-muse-glimmer-architecture.md))*

### tree reduction

parallel reduction by stride-halving folds with a barrier per pass; log₂(n) steps inside one threadgroup *([Ch 21](chapters/21-sampling-argmax-and-grammar.md))*

### trust ladder

the three strictness rungs of the disaggregated lanes: bit-exact full handoff → exact tokens + declared bounded logits → parity-within-noise; each rung is declared in the lane's identity, with its costs and what it rules out. *([Ch 32](chapters/32-precision-across-the-handoff.md))*

### TTFT (time to first token)

the latency from prompt submission to the first generated token; the user-visible quantity prefill dominates *([Ch 27](chapters/27-why-disaggregate.md))*

### TTFT cliff

the linear, compute-bound growth of local prefill TTFT with prompt depth (6.48 s at 2,048 to 570.12 s at the 131k class on one Mac) *([Ch 27](chapters/27-why-disaggregate.md))*

### two-phase migration

copy/move where the destination durably commits before a move may delete the source; status idempotently queryable after ambiguous failures *([Ch 26](chapters/26-delta-handoff-and-migration.md))*

### two-phase reduction

chunk the input (⌈202,048/1,024⌉ = 198 threadgroups), reduce each chunk to a (value, index) partial, then reduce the partials in one final threadgroup *([Ch 21](chapters/21-sampling-argmax-and-grammar.md))*

### ULP

unit in the last place: the smallest increment a floating-point representation can express at a given exponent; the rejected hybrid's first divergence was a single f16 ULP flip in one layer-1 V value *([Ch 35](chapters/35-ordering-hazards-and-the-dispatch-gap.md); also Ch 32)*

### unfused control route

the serving default FFN widening: two pinned ggml matvecs plus the pointwise `muser_silu_mul_inplace`, exact to the comparator's graph *([Ch 18](chapters/18-swiglu-ffn.md))*

### unhealthy latch

poisoned accelerator/session state flips a permanent flag: every request 503s until an operator restarts; fail-closed as user-visible HTTP *([Ch 37](chapters/37-server-sessions-and-security.md))*

### unified memory

Apple Silicon's single physical DRAM pool shared by CPU and GPU; no copy needed to move data between them *([Ch 3](chapters/03-unified-memory-and-buffers.md))*

### untied embeddings

`token_embd.weight` and `output.weight` are two independent required tensors (`config.rs:294-298`) *([Ch 9](chapters/09-muse-glimmer-architecture.md); also Ch 11)*

### verifier-only ceiling

`output_tokens ÷ sum(remote verifier wall)`: an upper bound granting zero cost to drafting, transport, installation, and scheduling; the decisive rejection bound when it sits under the bar. *([Ch 33](chapters/33-speculation-and-the-distributed-verdict.md); also Ch 40)*

### verify length

proposals per round: exactly 3, 7, or 15; harness pins 15, serving froze 7 for natural-text robustness *([Ch 8](chapters/08-the-dflash-draft.md))*

### visible window

min(position+1, window) rows the kernel may read; SWA masks by addressing, never reading out-of-window tokens *([Ch 16](chapters/16-attention-decode-kernels.md))*

### vocab / vocab_size

the set (and count) of token ids; 202,048 for Muse Glimmer *([Ch 11](chapters/11-token-embedding-lookup.md))*

### VRAM

a discrete GPU's own private memory pool, filled by copying across PCIe *([Ch 3](chapters/03-unified-memory-and-buffers.md))*

### W4A16

weight-only NVFP4 mode: 4-bit weights, wide activations quantized per 256-value super-block to a Q8-K-style grid; the selected product artifact *([Ch 7](chapters/07-nvfp4-native-lane.md))*

### W4A4

weights and activations both 4-bit (FP4 groups of 16, gated by `input_scale_inv`); on the Mac, qualified for batch parity only *([Ch 7](chapters/07-nvfp4-native-lane.md))*

### WAR hazard

write-after-read: an overwriter must wait for outstanding readers — the ring-lap risk explicit origin bookkeeping prevents *([Ch 35](chapters/35-ordering-hazards-and-the-dispatch-gap.md))*

### warm hit

a request served entirely from installed state: no producer compute, no transfer (`producer_driven: false`); shallow 64.631 ms vs deep 0.6132 s / 1.0566 s are different scopes *([Ch 25](chapters/25-warm-reuse.md))*

### warp

CUDA's unit of 32 threads executing in lockstep; the counterpart abstraction to Metal's SIMD group, and the reason the ferrite-lineage kernels ported cleanly *([Ch 29](chapters/29-cuda-versus-metal.md))*

### WAW hazard

write-after-write: two writers of one buffer must serialize or the final bytes are order-dependent *([Ch 35](chapters/35-ordering-hazards-and-the-dispatch-gap.md))*

### WebSocket ticket

the 30-second single-use secret minted by /v1/ws-tickets so long-lived bearer keys never ride URLs; consumed and removed on first use *([Ch 37](chapters/37-server-sessions-and-security.md))*

### weight

one learned coefficient of the model, multiplied in during the forward pass; Muse Glimmer has ≈27.9 B of them by tensor count *([Ch 5](chapters/05-quantization-from-scratch.md))*

### weight-stream-bound

the structural cost of a remote verifier: one full model-weight read per verification round, which no protocol tuning removes (why the linear distributed lane's ceilings capped at 20.15–55.96 tok/s). *([Ch 33](chapters/33-speculation-and-the-distributed-verdict.md))*

### weights

the parameters collectively: the giant tables of learned numbers stored on disk in a GGUF *([Ch 1](chapters/01-why-inference-is-a-memory-problem.md))*

### wire as memory bus

the Part VI physical picture: between a CUDA address space and a Metal address space, the network carries what a memcpy would on either machine alone *([Ch 29](chapters/29-cuda-versus-metal.md))*

### wire floor

the minimum transfer time for a given payload at a given rate (2k ≈ 224 ms, 131k ≈ 3.75 s at 3.9 Gbps); the irreducible wire bill before any overhead. *([Ch 31](chapters/31-the-wire-discipline.md))*

### witnessed final logits

the exact-hit requirement that a cut ending a generation carries the final target distribution with the KV *([Ch 25](chapters/25-warm-reuse.md))*

### workgroup cap (nwg=32)

the split schedule's fixed 32 workgroups matching llama's launch; the admitted "depth-rent" vs higher occupancy *([Ch 16](chapters/16-attention-decode-kernels.md))*

### zero-copy

getting data to the GPU without moving bytes: the mmap'd GGUF becomes one MTLBuffer as-is *([Ch 3](chapters/03-unified-memory-and-buffers.md))*
