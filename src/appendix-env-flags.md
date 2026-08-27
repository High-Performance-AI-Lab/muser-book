# Appendix C — Lanes, flags, and environment

> **status:** draft  ·  **path:** Muse Glimmer, pinned Muser tree
>
> Two tables: the shipped lane matrix (what each weight lane is for, with
> its evidence tags), and the curated `MUSER_*` environment surface — the
> flags a reader operating or qualifying the engine can actually meet.
> Every meaning below was read from the code that consumes the flag, at
> pinned commit `6d0807da` (see [PINNED.md](PINNED.md)); the raw grep
> inventory holds ~84 names, most of them test/bench fixture plumbing that
> this table deliberately omits.

## C.1 The lane matrix

The shipped matrix at `[docs/muser-architecture.md]` (§ "Lanes"), with the
measured cells from the campaign ledger. Ratios are llama ÷ muser
(>1.0 means muser wins); every number keeps its scope tag.

| Lane | Prefill | Decode | Speculative | Intended use |
|---|---|---|---|---|
| **Native NVFP4** (`muser.weight_precision=nvfp4`, `loader.rs:73-90`) | GX10 tensor-core FP4 via disaggregated Handoff V2; Mac-local batch graph as fallback (`prefill.rs:10-12`) | Mac NVFP4 weights, FP16 KV: **35.491 tok/s** vs kquant 35.440 — parity within noise (+0.1444 %), **never claimed faster** `[claims #11]` | **Rejected, fail-closed** (Fallback B): native W4A4 batched verify ran 6.805 tok/s against the 107.9 bar `[nvfp4-fast-lane-evidence; ledger F-series]` | Fast product lane |
| **kquant / reference** (`q4_k_xl`, the release artifact) | Mac batch prefill graph (Appendix B.4) | **35.440 tok/s** (CV 0.037 %), the control cell `[ledger P1.3]` | **107.9 tok/s** — the pre-window-fix five-rep verdict (107.9136, ratio 1.3273) that survives as the kquant spec *bar*; the current fixed-window synthetic restatement is 1.23692 @2,048 / 1.20323 @16,384 / 1.19616 @32,768, five exact reps each `[ledger L2; claims #15]` | Speculative serving + the explicit reference lock |
| **Exact NVFP4** (`MUSER_NVFP4_EXACT=1`, **producer-side Python**) | Integer-dot verification producer on the GX10 | Mac NVFP4 weights (same decode) | Verification only | Deterministic anchor for bounded-logit policies `[docs/muser-architecture.md §Lanes; Ch 32]` |

Two boundaries worth restating (Ch 32–33):

- **`MUSER_NVFP4_EXACT` does not exist in Rust.** It is read only by the
  producer-side Python on the node — `resident_producer.py:32-36` maps it
  to the closed `exact`/`native` producer lane (values other than `0`/`1`
  abort), and the *native* benchmark refuses `=1` outright because
  importing the exact modules would invalidate the native-path claim
  (`benchmark_native_prefill.py:99-102`). The Mac-side counterpart is the
  receiver config enum `Nvfp4ProducerMode` (`muser-cluster/src/config.rs`),
  not an env flag.
- Decode parity is a **paired five-rep measurement** on one cell; the
  unquantized F16 LM head (~3.46 ms/token vs kquant 1.75) is why the
  native lane's edge is only +0.1444 % `[claims #11]`.

## C.2 The curated `MUSER_*` flag table

Grouped by subsystem. "Default" is what the code does when the flag is
absent, where that is evident from the consumer. Unless noted, a flag's
presence (any value) enables it.

### Engine and lane selection (Rust, `muser-engine`)

| Flag | Meaning (from the consuming code) | Default | Chapter |
|---|---|---|---|
| `MUSER_GGML_METALLIB` | Path to the pinned llama.cpp `.metallib`; loads the third kernel source (`context.rs:122-131`) and enables the ggml matvec/matmul/norm/rope/unary and `flash_attn_ext` PSOs. Q6_K tensors refuse to load without it (`decode.rs:114`) and the llama attention routes expect it fail-closed (`attn.rs:493`) | unset — llama-pinned kernels unavailable; engine runs fallback kernels where they exist | [4](chapters/04-pso-and-three-kernel-sources.md), [13](chapters/13-the-qkv-gate-matvec-family.md), [16](chapters/16-attention-decode-kernels.md) |
| `MUSER_GGML_METALLIB_RECEIPT` | Provenance receipt path for the metallib, consumed by node qualification tooling when no explicit `--ggml-metallib-receipt` was given (`muser-server/src/node/mod.rs:83`) | unset | [4](chapters/04-pso-and-three-kernel-sources.md), [38](chapters/38-measuring-against-llama-cpp.md) |
| `MUSER_CROSS_VENDOR_QK` | Switches every op in the token graph to the strict-f32 cross-vendor kernels (fast-math OFF library): projections (`qkv.rs:311`), norms and fused tails decompose with barriers at each model-dtype boundary (`norm.rs:208-225`), attention (`decode.rs:5645`), gate (`gate.rs:14`), softcap (`lmhead.rs:197`). The CUDA-parity arithmetic ABI lane | unset — serving fast-math kernels | [29](chapters/29-cuda-versus-metal.md), [32](chapters/32-precision-across-the-handoff.md) |
| `MUSER_CROSS_VENDOR_ROPE_CACHE` | Path to a retained RoPE frequency-table file: must be a regular file (no symlinks) of exactly `context_length × head_dim × 4` bytes, else fail-closed (`decode.rs:1217-1248`); also routes RoPE through the cross-vendor kernel (`rope.rs:62-64`) | unset — in-memory cached frequencies | [32](chapters/32-precision-across-the-handoff.md) |
| `MUSER_CROSS_VENDOR_ROPE_BYPASS` | Skips the RoPE dispatch entirely in cross-vendor comparisons (`rope.rs:65-67`) | unset — RoPE runs | [14](chapters/14-qk-norm-and-rope.md), [32](chapters/32-precision-across-the-handoff.md) |
| `MUSER_FERRITE_FFN_GATE_UP` | Uses the fused Ferrite `ffn_q4k_gate_up_silu_4r2s` FFN kernel — only when both gate and up tensors are Q4_K, which the release artifact's are (`decode.rs:5819-5836`; gate read once `decode.rs:1334`) | unset — exact two-matvec + `muser_silu_mul_inplace` control | [18](chapters/18-swiglu-ffn.md) |
| `MUSER_NO_FUSED_PREFILL_DUAL_NORM` | Diagnostic control: splits the fused dual-eps norm tails into their separate kernels (`decode.rs:1330-1331`) | unset — fused tails on | [12](chapters/12-rmsnorm-and-the-dual-eps-sandwich.md), [35](chapters/35-ordering-hazards-and-the-dispatch-gap.md) |
| `MUSER_SERIAL_PREFILL_DISPATCH` | Encodes the prefill graph serially (non-concurrent dispatch type) (`decode.rs:1332-1333`) | unset — concurrent dispatch | [35](chapters/35-ordering-hazards-and-the-dispatch-gap.md), [36](chapters/36-prefill-vs-decode-paths.md) |
| `MUSER_NO_LLAMA_FA_PREFILL` | Forces the local FlashAttention-2 route over the llama-pinned prefill kernel: the explicit-disable input of `llama_fa_prefill_route_available` (`decode.rs:56-63`, read `:1481`) | unset — llama pinned prefill kernel eligible at ≥ 20 queries | [36](chapters/36-prefill-vs-decode-paths.md) |
| `MUSER_MULTI_COL_VERIFY` | Default-off multi-column verify matvec: `=1` admits only dtypes whose multi-column output is bitwise identical to the per-token matvec (Q4_K, Q5_K); `=all` adds Q6_K (agrees to a few ULP, not bitwise) (`multicol.rs:49-82`) | unset — one full matvec per token per projection | [33](chapters/33-speculation-and-the-distributed-verdict.md) |
| `MUSER_NO_M16_N32` | Disables the M16 weight-stationary tiles: the kquant 16-row blocks (`qkv.rs:556-579`) and the NVFP4 two-pass prequant M16 route (`decode.rs:5955-5958`, `qkv.rs:163-168`) | unset — M16 tiles eligible | [33](chapters/33-speculation-and-the-distributed-verdict.md), [36](chapters/36-prefill-vs-decode-paths.md) |

### Profiling and diagnostics

| Flag | Meaning | Default | Chapter |
|---|---|---|---|
| `MUSER_METAL_PHASE_PROFILE` | Runs the single-token graph through a `PhaseProfiler` (one command buffer + wait per closure) and prints the labeled per-phase report (`decode.rs:5440-5447`; also the one-token branch `:2133`). The instrument behind the +196-closure dispatch-gap accounting — closures, not raw Metal dispatches | unset | [35](chapters/35-ordering-hazards-and-the-dispatch-gap.md) |
| `MUSER_METAL_BATCH_PHASE_PROFILE` | Same profiler for the batch (prefill/decode-group) graphs (`decode.rs:2881`, `:3184`) and the `muser-metal-phase-diagnostic` binary (`muser-bench/src/metal_phase.rs:138`) | unset | [35](chapters/35-ordering-hazards-and-the-dispatch-gap.md), [36](chapters/36-prefill-vs-decode-paths.md) |
| `MUSER_STREAM_DECODE_PROFILE` | `=1` installs streamed-decode diagnostics retrievable via `take_stream_decode_diagnostics` (`decode.rs:31-35`) | unset | [10](chapters/10-the-forward-pass-at-a-glance.md) |
| `MUSER_METAL_LIVE_TRACE` | `=1` live-trace mode in `muser-metal-phase-diagnostic`; must be absent or exactly `1`, and forbids the isolated phase profilers (`metal_phase.rs:132-142`) | unset | [35](chapters/35-ordering-hazards-and-the-dispatch-gap.md) |
| `MUSER_METAL_CAPTURE_PAUSE_MS` + `MUSER_METAL_CAPTURE_READY_FILE` | GPU frame-capture rendezvous in the phase diagnostic: an O_EXCL ready file plus a pause (both or neither; requires live trace) (`metal_phase.rs:364-382`) | unset | [35](chapters/35-ordering-hazards-and-the-dispatch-gap.md) |
| `MUSER_TTFT_CAPTURE_READY_FILE` + `MUSER_TTFT_CAPTURE_PAUSE_MS` | Server-TTFT capture rendezvous in `scripts/bench_server_ttft.py:197-230`: creates an O_EXCL `muser.server-ttft-capture-ready.v1` JSON file, fsyncs it, then pauses 1,000–30,000 ms so an external capture can attach before the measured request | unset | [31](chapters/31-the-wire-discipline.md), [38](chapters/38-measuring-against-llama-cpp.md) |

### DFlash (speculative lane)

| Flag | Meaning | Default | Chapter |
|---|---|---|---|
| `MUSER_DFLASH_VERIFY_LEN` | Overrides the default verification length for tuning runs; read exactly once (a mid-process change would falsify earlier reported `draft_len`) (`muser-server/src/openai.rs:56-67`) | `DFLASH_VERIFY_LEN = 7` (`openai.rs:49`) — the frozen serving length | [33](chapters/33-speculation-and-the-distributed-verdict.md) |
| `MUSER_DFLASH_GATE` | `=off` is the diagnostic kill switch for the acceptance window gate (8-round window, 0.25 floor, warmup 2, re-qualify backoff); any other value keeps the gate on (`dflash/spec.rs:120-154`) | gate on | [33](chapters/33-speculation-and-the-distributed-verdict.md) |
| `MUSER_DFLASH_CYCLE_TRACE` | `=1` populates the per-cycle `DFlashCycleTrace` (draft/verify/cycle ns, drafted/accepted); never consulted by any route or acceptance decision (`spec.rs:95-96`, `:147`) | unset — empty trace | [33](chapters/33-speculation-and-the-distributed-verdict.md), [38](chapters/38-measuring-against-llama-cpp.md) |
| `MUSER_DFLASH_PRE_DRAFT_IDLE_MS` | Diagnostic idle injection before draft rounds; requires `MUSER_DFLASH_CYCLE_TRACE=1` (`spec.rs:1718-1726`) | unset | [33](chapters/33-speculation-and-the-distributed-verdict.md) |
| `MUSER_DFLASH_MIRROR_OVERLAP` | `=1` enables the exact mirror-SD overlap route (draft overlapping target verify), only on macOS+Metal and only when the projection backend supports it (`spec.rs:443-452`) | unset | [33](chapters/33-speculation-and-the-distributed-verdict.md) |
| `MUSER_DFLASH_SAMPLED_REPLAY` | `=1` keeps the previous sampled-verification route (verify-all, rollback, re-run) for one release; default is the single-pass transactional route (`spec.rs:2326-2341`) | single-pass | [33](chapters/33-speculation-and-the-distributed-verdict.md) |
| `MUSER_DFLASH_SINK` / `MUSER_DFLASH_WINDOW` | Diagnostic-only overrides of the draft context sink/window geometry; absent keeps the shipped (GGUF-read) geometry (`spec.rs:126-133`, `:520-537`) | shipped geometry — sink 64, window from `dflash.attention.sliding_window` | [8](chapters/08-the-dflash-draft.md) |
| `MUSER_DFLASH_PREPARE_TRACE` | Trace prints around prepared-draft execution (`spec.rs:1083-1140`) | unset | [8](chapters/08-the-dflash-draft.md) |
| `MUSER_DFLASH_CAPTURE_FC_PIPELINE` | Selects the public-CoreML FC-slice capture pipeline inside the staged target verifier — the previously invisible v8 overlap cost (`spec.rs:456-458`) | unset | [33](chapters/33-speculation-and-the-distributed-verdict.md) |

### GX10 / wire (producer-side, on the node)

These are **Python-side** flags in `scripts/gx10/`; the Mac Rust code only
measures what they configure.

| Flag | Meaning | Default | Chapter |
|---|---|---|---|
| `MUSER_NVFP4_EXACT` | **Producer-side Python only** (§C.1): `0`/`1` selects the closed native/exact producer lane; any other value aborts (`resident_producer.py:32-36`); the native benchmark refuses `=1` (`benchmark_native_prefill.py:99-102`) | `0` — native | [28](chapters/28-the-gx10-and-vllm-nvfp4-prefill.md), [32](chapters/32-precision-across-the-handoff.md) |
| `MUSER_GX10_PACING_BYTES_PER_SECOND` | Overrides the handoff payload pacing ceiling without a redeploy; a value the kernel refuses still fails closed in `configure_linux_pacing` (`scripts/gx10/llamacpp/muser_v2_send.py:55-76`) | `1_000_000_000` B/s (8.0 Gbps — ~15 % under the 9.4 Gbps raw line) | [31](chapters/31-the-wire-discipline.md) |
| `MUSER_GX10_WIRE_TRACE` | Gates per-segment wire telemetry; unset, empty, or `0` disables (`muser_v2_send.py:956-960`) | off | [31](chapters/31-the-wire-discipline.md) |
| `MUSER_VLLM_WATCHDOG_SECONDS` | Watchdog timeout for the resident vLLM producer container (`resident_producer.py:24`) | `900` s | [28](chapters/28-the-gx10-and-vllm-nvfp4-prefill.md) |
| `MUSER_DFLASH_JOBS_FIFO` | FIFO path over which the resident producer hands DFlash capture jobs to `muser_vllm` (`resident_producer.py:484`; consumer `muser_vllm/dflash_capture.py:352`) | set by the producer runtime | [28](chapters/28-the-gx10-and-vllm-nvfp4-prefill.md), [33](chapters/33-speculation-and-the-distributed-verdict.md) |
| `MUSER_ACCELERATOR_LEASE_WAIT_SECONDS` | Bounded wait (clamped 0–60 s) for the node's accelerator lease before the producer fails (`resident_producer.py:40-44`) | `0` — no wait | [28](chapters/28-the-gx10-and-vllm-nvfp4-prefill.md), [38](chapters/38-measuring-against-llama-cpp.md) |

### Qualification and bench discipline

| Flag | Meaning | Default | Chapter |
|---|---|---|---|
| `MUSER_ACCELERATOR_LEASE` | `=1` is the proof that the process is a child of `accelerator_safe.py`; fail-closed bench executors refuse to run GPU work without it (`muser-bench/src/remote.rs:371`, `kvpack.rs:104`) | unset — execution refused | [38](chapters/38-measuring-against-llama-cpp.md) |
| `MUSER_REMOTE_QUALIFY` | Path override for the qualification binary the node tooling drives (`muser-server/src/node/mod.rs:135-139`) | `<repo>/target/release/muser-remote-qualify` | [30](chapters/30-handoff-v2-transport.md), [38](chapters/38-measuring-against-llama-cpp.md) |
| `MUSER_REMOTE_QUALIFY_SERIAL` | Runs the remote qualification reps serially instead of interleaved (`remote.rs:1666`, `:2534`) | unset — interleaved | [38](chapters/38-measuring-against-llama-cpp.md) |
| `MUSER_REMOTE_CACHE_DIFF` | Emits a full KV-plane diff when a qualification cell diverges (`remote.rs:590`, `:1624`) | unset | [32](chapters/32-precision-across-the-handoff.md) |
| `MUSER_REMOTE_FIRST_DIVERGENCE` | Enables the first-divergence hunt (the layer-0 ladder chase of the wizard story; `remote.rs:319`, `:1921`) | unset | [32](chapters/32-precision-across-the-handoff.md) |
| `MUSER_REMOTE_CACHE_PROBE` | Layer-0/token-one KV probe used on divergence (`remote.rs:1852`) | unset | [32](chapters/32-precision-across-the-handoff.md) |
| `MUSER_CUDA_METAL_COMPAT_STRICT` (with the `MUSER_CUDA_CPU_ORDER_*` family) | **Comparator-side**: lives in the patched llama.cpp build (`scripts/gx10/llamacpp/muser_cuda_metal_compat.patch`), where it forces llama's CUDA/CPU-order arithmetic variants for cross-vendor parity comparisons — not read by Muser's own Rust | unset in the engine | [29](chapters/29-cuda-versus-metal.md), [32](chapters/32-precision-across-the-handoff.md) |

### Server runtime

| Flag | Meaning | Default | Chapter |
|---|---|---|---|
| `MUSER_HOME` | The server's home root: default model path (`model.rs:129`), TLS material root (`tls.rs:25`), session store layout (`session_store.rs:157`), dashboard/static root (`axum_httpd.rs:1362`) | falls back to `$HOME`-relative paths | [37](chapters/37-server-sessions-and-security.md) |
| `MUSER_HOST` / `MUSER_PORT` | Bind host and port, wired through clap's `env` on the server CLI (`cli.rs:226-232`, `:407-411`) | port `4949` | [37](chapters/37-server-sessions-and-security.md) |

### Omitted, and why

The raw inventory (`_research/env-flags-raw.md`) holds ~84 names; the
remainder are deliberately left out of this table:

- **`MUSER_CACHE_ABI`** is *not an env flag at the pin* — it is the const
  cache-ABI identity string `"muser-muse-glimmer-f16-logits-v2"` in
  `muser-kvpack/src/layout.rs:18` (the research map's §14 row for it is
  wrong; the string simply matched the `MUSER_*` grep).
- **`MUSER_MC_NSG` / `MUSER_MC_NR0`** are shader-side constants that must
  match the dispatch shape, documented at `multicol.rs:41-44` — not
  runtime flags.
- **`MUSER_MODEL` / `MUSER_MODEL_SHA256`** gate the release-real-model
  test identity (`chat_template.rs:241-246`) — test-only.
- **`MUSER_COMPARATOR_*`, `MUSER_BUILD_*`, `MUSER_NVFP4_QKV_*`,
  `MUSER_LLAMA_*` (stage/fixture vars), `MUSER_GX10_*` stage-dir vars** —
  fixture plumbing for the parity comparators, not operator surface.
- **`MUSER_ACCELERATOR_LEASE_FD`, `MUSER_ACCELERATOR_CELL`,
  `MUSER_DFLASH_ATTENTION_F32`, `MUSER_DFLASH_ROPE_NCO`,
  `MUSER_TTFT_CAPTURE_REUSE_PROMPT`, `MUSER_DEBUG_STOP_AFTER_LAYER`** —
  single-use diagnostic/test hooks whose consumers I could not confidently
  explain from code in the time budget; omitted rather than guessed.

---

*What comes next: the master bibliography — [Appendix D](bibliography.md).*
