# Muser — measured-numbers ledger for the book

Compiled 2026-08-27 from the muser repo at pinned commit `6d0807da`
(read-only). Every number carries its evidence tag: the doc section or the
receipt path under `muser-receipt://`. Where a number was
superseded, both the old and the current value are listed. Numbers marked
**[precedent-7B-ferrite]** are ancestor-lab context and are never Muser
results (see §4).

Canonical source hierarchy: `docs/goal-parity-ledger-2026-08.md` (the
append-only campaign ledger, 5,751 lines) > `docs/benchmarks.md` (public
summary of the ledger) > `docs/launch-claims.md` (what copy may say).
When this file and the ledger disagree, the ledger wins.

---

## 1. Canonical numbers table

### 1a. Hardware and comparator identities

| metric | value | scope | evidence tag | caveats |
|---|---|---|---|---|
| Decode Mac | Apple Silicon, **M3 Ultra, 96 GB** unified memory | all Mac-side measurements | memory-footprint.md §intro ("the 96 GB M3 Ultra"); release-provenance.md ("the designated 96 GiB M3 Ultra"); ledger 2026-08-23 readiness entry | The enclosure (Mac Studio etc.) is **never named** in these docs — do not invent it. "One Apple Silicon Mac" (benchmarks.md §Methodology). ~800 GB/s memory class (ledger L0). |
| Remote prefill node | one **ASUS GX10 (DGX Spark, NVIDIA GB10)**, `producer-1` | all disaggregated measurements | AGENTS.md; ledger 2026-08-23 topology amendment; benchmarks.md §Methodology | GB10 driver 580.173.02 (past the CX-7 throttle class); never a decode destination. |
| Lab link (current) | Mac `en0` Ethernet `192.0.2.10` ↔ GX10 `enp1s0f0np0` `192.0.2.20`, wired MikroTik 10GbE fabric | measurements after 2026-08-23 | AGENTS.md; ledger "GX10 topology migration" amendment; gx10-return-runbook §Correction | Mac Wi-Fi is `en1` and never carries a measurement. Historical (pre-08-23) numbers used the retired `retired /30` direct link (`en0`↔`enP7s7`). |
| Comparator | llama.cpp `89e0aa6fd362…`, `flash_attn_ext` prefill route, same pinned model; llama's own `draft-dflash` in spec lanes | every ratio on this page | benchmarks.md §Methodology; ledger Stage A entry gate | All ratios are **llama ÷ muser**; >1.0 means muser wins. |
| Target model | Muse Glimmer 30B: 52 layers, 39 SWA (window 2,048) + 13 NoPE, 2 KV heads × head_dim 128, max ctx 131,072; kquant GGUF 16,756,681,056 B, SHA-256 `7e9b74b7…` | all lanes | memory-footprint.md §Pinned geometry; ledger Stage A | SWA/NoPE split drives the KV arithmetic below. |

### 1b. Lane throughputs (local, Mac)

| metric | value | scope | evidence tag | caveats |
|---|---|---|---|---|
| kquant plain decode | **35.440 tok/s** (35.439527527, CV 0.037%) | 66-token prefix / 32 teacher-forced tokens, 5 reps, F16 KV, adjacent lease window | ledger P1.3 table; benchmarks.md §1 ("35.44") | kquant control cell. |
| NVFP4 plain decode | **35.491 tok/s** (35.490711722, CV 0.130%) | same cell as above, paired | ledger P1.3; nvfp4-fast-lane-evidence §Measured product numbers | **Parity within noise, never claimed faster** (+0.1444%). Unquantized F16 LM head costs ~3.46 ms/token vs kquant 1.75 ms — why the win is narrow. |
| DFlash speculative decode (the "107.9") | **107.9136 tok/s** median, CV 0.200% (llama 81.3047; ratio **1.3273**) | kquant lane, 2,048+256 streamed, verify-len 15, 5 reps, pre-window-fix binary | ledger L2 Stage B verdict | Measured **before** the 2026-08-21 draft-window fix; the fixed-window synthetic restatement is 1.236 at 2,048 (see next rows). The "107.9 tok/s bar" is quoted throughout later docs as the kquant spec bar. |
| Spec decode, fixed window, current | decode ratios **1.23692** @2,048, **1.20323** @16,384, **1.19616** @32,768 (5/5 exact reps each) | fixed synthetic fixtures, verify-len 15, funded-fix binary `419b670` (2,048) + respec2 binary (16k/32k) | launch-claims #15; ledger "Spec-prefill funded-fix requalification" + "Synthetic spec matrix deep-cell restatement"; receipts `spec-prefill-fix-20260822/aggregate-a2/…command.log`, `respec2-deep-20260822/aggregate-a1/…command.log` | **OPERATOR REVIEW REQUIRED.** Synthetic only; never generalize to natural text, NVFP4, or untested depths. The 8,192 and 65,536 cells are single-rep diagnostics (1.214†, 1.188†). |
| 131,008 wall parity | **1.02536×** end-to-end wall (prefill 1.02460×) | 131,008 prompt / 48 exact output tokens, 5 reps | launch-claims #16; benchmarks.md §2; receipt `spec-prefill-fix-20260822/aggregate-a2/…` | First 131k-class wall result above parity: 0.9768 → 0.98400 → 1.02536 across the fix lineage. **Do not cite the 1.64960× decode figure** — asymmetric first-round accounting (2026-08-23 ledger amendment). |
| Plain six-depth matrix (decode) | 1.0504 / 1.0429 / 1.0414 / 1.0479 / 1.0274 / 1.0277 at 2,048→131,008 | 5 exact-token reps per depth | benchmarks.md §1; ledger "Phase 2 non-spec context matrix" 2026-08-20; receipt root `ctx-matrix-plain-b972b55-20260819/` | Prefill means 1.0139–1.0397. Supersedes the one-sample 0.781× deficit of 2026-08-14. **OPERATOR REVIEW REQUIRED** wording. |
| Natural-text spec decode | wins python-like (16,384: 1.186; 8,192 suffix: 1.321), **loses** high-acceptance shallow text (rust 2,048: 0.931, →0.945 at vlen 7) | 128-output natural-text cells | benchmarks.md §2; ledger "Spec re-measurement at the fixed window" | Cross-engine outputs diverge on real text; speed stands without an exactness gate. Basis for frozen serving verify-length 7. |

### 1c. TTFT — disaggregated (GX10 NVFP4 → Mac)

| metric | value | scope | evidence tag | caveats |
|---|---|---|---|---|
| 2,048-token final-lane TTFT | **1.493 s** median, CV 0.22%, ≥**6.23 Gbps** installed payload, deterministic | final image `593b96a`, 2,048/256, 1 warmup + 5 counted | launch-claims #6; ledger T-series "Final packet on the final image"; receipt root `nvfp4-pacing8g-20260818/p4-wrapper23/` (verified: `installed_payload_gbps_min = 6.228`, `stable/deterministic: true`) | Counted-warmup convention is part of the claim. |
| 2,048 in the public matrix | 1.520 s vs local 6.48 s → **4.26×** | Phase-4 matrix, 5 reps, CV 0.49% | benchmarks.md §3; ledger "Phase 4 disaggregated GX10→Mac context matrix" | A different (earlier, 69e6037-era) packet than wrapper23 — same lane, different lineage. |
| Post-router re-qualification | 1.535889499 s median TTFT, CV 0.322%, payload 6.4592–7.2065 Gbps | after 2026-08-23 MikroTik migration, 2,048/256 P4 | ledger "Post-router GX10 lane requalification"; receipt `final-campaign-20260823/attempt-4/p4/P4_VERDICT.json` | Re-anchors the claim to the switched topology. |
| 130,815-token TTFT | remote **137.405 s** median, CV 0.576%, min payload **6.995 Gbps**, vs local 131,008 mean **570.122 s** → **4.149×** | EEE-off arm, same night/producer/fixture, 1 warmup + 5 counted | launch-claims #6; ledger "EEE A/B at 130815" 2026-08-21; receipts `kvpack-ladder-20260820/stage2-130815-rerun/` | Median-based; the earlier stall-contaminated Phase-4 cell reported 3.886× on means (superseded). Local baseline is 0.15% deeper (131,008 vs 130,815) — payoff conservative. |
| EEE-active arm (the intervention) | 138.886 s median, CV 2.0013%, per-rep payload [7.213, 7.445, 7.315, **1.728**, 7.275] Gbps | same session, only EEE differs | ledger "EEE A/B at 130815" | One rep lost a ~6.4 s retransmission ladder; `stable:false`. Attribution by intervention, not correlation. |
| Full depth matrix payoff | 4.26 / 4.08 / 3.87 / 3.75 / 3.86 / 3.89× at 2,048→130,815 (3.75–4.26× band) | Phase-4 matrix, 5 reps/depth | benchmarks.md §3; ledger Phase-4 entry; receipt root `phase4-disagg-20260820/` | Public docs quote the 3.75–4.26× band (CHANGELOG, disaggregated-prefill.md). |
| Cold integrated headline (F-series) | **3.881 s** cold disagg incl **1.87 s** native producer compute, vs ~**6.5 s** local serving prefill; paced wire 3.925 Gbps | 2,048-token accepted integrated cell, 2026-08-17 | nvfp4-fast-lane-evidence §Measured product numbers; ledger F-series operating amendment | Operator-accepted engineering headline, **not** a five-rep stability claim. The historical **5.83×** (exact Spark 46.8 s vs 275 s exact-Mac mirror) is **retired** — never use it (P4 note, launch-claims #6). |
| G2 cold/warm separation | cold first-request median **1.796 s** (CV 5.4%) vs warm **1.683 s** (CV 20.2%, published as median+range) | 5+5 controlled reps | nvfp4-fast-lane-evidence §G2; receipt root `nvfp4-g2-ttft-20260818/` | The earlier CV-21.40% packet was a mixed-state artifact. |
| Sustained deep load | eight consecutive 130,815-token handoffs, zero producer deaths, deterministic; payload 6.87→3.47 Gbps (last two = observed tail, all ≥3.0 floor) | bounded soak, gens 960245–960252 | benchmarks.md §3; ledger "eight-handoff deep soak" 2026-08-23; receipts `final-campaign-20260823/attempt-4/soak/run-attempt-3/SOAK_VERDICT.json` | Not a 20-rep W4 stability packet. One producer died on the 9th deep handoff during the EEE-off sequence (separate incident, claim #13). |

### 1d. Warm reuse and delta handoff (kvpack)

| metric | value | scope | evidence tag | caveats |
|---|---|---|---|---|
| Shallow warm hit | **64.631 ms** median, CV 0.424% | 2,048-token resident prefix, 5 fixed samples after 1 unmeasured warmup | ledger P4 cell; launch-claims #11 | The "≈65 ms" ladder rung 1. |
| Warm reuse at depth | first token **0.6132 s** @65,536 (cold 68.6166 s) and **1.0566 s** @130,815 (cold 147.8321 s); cold/warm text bit-identical; **no producer drive on warm hits** | isolated-depth cold/warm/miss legs, one sample per depth | ledger "Kvpack ladder stage-5 isolated-depth verdict"; launch-claims #11; receipts `kvpack-ladder-20260820/attempt-9-…-stage5-warmhit/` (verified: both depths `legs_valid/outputs_match: true`) | **Two depth-specific samples, not a distribution**; never "decode faster". Miss controls (8,192 unrelated) ~10.5–12.9 s prove it is reuse, not cache-forever. |
| Delta handoff (deep) | delta 517,983,232 B vs full 954,190,848 B = **54.2851%** of bytes; output SHA-256 **exactly equal** (`2526a55d…19778`) | 32,768-token held prefix, 65,536-token request | benchmarks.md §4; kvpack.md; ledger stage-6 verdict; receipt `kvpack-ladder-20260820/attempt-10-…-stage6-delta/stage6-delta-65536/stage6-verdict.json` (verified: `delta_share_of_full: 0.5428507652…`, `exact_against_full_handoff: true`) | **Suffix-only wire, not proven suffix-only compute** (launch-claims #12). |
| Delta handoff (shallow) | 49.98% of full bytes, bit-exact | 1,024-token held prefix, 2,048-token request | ledger T-series "Delta-only prefill (W3)"; receipt `nvfp4-pacing8g-20260818/delta-wrapper7/` | Exactly the suffix share at that geometry. |
| Producer prefix caching | cache-hit rerun payload digests bit-identical to fresh compute (all three handoffs); stays **opt-in** until a soak at final identity | `pcache2-a/b-20260819` cells | ledger T-series follow-ups; receipts `nvfp4-pacing8g-20260818/pcache-wrapper{A2,B2}/` | This is the only "prefix parse-cache"-class fact in these docs: producer-side vLLM prefix caching, qualified but not default. Resident radix identity binding/cut-capping landed in `4f86663`. |

### 1e. Wire rates and link facts

| metric | value | scope | evidence tag | caveats |
|---|---|---|---|---|
| Raw ceiling (historical) | **~9.40 Gbps** single-stream both directions, MTU 1500, zero tuning, zero retransmits over 30 s | pre-rebuild direct 10GbE `/30` | ledger T0; sealing-plan §W0; disaggregated-prefill.md ("~9.4 Gbps raw single-stream ceiling") | Must be re-proven after topology changes. |
| Raw ceiling (post-rebuild, switched) | GX10→Mac **9.256 Gbps** (also 9.218, 9.291 in adjacent probes); Mac→GX10 only **6.161/6.501/5.410 Gbps** — asymmetric, retained as deviation | MikroTik fabric, 2026-08-23 | ledger attempt-3/4 + readiness entries; receipts `final-campaign-20260823/attempt-{3,4}/phase0/tcp-*.json` | Reverse direction is **not** promoted to a pass; product direction is the healthy one. |
| Link merit gate | every counted rep ≥ **3.0 Gbps** installed-payload rate; TTFT CV ≤ 2% | p4 packet gates (post §7.4 ruling) | ledger "Link-gate re-spec executed"; benchmarks.md §Methodology | Rate CV stays in receipts for audit only. |
| Pacing ladder | 3.91 Gbps (500 MB/s pin) → **5.89 Gbps median** at the 8 Gbps pin (`SO_MAX_PACING_RATE`, fail-closed readback) | installed payload, 2,048-class cells | ledger T1; sealing-plan W1 | The 3.9 Gbps was never the hardware — it was the sender's own pin. |
| Wire clock | Linux `TCP_INFO.busy_time` is the only honest link denominator (userspace send-time and receiver first-read clocks both rejected) | production transport | ledger P4 "The original installed-payload row…"; N5 | The retired 5.581/4.550/5.309/5.769/4.765 Gbps row (CV 8.9985%) is failed-metric evidence. |
| Wizard qualification rates | **9.812 / 8.887 / 8.690 Gbps** installed payload across three handoffs | combined-lane enrollment, 2,048/256 | launch-claims #9; ledger attempt 31 table; receipt `wizard-validation-20260823/attempt-31-combined-full-20260824T132639Z/validation-summary.json` | Onboarding-recipe scope, not a serving-throughput claim. |

### 1f. Decode-dispatch-gap accounting (the +196)

| metric | value | scope | evidence tag | caveats |
|---|---|---|---|---|
| Closure counts | production **760** vs legacy **564** profiling closures, delta **+196** at position 2,048 | one-token Metal decode graphs, pinned 16,756,681,056-B target, 2,048-token fixture, one teacher token | decode-dispatch-gap-20260815.md §Corrected closure-count diff | Closures are Rust profiling closures (one command buffer + wait each), **not** raw Metal dispatches. The 760/564 comparison is not a kernel or host-encode measurement. |
| The reconciliation | **104** separated norm-boundary groups, **39** SWA wrapped-ring staging groups, **52** KV-publication/attention splits, **1** last-row copy | same | decode-dispatch-gap §label table; ledger Stage A entry | The 104-group fusion exists but is **numerically inexact** (rejected); the 52 splits are session structure, not waste. |
| The one exact removal | 1 closure + one 6,656-element f32 copy: GPU −0.136 ms (−0.34%) | single-run diagnostic | decode-dispatch-gap §Landed and rejected reductions | Wall +4.380 ms = submit/wait noise; no wall claim. |
| Rejected hybrid | full-logit max abs err 4.63e-4; normalized-logprob max **3.197e-4 > 1e-4 contract**; 201,970/202,048 logits differ; first KV diff layer 1, value element 524,115 (f16 bits 39,892 vs 39,893) | hybrid retained-activation schedule | decode-dispatch-gap §Rejected hybrid postmortem; receipts `pinned-token-parity-20260814-v{3,4}/` | Removed, not hidden behind a tolerance. |
| Baseline GPU time | 40.330 ms (760 groups); exact dual-norm fusion → 39.274 ms (655); one-query GQA FA2 → 37.097 ms | bounded phase diagnostic | decode-dispatch-gap reduction table | Post-A6 streamed decode 28.290 tok/s vs llama 33.428 (0.8463×) — Stage A still open at that point. |

### 1g. Distributed speculative decoding — measured and rejected

| metric | value | scope | evidence tag | caveats |
|---|---|---|---|---|
| All-accept control | **110.59 tok/s**, 477/477 proposals, 34/34 Mirror commits, 512 tokens / 35 rounds | linear GX-verifier lane, standard trace | nvfp4-distributed-speculative-frontier §End-to-end linear-lane verdict; launch-claims #14 | **Never cite as serving performance** — it is a positive control under forced acceptance. |
| Real acceptance | **9.23%** (docs) / **26.31%** (python) / **38.07%** (rust) | three organic 256-token strata | same | Same weights both engines (llama 65–81% on identical fixtures) — muser-side conditioning was the defect at the time; organic ceilings below bar regardless. |
| Verifier-only ceilings | **20.15 / 40.04 / 55.96 tok/s** (docs/python/rust) | output ÷ GX verifier wall, **zero** cost granted to Mac draft/transport/install/scheduling | same | The decisive rejection bound: below the 107.9 tok/s bar under physically impossible assumptions. |
| End-to-end measured | 15.532 / 11.172 / 15.412 tok/s (docs/python/rust) | same | same | Python/rust walls overlapped unrelated local work — point estimates only. |
| GX verifier screen | composite M16 + captures: 107.152 ms median target wall; projected **114.93 tok/s** at ≥99.151% IID per-edge acceptance | 31 warm prefix-cached runs | frontier §Decision | The projection set the preregistered bar the organic runs then failed. |
| Mac weight-only verifier no-go | best 227.864 ms GPU / 239.564 ms wall (13.9% over the 200 ms gate; 1.77× the 128.400 ms kquant reference); hard ceiling 70.2 verified rows/s | Fallback A, local 16-row verify | ledger "Fallback A follow-up — weight-only verifier final no-go" | Final decision: **Fallback B** — spec stays kquant-only at 107.9; native lane is plain decode. |
| Native spec no-go | **6.805 tok/s** (6.81) vs 107.9 bar; verify consumes 35.915 s of a 37.619 s decode span | NVFP4 W4A4 batched target execution | nvfp4-fast-lane-evidence; ledger F-series remediation | Why `producer_mode: native` fails closed on DFlash by construction. |

### 1h. EEE — the link ruling story

| metric | value | scope | evidence tag | caveats |
|---|---|---|---|---|
| N2 collapse | installed payload 0.062–5.526 Gbps across identical conditions (~90× wire-span variance); receiver verify+install+seal+commit constant ~0.2 s | strict 1×2048×256 cell, six reps | ledger N2 table; receipts `gx10-link-diagnostic-n2-*` | EEE-off probe reps: 6.215 / 5.208 Gbps. Ruled out receiver backpressure and producer compute. |
| Deep-payload blackouts | discrete retransmission ladders quantized at **6.42 ± 0.03 s** after 41–47 s of LPI idle; counted reps split 0.68–1.73 vs 7.20 Gbps | 130,815 payload (~1.74 GB NoPE burst) | ledger "EEE link ruling — operator decision (2026-08-20)"; receipt `phase4-disagg-20260820/130815-g900091/` | Ruling: **EEE-off enrolled as the link invariant**, ships as production guidance. |
| Why the burst pattern | SWA groups (~82 MB) stream early; the NoPE bulk (95.7% of payload) cannot start until CUDA finishes layer 51 — 41–47 s of forced idle then one burst onto an LPI link | architecture | kvpack-merge-handoff §6 "Pacing reality" | Coherent with the N2/EEE observations. |

### 1i. NVFP4 quality gates

| metric | value | scope | evidence tag | caveats |
|---|---|---|---|---|
| The docs exceedance | native vs kquant top-token **15.134%** vs calibrated gate **13.339%** (relative PPL +4.227%) at **65,536**, docs document | stage-3 yardstick | launch-claims #10; ledger stage-3 compact rerun table; receipts `kvpack-ladder-20260820/attempt-5-…-stage3-compact/stage3-e2-quality/stage3-yardstick-65536.json` | Published as a **content-local sensitivity**: not replicated cross-document, not persistent at 131,008 (docs corpus too short to test). kquant lane remains selectable as reference route. |
| Passing rows | 65,536: rust +1.079%/2.948% (gate 3.909%), python +1.356%/3.482% (gate 4.014%); 131,008: rust +1.250%/3.435% (3.905%), python +1.324%/2.985% (3.688%) | stage-3 | same tables | Aggregate verdict: **PASS as content-sensitive envelope**, not seal-eligible. |
| Route-exhaustion matrix | 8 native vLLM variants (chunked prefill, BF16, Triton, batch-invariant CUTLASS, FlashInfer B12X/cuDNN, engine ceiling) — all fail the docs 65536 row identically or worse | stage-3 isolation | ledger "native-route exhaustion"; matrix `…/overnight-20260822T022429Z/docs-65536-native-route-matrix.json` | No runtime change promoted; gates untouched. |
| E1 yardstick (quant-vs-quant) | Q6-vs-kquant long-context disagreement 6.251–12.259% sets calibrated gates 8.796–15.299%; native exceeds by 1.746–1.996 points at 8k/16k/32k | E-series | nvfp4-fast-lane-evidence §E1; ledger E1 | PPL passes every cell; the provisional 15% gate retired as universal. |
| E2 content control | only the docs document exceeds (8k/16k/32k); rust and python pass every length; all 15 PPL cells pass | three fixed nested documents | ledger E2/E3; receipt `nvfp4-e-series-20260817/e2/content-control-routing.json` | Disposition: **no context cap through 32k; publish the sensitivity**. |
| Semantic gate (D2) | agentic golden set **23/24 on both lanes**, identical per-category outcomes, identical single failure (lookup-004) | 24-task multi-turn tool set, 512-token budget | nvfp4-fast-lane-evidence §D2; ledger D2 | Token-level drift does not degrade task-level success. |
| Drift envelope (2,048/32) | 32/32 tokens identical; logit max/mean abs err 7.270581/1.040619; KV max deltas 9.625 (key) / 18.458 (value) | fast-vs-exact, standard fixture | nvfp4-fast-lane-evidence §Determinism | 2,048/256 five-rep comparator: 100% token agreement, logit 10.884401/1.233789. These bounded-logit deltas are the wizard's native-lane rule (max < 11, mean < 1.25). |
| Checkpoint bake-off | Inferact rejected (worse in all 30 E2 cells; confident flips 52/95/176 vs RedHat 23/33/56 at docs 8k/16k/32k; McNemar p ≤ 5.7e-05); prefill stays RedHatAI, decode stays Dudeman | full E2 sweep | ledger "Checkpoint bake-off"; receipt `nvfp4-bakeoff-20260817/checkpoint-decision.json` | i.i.d. Wilson gates overconfident up to 4.85× under position clustering; 83–88% of docs disagreements are near-tie flips. |

### 1j. ANE (Apple Neural Engine)

| metric | value | scope | evidence tag | caveats |
|---|---|---|---|---|
| ANE vs Metal | latest stable v9 3×256 result only **0.827× Metal** | public-CoreML DFlash drafting POC | launch-claims #5 | **No v0.1 launch claim.** Experimental/post-release, excluded from qualification, never selected by `auto`. |
| POC lineage | 0.644×, 0.704×, 0.711× Metal across v4–v6 splits; best 238.637 ms ANE vs 153.681 ms Metal | focused live POCs, 2026-08 era | release-provenance.md (dated research evidence) | Historical research evidence, explicitly overridden by the v0.1 scope notice (documentation-truth-pass). |

### 1k. Memory footprint facts

| metric | value | scope | evidence tag | caveats |
|---|---|---|---|---|
| KV bytes/row/layer | 2 KV heads × 128 × 2 B × (K+V) = **1,024 B** | f16 K and V, per layer-token | memory-footprint.md §KV formula | Topology-derived arithmetic, **not** a measured peak-RSS. |
| One-slot / four-slot KV @131,072 | **1.827 GB / 7.306 GB** (decimal GB) | release config = 4 slots | memory-footprint.md table | 8,192: 0.191/0.763 GB; 32,768: 0.518/2.072 GB. Summing artifacts + KV is only a lower bound. |
| Artifact sizes | target GGUF **16,756,681,056 B**; DFlash GGUF 1,631,205,312 B; vision projector 1,400,328,928 B | on-disk manifest | memory-footprint.md §Other material allocations | On-disk ≠ resident/wired; DFlash and vision load only when configured. |
| Prefill chunking | 512 positions; ~0.99 GB of f32 batch-activation widths (reused) | arithmetic | memory-footprint.md | "Must not be labeled peak RSS". |
| Correction | memory-footprint.md's "Metal KV buffers allocated without a CPU memset" is **wrong for live planes** — they zero-fill by design (`decode.rs:1299-1301`); only detached remote-install generations use uninitialized storage | 2026-08-20 audit | kvpack-merge-handoff §3 D2 | Cite the corrected fact, not the doc line. |
| Deep payload size | 130,815 cell moves **1,823,184,896 B** (reconciles to the byte: 130,814 × 13,312 B NoPE + 3 × 2,048 × 13,312 B SWA) | wire payload | kvpack-merge-handoff §3 D1; receipt `phase4-disagg-20260820/130815-g900091/out-p4/f-p4-text-g900091-client.json` (verified: `payload_bytes = 1823184896`) | The "~7 GB payload" figure that appeared in early docs is **wrong by ~4×** (7–8 GB is the producer's `--kv-cache-memory-bytes` allocation). |

### 1l. Wizard (one-button onboarding) validation

| metric | value | scope | evidence tag | caveats |
|---|---|---|---|---|
| Native/text PASS | attempt 9: all 7 labels; three 2,048/256 handoffs exact tokens (digest `42f09900…`), bounded logit deltas 10.884401/1.233788776 (<11/<1.25), payload 6.866/6.976/6.708 Gbps; `state=healthy` | 2026-08-24, fresh enrollment, whole-attempt lease | launch-claims #9; ledger attempt 9; receipt `wizard-validation-20260823/attempt-9-native-live-20260824T051305Z/validation-summary.json` | Native uses exact tokens + reviewed bounded-logit rule (not bit-identical logits). |
| Combined PASS | attempt 31: 7/7 stages; three handoffs with exact target tokens, exact full logits (max/mean delta 0), exact DFlash tokens/trace; **9.811736 / 8.886919 / 8.689889 Gbps**; `state=healthy`; canonical resident restored | 2026-08-24 | launch-claims #9; ledger attempts 10–31; receipt `wizard-validation-20260823/attempt-31-combined-full-20260824T132639Z/validation-summary.json` | Costed fix behind it: versioned cross-vendor arithmetic ABI (4–7 accelerator-hours). Combined wording still operator-review draft. |

---

## 2. Narrative arcs (chronological, 2026-08-14 → 2026-08-24)

**Arc 1 — The parity campaign: 0.781× to a six-depth matrix above parity.**
Question: can a from-scratch Rust/Metal engine match pinned llama.cpp bit-for-bit and
in throughput? On 08-14 a single-sample diagnostic showed 3.6% faster prefill but
decode at **0.7814×** (receipt `human-test-target-only-run-20260814-v1/target-comparator.json`;
documentation-truth-pass §Performance). The 08-15 A-series attacked the +196-closure
dispatch gap (see Arc 6) and topped out at 0.8463×; the J0 operator contract then
retired muser's self-referential hash and made **llama's own bytes** the gate
(`fc37487b…`, 808,192 B); J1 transplanted llama's attention DAG bit-exactly; J3
measured Stage A **met** at 1.05183× decode / 1.03703× prefill (5 reps). Stage B
spec initially failed at 0.8670× (K2: six levers rejected), then the L-series
microbenchmark-first n32 tile took the 16-row verify matmul from ~148 to ~83 ms/cycle
and the five-rep verdict to **1.3273×** (107.91 vs 81.30 tok/s). The production
Phase-2 matrix (08-20) closed it: 30/30 cells ≥1.0× on decode and prefill means
(`ctx-matrix-plain-b972b55-20260819/`). Verdict chain: Stage A met (J3), Stage B met
(L2), matrix passes — all synthetic, all exact-token-gated.

**Arc 2 — The pacing ladder and the EEE blackout.**
Question: why did a 10GbE link move only 3.9 Gbps, and why did deep-payload TTFT CV
explode? T0 proved the raw link sustains 9.40 Gbps with zero tuning — the 3.9 was the
sender's own 500 MB/s pacing pin (08-18). Raising it to 8 Gbps took installed payload
to 5.89 Gbps median. A separate bimodal ~1 s stall was root-caused to the **evidence
volume's directory-fsync tail** in the replay-ledger commit path — operational state
moved to the internal disk, TTFT median 1.596 s at CV 0.56% (the durability lesson now
in AGENTS.md). Then the deep 130,815 cells collapsed: the N-series and the 08-20 A/B
attributed it to **EEE LPI retrains** — discrete 6.42 ± 0.03 s retransmission ladders
after 41–47 s of forced link idle before the 1.74 GB NoPE burst. The 08-20 operator
ruling enrolled **EEE-off as a production link invariant**; the 08-21 A/B confirmed by
intervention (arm A median 138.886 s CV 2.0% with one 1.728 Gbps rep; arm B 137.405 s
CV 0.576%, floor 6.995 Gbps). Evidence: ledger N0–N5, T-series, "EEE link ruling",
"EEE A/B at 130815".

**Arc 3 — The kvpack ladder (stages 2–6).**
Question: does reuse actually deliver, with controls, on live hardware? The ladder ran
as ordered fail-closed stages and failed loudly three times before passing: stage 3
first refused on a vLLM context-cap override (132,032 > 131,072), then failed its
yardstick on the docs 65536 row, then — after restoring the preregistered
two-of-three-documents aggregate rule and a compact-retention repair (full-vocab
teacher rows cost ~12 GiB at 65k, ~25 GiB at 131k) — passed as a **content-sensitive
envelope** (docs 15.134% vs 13.339% published, not hidden). Stage 5's first 65,536 run
reported `outputs_match: false` — retracted as an infrastructure timeout (producer
killed at a 240 s default), after which the isolated-depth verdict passed (0.6132 s /
1.0566 s warm, bit-identical). Stage 6 proved the delta witness (54.2851%, exact
SHA-256). Stage 4 proved the safe producer swap and RUNG-1 control. Evidence: ledger
entries 2026-08-21/22 under `kvpack-ladder-20260820/attempt-{3,5,9,10,13}-…`.

**Arc 4 — The wizard: attempts 9 and 31.**
Question: can a stranger's GX10 become a qualified prefill node with one button?
Attempts 1–8 each failed closed on a real defect: stale 1,024-window geometry
residuals, a `flash_attn_ext` route overriding strict cross-vendor arithmetic
(`b972b55`, digest `a8d41633…`→`3d7ae82e…`), a stale RoPE-cache manifest, and a
wizard that computed its 3.0 Gbps gate from the wrong clock (0.67 Gbps reported
against a true 6.71 Gbps median). **Attempt 9 (native/text) passed 7/7** with exact
tokens under the bounded-logit rule. The combined lane needed attempts 10–31: a
layer-0 ladder chase (first mismatch `attn_norm-0` element 4 → K RoPE 256 →
`attn_out-0` element 4,096 = start of output row 2) isolated two arithmetic-ABI
splits — CUDA's serial 128-dim attention reduction vs Metal's 32-lane tree, and F32
vs F16 residual materialization — fixed in `27b5790`/`80f294f`; **attempt 31 passed
7/7** with bit-exact logits and 9.812/8.887/8.690 Gbps. Evidence: ledger §2b entries
2026-08-24; receipts under `wizard-validation-20260823/`.

**Arc 5 — Distributed speculation: rise and fall in one day.**
Question: can the GX10 verify speculative rounds remotely? The 08-18 frontier doc
first overturned an assumption (checkpoint unification is unnecessary — lossless
speculation needs one authoritative target endpoint, any approximation on the other).
A 107.152 ms composite M16 screen projected 114.93 tok/s **if** per-edge acceptance
≥99.151%. The end-to-end runs then rejected the lane: the all-accept control hit
110.59 tok/s, but organic acceptance collapsed to 9.23/26.31/38% and the
**verifier-only ceilings (20.15/40.04/55.96 tok/s)** — granting zero cost to drafting,
transport, installation, and scheduling — stayed below the 107.9 bar. Rejected for
serving; the only retained performance experiment is a hardware-aware token tree.
Evidence: `nvfp4-distributed-speculative-frontier-20260818.md` throughout.

**Arc 6 — The decode-dispatch-gap diagnosis.**
Question: where does the 22% decode deficit live — kernels, dispatch, or host? The
08-15 note first corrected its own instrument (PhaseProfiler counts closures, not
Metal dispatches; two label defects had shifted every legacy timing), then reconciled
the 760-vs-564 closure gap exactly into 104 norm-boundary groups + 39 SWA staging +
52 KV-publication splits + 1 copy — and found **no repeated closure doing identical
arithmetic**. The only exact removal (one copy) bought −0.136 ms GPU; every fusion
that would remove the 104 groups changed bits (hybrid postmortem: logprob error
3.197e-4 over the 1e-4 contract, first divergence one f16 ULP in layer-1 V). The gap
survived bit-exactness until J0/J1 changed the anchor itself. Evidence:
`decode-dispatch-gap-20260815.md`; ledger Stage A close-out (pinned-kernel audit at
`89e0aa6`: llama's `flash_attn_ext_vec` uses an intentionally different reduction
DAG).

**Arc 7 — The half-window draft (the bug that rewrote every spec number).**
Question: why did muser lose natural-text spec cells it should win? Three wrong
hypotheses were eliminated with counters (governor, draft context window, prompt
depth) before the 08-21 root cause: muser never read
`dflash.attention.sliding_window` (2,048) from the GGUF and hardcoded sink 64 +
window 1,024 — the draft ran on **half its trained window** for the entire campaign.
The fix (`a7a4d11`) took acceptance from 1.1% → 72.7% (python suffix 8192) and made
natural-text cells token-exact; it also **lowered** every synthetic spec number ~5%
(the draft now does real work). Corollary enrolled in the record: the period-8
synthetic fixture is predictable from token identity alone and **cannot detect this
class of defect** — natural-text cells are now a standing part of the matrix.
Evidence: ledger "ROOT CAUSE FOUND AND FIXED", "Spec re-measurement at the fixed
window", campaign-review-brief SUPERSEDED banner.

**Arc 8 — The funded prefill fix and 131k wall parity.**
Question: the two spec-mode prefill misses (2,048 at 0.9968, 131,008 at 0.975) —
real or apparatus? The 2,048 miss closed with no code change (measurement-order
carryover; controlled rerun 1.0017 with every rep ≥1.0007). The 131k miss was
decomposed by instrumented traces (capture readback ≈ +5.0 s; assistant K/V build
12.42 s — superseding four earlier attributions), the operator funded the fix
(`419b670`: GPU-resident capture, pipelined assistant K/V, prepare span booked as
prefill), and the 08-23 requalification crossed **wall parity at 131,008 for the
first time: 1.02536×** (0.9768 → 0.98400 → 1.02536). The 08-23 accounting amendment
then barred the tempting 1.64960× decode figure as an asymmetric cross-engine
measure. Evidence: ledger "Overnight matrix 2026-08-21" P2–P4, "Spec-prefill
close-out ruling", "funded-fix requalification", "AMENDMENT — 131008 decode
accounting audit".

---

## 3. The book's recurring question

The Ferrite book used *"where does the 8% gap live?"* as its spine. Candidates for
the Muser book, grounded in these docs:

1. **"Where does the dispatch gap live — once bit-exactness is non-negotiable?"**
   The +196-closure gap is the perfect recurring shape: it looks like waste, it
   reconciles exactly into named families, and then every cheap removal changes bits
   (hybrid: 3.197e-4 over a 1e-4 contract; A7/A8/A11–A15 all rejected on exactness or
   speed). The arc only resolves when the anchor itself changes (J0: llama's bytes
   become the gate; J1: adopt llama's DAG). It recurs later at 100× scale in the
   wizard's arithmetic-ABI chase (one f16 ULP in layer-1 V → 51.7 M differing
   logits). Justification: decode-dispatch-gap-20260815.md throughout; ledger Stage A
   close-out ("no untried llama scheduling transplant compatible with the fixed
   production hash"); ledger attempts 14–29.
2. **"What does precision cost — and where does the cost hide?"**
   NVFP4 decode is parity-within-noise (35.491 vs 35.440) and D2 agentic success is
   identical (23/24 both lanes), yet the same quantization costs 6.81 vs 107.9 tok/s
   in the W4A4 verify path and surfaces a 15.134%-vs-13.339% top-token exceedance on
   exactly one content class at one depth. The cost of precision in this program is
   never global — it hides in batch shapes and content classes, and the gates exist
   to localize it. Justification: nvfp4-fast-lane-evidence (P1.3, D1/E1–E3, D2);
   ledger stage-3 route-exhaustion matrix; launch-claims #10.
3. **"Why does disaggregation pay at depth — and what does the wire charge?"**
   Every depth pays 3.75–4.26×, but the honest ledger is a bill of charges: the wire
   floor (~3.75 s at 131k for 1.82 GB), the pacing self-cap (3.9 of 9.4 Gbps was our
   own pin), the fsync tail in our own commit path, EEE's blackouts on our own burst
   schedule — and then reuse collapses the bill (warm 1.06 s; delta 54.2851% of
   bytes). The question generalizes: the link is never the constraint until it
   suddenly is, and every stall in this campaign was self-inflicted infrastructure
   until proven otherwise. Justification: sealing-plan §4–§5; ledger T-series, N-series,
   EEE ruling; kvpack.md §reuse ladder; kvpack-economics.md.

(Honorable mention, methodological spine: *"what does the fixture hide?"* — the
period-8 synthetic stream certified a broken draft lane for an entire campaign,
ledger "ROOT CAUSE FOUND AND FIXED" consequence 2. Works as a chorus line rather
than the spine.)

---

## 4. Ferrite-lineage numbers (ancestor context — never Muser results)

These appear in the docs only as explicitly labeled precedent or as dated
pre-Muser research evidence. They describe the **historical Ferrite research
lineage on a 7B model**:

| number | what it is | where it appears | label |
|---|---|---|---|
| 34.9 / 308 t/s | Ferrite 7B decode/prefill-class throughputs | launch-claims.md §Ground rules (example list) | `[precedent-7B-ferrite]` |
| 21.9–30.1× restore | Ferrite KV restore speedups | launch-claims.md §Ground rules | `[precedent-7B-ferrite]` |
| 24.6 GB/s fabric | Ferrite inter-host fabric rate | launch-claims.md §Ground rules | `[precedent-7B-ferrite]` |
| 1.42× ANE+GPU concurrency | Ferrite on-device ANE concurrency aggregate | launch-claims.md §Ground rules; also appears in release-provenance.md as a dated on-device probe ("~97 GB/s concurrent ANE contribution, 1.42× aggregate over its synthetic Metal stream, 0.02–0.03% Metal tax") | `[precedent-7B-ferrite]` / dated research evidence — the ANE numbers in release-provenance.md are pre-v0.1 research, explicitly overridden by the v0.1 scope notice (documentation-truth-pass) |
| ~29.0–29.6 ms/token | Ferrite's rejected dirty QKVG/FFN dispatch-consolidation A/B records (2026-08-11) | extraction-manifest §Stage 2 close | ancestor-lab A/B, excluded from the release campaign |
| Qwen-specific FP4 evidence | "historical Spark producer assets … the historical FP4 evidence is Qwen-specific and is not Muse P0 evidence" | ledger P0 §Reused lineage | ancestor context only |
| `qwen25_logit_parity.py` | Ferrite comparator script that `scripts/evaluate_logits.py` reduces (at `51ad7e7e…`) | extraction-manifest §Stage 3 | apparatus lineage, not a result |

Note: **no "A18" number appears anywhere in these docs**; the only Qwen2.5-class
references are the two rows above. The Muser ANE figure that *is* this program's
result is the 0.827× of launch-claims #5 (§1j above).

---

## 5. Attribution facts (NOTICE + extraction-manifest)

What Muser adapted from the private Ferrite research tree — for correct book
attribution:

- **NOTICE (authoritative):** the GGUF parser, Muse configuration/loader, mmap weight
  access, quantized CPU math, BPE tokenizer, activation capture, and CPU reference
  graph were adapted from Ferrite at commit `83cfd55584dde68a9affca9c76af6a6124a3cf32`
  (Copyright Alvaro Videla, MIT). llama.cpp/ggml at `89e0aa6fd…` supplies `llama.metallib`.
  kvpack crates are coordinated `0.1.0-alpha.2`.
- **What was NOT taken (the deliberate rebuild):** Ferrite's entire `forward_gpu` VM
  engine (~80 files: `vm_*`, kernel selector, route receipts, koopman/geoprecision
  shadows) — the VM path was dead for Muse; muser's decode.rs/prefill.rs are
  reimplementations transcribed against the golden capture. Multi-arch dispatch,
  the IQ/MLX quant zoo, MoE, the DEAD registry of rejected approximations, and
  Ferrite's paged-KV telemetry seam were all left behind.
- **Shaders:** 15 files pulled byte-for-byte at Ferrite `a85048a90…` (per-file
  SHA-256 table in extraction-manifest §Stage 2), plus three adapted pulls —
  `flash_attn_v2.metal`, `batch_sgm_q4_aligned.metal`, `argmax_f32.metal` — each
  with source and muser SHA-256. The SWA **ring-address translation is
  Muser-owned** (Ferrite indexed by absolute position; the ring modulus was
  unwired/stubbed in Ferrite — a named OOB hazard muser fixed from day one).
- **DFlash GPU lineage:** the five-layer Metal forward transplants Ferrite's accepted
  DFlash lineage (`f332877600`, `84e4a8018`, `d26c51434`, `3063a4762`, `8e2b9cc4f`,
  `60f76e63f`), incl. `DFlashGpuLmHeadProjector` (`0b2f3d144`) and the two-phase GPU
  argmax (`be6ea89a4`, `56be14fa0`).
- **Comparator apparatus** (contracts, not runtime): `evaluate_logits.py` from
  `qwen25_logit_parity.py` (`51ad7e7e…`); the pre-quantization fixture patch from
  `llama-bench-fixtures.patch` (`17f9e96c…`); `llama_perplexity_evidence.py` from
  (`58ff9189…`); the append-only/no-retry accelerator discipline from the canonical
  Ferrite runbook (`3a846653…`). "No Ferrite crate, process, or library is linked
  into Muser or the comparator."
- **Approved wording (launch-claims #7):** "A standalone Muse-only engine assembled
  in days on our Ferrite research lineage." Never "one week from scratch", never
  omit the lineage attribution.

---

## 6. Claim-discipline crib sheet (10 rules for chapter authors)

1. **Ratios are llama ÷ muser.** Above 1.0 means muser wins. State it every time.
2. **Synthetic vs natural is load-bearing.** The 1.20–1.24× decode and the six-depth
   matrix are synthetic-fixture results; natural text diverges cross-engine and spec
   decode *loses* on high-acceptance shallow text (0.931 rust@2k). Never let a
   synthetic number become a workload claim.
3. **Five-rep means (or explicitly not).** Counted cells are five reps after one
   discarded warmup (disagg) or with 60 s cooldowns (spec matrices); single-rep
   diagnostics are marked † and one-sample cells are labeled as such (0.781×, 3.881 s).
4. **Never generalize beyond the tested range** — depths, lanes (kquant vs NVFP4),
   hardware, and content classes are all scope. The 2,048/130,815 depths of one
   claim are not the depths of another.
5. **OPERATOR REVIEW REQUIRED rows are not copy.** #2, #6, #11, #12, #15, #16, #17
   (and the combined-lane draft row) carry proposed wording only; the release lock
   is authoritative while in containment.
6. **Never cite the all-accept control as serving performance** (110.59 tok/s), nor
   the retired 5.83×, nor the superseded pre-window-fix spec numbers (1.3273/1.3012),
   nor the barred 1.64960× decode figure.
7. **Exactness is a gate, not a hope** — cells that fail token-exactness are not
   reported as passes; bit-exactness claims state which bytes (tokens, full logits,
   KV, traces) and under which anchor (llama bytes since J0; producer-vs-Mac since P0).
8. **Distinguish notarial vs non-notarial**: everything in this campaign is
   unsealed engineering evidence (`seal_eligible: false`); "measured" means measured
   on Muser, this program, under a retained receipt — never "measured once, on any
   hardware, ever."
9. **Publish the sensitivity, don't footnote it**: the docs@65,536 exceedance
   (15.134% vs 13.339%), the EEE-off dependency, the reverse-link asymmetry
   (6.161 Gbps), and the missing-depth cells are part of the claim, not noise.
10. **Evidence wins over wording.** When copy and receipt conflict, the receipt wins
    and the row gets corrected; a gate that rejects your evidence means the evidence
    is wrong until proven otherwise.

---

## 7. Landmines — numbers commonly mis-cited

- **110.59 tok/s** — all-accept *control*, never serving (frontier doc, claim #14).
- **5.83×** — retired exact-mirror comparison; the product claim is 3.881 s vs ~6.5 s
  or the 4.149× EEE-off median.
- **1.3273× / 1.3012× / 107.91 tok/s spec** — pre-window-fix; the current synthetic
  restatement is 1.23692 @2k (the 107.9 figure survives only as the kquant spec *bar*).
- **0.781×** — single sample, 2026-08-14, superseded by the six-depth matrix.
- **1.64960× decode @131k** — barred by the 2026-08-23 accounting amendment; use wall
  (1.02536×).
- **"~7 GB payload" @130,815** — wrong by ~4×; the wire payload is 1,823,184,896 B.
- **35.491 vs 35.440** — parity within noise; never "NVFP4 is faster."
- **The 65,536 warm-hit `outputs_match: false` cell** — retracted infrastructure
  timeout, not a cache-correctness failure; the valid cell is bit-identical.
- **"~64 ms warm hits" at depth** — 64.631 ms is the *shallow* (2,048) figure; deep
  warm hits are 0.613/1.057 s, and both are single samples, not distributions.
- **macOS "no CPU memset" of KV planes** — corrected: live planes zero-fill by design.
- **M3 Ultra, 96 GB** — the designated qualification host; the docs never name the
  enclosure, and no smaller-memory config may be advertised as supported.
- **9.4 Gbps symmetric** — the *pre-rebuild* reference; the switched fabric is
  asymmetric (9.256 one way, 6.161 the other) and must be re-proven before use.
