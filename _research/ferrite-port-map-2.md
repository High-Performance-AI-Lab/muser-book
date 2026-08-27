# Ferrite → Muser port map, part 2 — chapters 14–25, appendices, CORRECTIONS

Audit of the back half of the ancestor book
(`<ferrite-rs checkout>/inference-book/`, read-only) feeding the
Muser port. Ancestor: Qwen2.5-1.5B Q4_K_M, A18 Pro/8 GB `[A18-neo]`, 45.95
GB/s ceiling, 33.20 tok/s tg128, 0.92× vs llama.cpp. Muser (per
`muser-book/PINNED.md`, repo pin `6d0807da`): Muse Glimmer 52-layer ~30B; 39
SWA layers with a 2048-token KV ring + 13 full-attention NoPE layers; GQA
32Q:2KV; sigmoid attention gate; kquant + native NVFP4 weight lanes; DFlash
speculative decoding; durable kvpack KV reuse; disaggregated remote prefill
from a GB10 (vLLM NVFP4) over mTLS+HMAC Handoff V2; fail-closed evidence
culture (parity ledger, release lock).

Ground truth newly verified in the muser tree (drives the ch 21–22
dispositions): Muser has **no `Vec<Op>` VM program, no `Op` enum** — decode is
hand-written encode methods (`encode_token`, `encode_greedy_pipeline_command`
in `crates/muser-engine/src/decode.rs`) calling `kernels.encode_*` directly,
i.e. the shape of Ferrite's legacy `Qwen2LayerwiseNative` route, not its
`vm-sync` route. Buffers default **tracked** (`shared_tracked()` in
`crates/muser-engine/src/metal/buffer.rs` returns plain `StorageModeShared`;
the comment records that untracked was enabled globally in `b9678d4` and
reverted because the explicit-barrier contract didn't exist — it "empirically
changed DFlash conditioning"). Barriers are targeted
`memory_barrier_with_resources` calls (staged weights, KV handoff), with
`MUSER_SERIAL_PREFILL_DISPATCH` for encoder type. Ferrite's compiled-program +
frozen-barrier-plan machinery is **lineage/contrast** for Muser, not shared
design.

Dispositions: **PORT-ADAPT** (keep device, re-ground numbers/paths) ·
**PORT-REWRITE** (keep the question, rebuild the answer) · **KEEP-AS-LINEAGE**
(teach as labeled ancestor contrast) · **DROP**.

---

## Ch 14 — The paged-Q8 KV cache (`14-paged-kv-cache.md`, 1151 lines)

**Teaching goal.** The KV cache is the one decode structure that grows with
context; full cost arithmetic of storing it (f32→Q8_0), then the paged layout
(16-token blocks behind a block table) from allocator down to shader
addressing, honestly audited for footprint and bandwidth.

**Structure.** 14.1 What the KV cache is · 14.2 Why quantize (Q8_0, not f32) ·
14.3 The paged layout — the core idea · 14.4 The `BlockArena` struct · 14.5
Q8_0 block layout — 136 B/row, 2176 B/block · 14.6 The block table · 14.7 The
store kernel `mha_kv_store_q8_paged` · 14.8 The contiguous fallback —
`FERRITE_NO_PAGED_KV=1` and the precision swap it hides · 14.9 The live arena
owner `BucketArena` · 14.10 The store-into-attention fusion · 14.11 Where the
gap lives · 14.12 Tradeoffs.

**Pedagogical devices worth porting.**
- *Dual derivation of the footprint* (Figs 14.1/14.2): same total in block
  form (`28×2×128×2176`) and element form (`elems÷32×34`), cross-checked —
  numbers are re-derivable, not received. Muser's ring/full/kvpack footprint
  section needs exactly this.
- *Paging = OS virtual-memory analogy* (logical pages → physical frames via
  page table), with "indirection per block, not per token."
- *The slab diagram* (Fig 14.3): out-of-order physical blocks, per-sequence
  block tables, free list, prefix-sharing arrows — the chapter's best visual.
- *Worked addressing example* (Fig 14.6): `pos=20, bt_head=[0,2176,6528…]` →
  `blk=1, intra=4, offset=2720`, byte-ruler drawing.
- *"One entry per block → 16× smaller upload"* (8 KB vs 128 KB at 32 K ctx) —
  the quantitative payoff of coarse indirection.
- *The 2×2 layout×precision decomposition* (Fig 14.8) with multiplicative
  check (`1.0146×1.0932=1.1091 ✓`) — cleanest confound-splitter in the half;
  reusable for any Muser KV-format A/B.
- *"Bytes bound bandwidth, not time"* (§14.11): the ctx≈65 K crossover
  argument bounded as *untested hypothesis* against latency/occupancy cost — a
  measurement-epistemics device.
- *Logical-vs-actually-allocated footprint audit*: three planes (F16-contig +
  Q8-contig + Q8-paged = 924 MiB at ctx 16384 vs a fraction logical), hedged
  "confirmed dead weight / suspected."
- *Tradeoffs as named structural bets*, each with measured/arithmetic
  consequence and honest "cannot back with a quality number."

**Ferrite-specific to re-ground.** 28 layers × 2 KV-heads × head_dim 128; 117
MB f32 / 31 MB Q8_0 @ ctx 2048; GQA 12:2; `BLOCK_TOKENS=16`, 2176 B blocks,
136 B rows; paths (`kv/block_arena.rs`, `kv/global_arena.rs`,
`attention.metal`, legacy `paged_kv.rs`, `compile.rs::mha_stores_current_kv`);
the `FERRITE_NO_PAGED_KV`/`FORCE_DECODE_Q8`/`NO_DECODE_Q8` triad + Table 14.1;
`[A18-neo]` +2–5 % tg128 (0.5B) and `[M3U]` 2×2 numbers
(286.14/311.44/289.07/317.36 tok/s); ~994 MB weights; 7.6 KB/token store
arithmetic; llama.cpp `461e59fe2` KV cites.

**Port disposition. PORT-REWRITE (flagged hazard).** Muser's KV story must not
inherit paging-as-default: (a) 39 SWA layers live in a **2048-token ring** —
capacity bounded, "grows with context" false for 3/4 of layers; the
O(n²)-recompute motivation re-derives per layer class; (b) 13 full-attention
NoPE layers grow unboundedly — footprint devices port there with new geometry;
(c) **kvpack durable reuse** is a dimension Ferrite lacks (needs its own
section: disk footprint, janitor, restore correctness; subsumes §14.9 prefix
snapshots); (d) disaggregated prefill means KV also *arrives over the wire*
into ring/slabs. Port the devices; rebuild the data structure around ring vs
growing vs durable. The OS-paging analogy survives as contrast: ask why Muser
doesn't need paging (or whether the 13 full layers do).

**Segues.** Opens recalling Ch 7/8 (attention math, `Op::KvStore`
suppression); closes to Ch 15 — "the data structure attention reads."
Strongest: §14.8's correction narrative (addressing flag secretly swapping
precision) — port its *shape* as a kvpack-format A/B caution.

---

## Ch 15 — Attention: the paged-Q8 decode kernels (`15-attention.md`, 1352 lines)

**Teaching goal.** Attention from zero (dot product → √d → softmax → weighted
V) with a fully worked numeric example, then the decode-attention *family* —
the position ladder — with `AllHeadsQ8` as worked example and
`SplitKVecQ8`/`FaVecDecode` as the real majority paths.

**Structure.** 15.0 The position ladder · 15.1 What attention computes · 15.2
The math, built carefully (15.2.1–15.2.6) · 15.3 Worked example, head_dim=2,
three tokens · 15.4 Multi-head · 15.5 GQA 12:2 · 15.6 Kernel signature · 15.7
Phase A fused KV store · 15.8 Phase B score-buffer attention (B1/B2/B3) · 15.9
Paged addressing — picture and cost · 15.10 Rust dispatch · 15.11
Online-softmax (flash) variant · 15.12 Tradeoffs · 15.13 Where the gap lives ·
15.14 The fourth rung: `FaVecDecode` · 15.15 What's next.

**Pedagogical devices worth porting.**
- *"Attention is the one operation that reaches sideways"* — everything else
  is per-position. Anchors the whole second half.
- *Q/K/V as information-retrieval folklore* (query = what I seek; key =
  indexed fields; value = payload).
- *The head_dim=2 worked example* (Fig 15.2): Q=[1,0], three keys, V=2/3/4 →
  out 3.0000 with every intermediate shown (scores, √2 scale, max-subtract,
  exp, Z=2.4931). The most portable artifact in the half.
- *The max-subtraction two-pass diagram* (Fig 15.1) with concrete values
  (`exp(0.7071−5.8)=0.0061`).
- *The 4-SIMD-group reduction ladder* (Fig 15.4): simd_max per group →
  `sg_max[4]` → barrier → combine — teaches simd reductions once.
- *Per-past-token cost ledger*: ~272 B/token/KV-head; per layer `seq×544 B`;
  ~12 % extra DRAM at 8 K ctx — the "when does attention matter" arithmetic.
- *Current-token bypass* (`t==pos` reads f32, not slab) — latency argument on
  the softmax critical path, honestly `[unverified]`.
- *Score-buffer vs online-softmax tradeoff*: `scores[4096]` cap vs running
  max/sum/accumulator; the `corr=exp(old−new)` rescaling at kernel-line level.
- *The position ladder as first-class object* (Fig 15.0):
  `splitk-q8@0-7,allheads-q8@8-63,splitk-q8@64+` (+`fa-vec@8192+` M-class),
  ground-truthed with route logging, fingerprint strings quoted. Muser needs
  its own ladder (ring/full layers, flash variants, DFlash paths).
- *Two-dispatch split-K flash decode* (§15.14): `[max,sum,O]` partials +
  reduce kernel; "across threadgroups, not just SIMD groups" as a structural
  step. Maps onto Muser flash decode variants.
- *NLL/top-1 fork adjudication methodology*: self-determinism, fork counts,
  teacher-forced NLL tiers, cross-mechanism fork-position-overlap analysis
  (36.5–43.8× enrichment ⇒ benign near-zero-margin positions). Reusable for
  Muser kernel-swap correctness arguments.

**Ferrite-specific to re-ground.** 12 Q-heads/2 KV-heads/head_dim 128;
`attn_scale=1/√128≈0.0884`; GQA 6:1 with `kv_h=tg_id/hpkv`; grid
`(12,1,1)×(128,1,1)`; `window_size=0` (Qwen2.5 has *no* SWA); `scores[4096]`;
`FC_ONLINE_SOFTMAX`; ladder thresholds 8/64/8192; +53.7–64.6 % FA wins;
kernel/PSO names and file:line cites; llama `kernel_flash_attn_ext_vec_reduce`
port provenance.

**Port disposition. PORT-REWRITE with PORT-ADAPT core.** Math sections
(15.1–15.5, the reduction pattern, online softmax) PORT-ADAPT with Muse
geometry (GQA **32Q:2KV = 16:1**, 52 layers, pinned-GGUF head_dim) **plus a
new sigmoid-gate subsection** (output gating changes the closing formula;
extend the worked example to show it). Kernel anatomy PORT-REWRites: for 39
SWA layers `win_start`/ring wraparound mod 2048 becomes *central* (Ferrite's
`win_start=0` was a footnote); 13 NoPE layers skip RoPE (ties to ch 13);
flash-decode split-K carries §15.14's structure with new thresholds. The §15.0
ladder device is mandatory — "which kernel actually runs" is the chapter's own
headline lesson.

**Segues.** Opens in medias res with a correction (the ex-hero kernel covers
56 positions) — the strongest opening in the half; port the "first, which
kernel actually runs" pattern. Closes to O-proj (Ch 16).

---

## Ch 16 — Output projection + residual (`16-oproj-residual.md`, 541 lines)

**Teaching goal.** The O-proj is the matvec you already know plus a one-token
`+=`; teach residual-fusion byte arithmetic and why in-place mutation is safe.

**Structure.** 16.1 What it computes · 16.2 Why — residual stream, identity
gradient · 16.3 Same v4 you know · 16.4 The kernel · 16.5 Rust dispatch · 16.6
VM dispatch arm · 16.7 Byte arithmetic of the fusion · 16.8 The barrier
guarantee · 16.9 Tradeoffs · 16.10 Where the gap lives.

**Pedagogical devices worth porting.** *Unfused-vs-fused traffic ledger* (Fig
16.1: 4×6 KiB=24 KiB → 2×6 KiB=12 KiB, itemized reads/writes) — trivially
re-derivable for any Muser residual fusion. *"The `+=` is a
read-modify-write"* — naming the hazard class of a one-character change.
*Disjoint-lane ownership argument*: whole rows per SIMD group ⇒ no in-dispatch
race ⇒ no barrier, contrasted with the column-split sibling that needs one
("ownership implies synchronization freedom"). *Ping-pong double-buffering as
the CPU-bound-future alternative*, off-default with a measured reason (97.6 %
GPU-busy). *Weight-budget pie* (Fig 16.3: O 3.6 %, Q 3.6 %, K+V 1.2 %, FFN 65
%, embed+lm 26 %) — normalizing every kernel against the whole-model byte
budget is house style worth keeping.

**Ferrite-specific to re-ground.** `[1536×1536]` W_o; 1.27 MB/layer Q4_K; 6
KiB hidden; ~994 MB; 28 layers; kernel/PSO paths
(`matmul_q4k_v4_residuals.metal`, `OprojDispatch::Q4kV4Residual`); §16.8's
`FERRITE_UNTRACKED_HAZARDS` premise (Muser is tracked-default — re-ground).

**Port disposition. PORT-ADAPT.** Residual folding is universal; Muser's
O-proj consumes the sigmoid-gated output (32 heads, lane-dependent quant).
Byte ledger and ownership argument survive with new numbers; §16.8 rewrites
against Muser's tracked-default + targeted resource barriers (see ch 22).

**Segues.** Opens from `attn_out`; closes to the FFN — "the dominant weight
stream." §16.2's identity-gradient recap (`∂(x+f(x))/∂x = 1+f'(x)`) is a good
cross-ref device.

---

## Ch 17 — The SwiGLU FFN (`17-swiglu-ffn.md`, 1076 lines)

**Teaching goal.** FFN and gated activation from zero (ReLU FFN → why gating →
SwiGLU → SiLU), then the fused gate+up kernel — with the standing correction
that production runs the *unfused* `SplitQ4k` route.

**Structure.** 17.1 What the FFN does · 17.2 The plain FFN · 17.3 Why gating ·
17.4 SiLU · 17.5 SwiGLU wiring · 17.6 CPU reference · 17.7 The matrix
operation by hand · 17.8 Fused (`SeparateQ4k`) vs production (`SplitQ4k`) ·
17.9 Rust dispatch · 17.10 Access pattern · 17.11 4sg vs v4 read patterns ·
17.12 Activation at PSO build time · 17.13 Tradeoffs · 17.14 Where the gap
lives · 17.15 What's next.

**Pedagogical devices worth porting.** *Gate/up decoupling* — "should this
feature fire" vs "what does it carry"; the cleanest gating motivation.
*Activation table + ASCII character sketch* (silu/relu/gelu at x∈{0,1,−1,2};
"silu dips slightly negative," min ≈ −0.278 at x ≈ −1.28) — model-agnostic;
extend for Muse's actual activation. *The lockstep MAC diagram* (Fig 17.3):
`x[k]` read once → both accumulators; the canonical fusion picture. *Per-row
byte ledger* fused 7876 B vs unfused ~14036 B, refined: the *hard* DRAM win is
intermediate-buffer elimination (~140 KB/layer), not x-once (cache-absorbed) —
"count what actually hits DRAM." *Function-constant activation selection*
(`[[function_constant(32)]]`, dead-code-eliminated GELU branch) — Muser's lane
selection is the analogue topic. *"One threadgroup = one output row. Burn this
in."* *The production-route honesty pattern*: quote the beautiful fused kernel
in full, then teach that the default is the three-dispatch unfused path and
that the quoted arithmetic is "the savings production forgoes" — exactly the
Muser book's fail-closed voice.

**Ferrite-specific to re-ground.** 1536→8960; `W_gate`/`W_up` 7.74 MB each;
~25 MB/layer average FFN; ~544 µs/layer at 45.95 GB/s; `SplitQ4k` policy
plumbing; GDN `silu_gate` disambiguation; 77-vs-85 % framing.

**Port disposition. PORT-ADAPT (verify architecture first).** If Muse
Glimmer's FFN is SwiGLU (likely; confirm from the pinned GGUF), §17.2–17.7
port with new dims — at ~30 B the FFN's byte dominance is *stronger*, and the
Muser gap chapter should say so with Muser's own budget. Re-derive
fused-vs-production from Muser's actual route; don't inherit.

**Segues.** Opens with Ch 7's two-things-per-block; closes hanging the Q6_K
down-proj "prime suspect" hook. The suspect-then-vindicate chain 16→17→18 is a
strong serial device.

---

## Ch 18 — Down projection + residual (`18-downproj-residual.md`, 615 lines)

**Teaching goal.** Close the FFN; teach the mixed-precision decision (which
tensors get extra bits) as an explicit table, and the residual-fused Q6_K
family whose production member is a llama.cpp port.

**Structure.** 18.1 What it computes · 18.2 Closing the FFN block · 18.3 The
Q6_K wrinkle · 18.4 The residual-fused family · 18.5 Rust dispatch · 18.6
Standalone `residual_add` fallback · 18.7 Access pattern · 18.8 The Q4_K_M mix
table · 18.9 Tradeoffs · 18.10 Where the gap lives · 18.11 What's next.

**Pedagogical devices worth porting.** *The Q4_K_M mix table* (Table 18.2):
per-tensor format map, alternating 14 Q6_K + 14 Q4_K layer sets named by
index, rationale column — **the template for Muser's kquant-vs-NVFP4
lane-assignment table**. *Step-shown +46 % arithmetic*: 210 vs 144 B/256
elems, 53,760 super-blocks, 11.29 vs 7.74 MB. *"Why the down-proj gets the
bits"*: last-projection-before-residual error-propagation, flagged
`[unverified]` as model-specific — reusable honesty template. *269:1
weight:activation ratio* — one line teaching why decode matvecs are
weight-stream problems. *Port-attribution reading*: "which engine's kernel
wins is per-shape, per-format" (Q6_K: llama port ~40 % faster; Q4_K: native v4
6–10 % faster) — the transferable lesson.

**Ferrite-specific to re-ground.** Q6_K block layout (210 B: `ql[128]`,
`qh[64]`, `sc[16]`, f16 `d`); 24.5 % of decode time; layer sets
{0,1,5,6,7,8,9,10,13,16,19,21,24,27}; `bench_q6k_downproj_ab.rs` H2′
pre-registration; file:line cites.

**Port disposition. PORT-REWRITE.** Muser has no Q4_K_M mix; the chapter's
question becomes "which tensors/layers take the NVFP4 lane vs kquant, at what
byte and quality cost, and what does dual-lane dispatch complexity cost?" Keep
Table-18.2's format, the step-shown arithmetic, the residual material. The
llama-port contrast becomes Muser's llama.cpp-compatibility reference lane.

**Segues.** Opens with `ffn_mid`; closes "boxes ①–⑦ covered" into the tail.
"Where the bytes are, not where the gap is" is a reusable sentence shape.

---

## Ch 19 — Final norm + LM head (`19-final-norm-lm-head.md`, 881 lines)

**Teaching goal.** The tail: final RMSNorm (recap) and the vocab projection —
the largest tensor — plus latency-hiding geometry and dequant scale-scheduling
as a comparative table.

**Structure.** 19.1 What this chapter computes · 19.2 The final RMSNorm · 19.3
Why the LM head is special · 19.4 The matrix operation · 19.5 The fallback
kernel `matvec_q6k_f32_lmhead_4sg` (19.5.1–19.5.5) · 19.6 Dequant schedules ·
19.7 Grid × threadgroup · 19.8 Sibling kernels + latent-bug flag · 19.9 Rust
dispatch · 19.10 Logits output · 19.11 Vocab-blocked alternative · 19.12
Tradeoffs · 19.13 Where the gap lives.

**Pedagogical devices worth porting.** *"17× more rows than the next-largest
projection"* + "one-shot bandwidth hit, not a per-layer multiplier."
*In-flight-memory-requests / latency-hiding definition box* (§19.3) —
glossary-inline-box pattern at its best. *One-row-per-SIMD geometry diagram*
(Fig 19.2): 4 groups streaming 4 rows ⇒ 4 block reads in flight; cooperative
`tg_x[256]` tile reused ×4. *Scale-schedule comparison table* (Fig 19.3): Q4_K
deferred / pre-decode-all / Q6_K hoist — fetch, apply-point, registers,
inner-loop ops; Muser's kquant vs NVFP4 dequant deserves exactly this.
*Latent-bug flag* (§19.8): Q4_K wrapper whose dispatch contradicts its
kernel's SIMD assumption — documented "suspect until tested," fail-closed
bookkeeping to replicate. *Router-refuses-to-fallback* (`required_kernel_id`
panics) — small on-brand exhibit. *Fence-convention note* (§19.2: which
annotations are added vs verbatim) — STYLE-level device preventing quote-drift
accusations.

**Ferrite-specific to re-ground.** vocab 151936, hidden 1536; 233,373,696
elements; Q6_K 191 MB (vs Q4_K 131 MB hypothetical); untied embeddings; 594
KiB logits; 2026-07-11 llama-port default-flip, `FERRITE_NO_Q6K_LMHEAD_LLAMA`,
`FERRITE_Q6K_LMHEAD_IMPL` sweep; GGUF ground-truth cites.

**Port disposition. PORT-ADAPT.** Structure universal: final-norm recap +
giant vocab matvec + logits handoff. Re-ground to Muse vocab/head format from
the pinned GGUF. New Muser material: the LM head's role in **DFlash target
verification** — "runs once per token" becomes "once per target step plus
draft scoring," changing §19.10 and ch 20's sequel.

**Segues.** Opens "after block 27, the residual stream is the model's final
thought"; closes to argmax.

---

## Ch 20 — Argmax (`20-argmax.md`, 701 lines)

**Teaching goal.** Turn logits into one token id; two-phase tree reduction
under the 1024-thread limit; why greedy; why the GPU answer is a 4-byte read.

**Structure.** 20.1 What this chapter computes · 20.2 Why greedy · 20.3 The
reduction problem — two phases · 20.4 Tree reduction in one threadgroup · 20.5
Both kernels · 20.6 Grid, threadgroup, inter-phase barrier · 20.7 Rust
dispatch + `Op::Argmax` · 20.8 Reading the result — four bytes · 20.9 CPU
fallback · 20.10 The greedy entry · 20.11 Sampling beyond greedy · 20.12
Tradeoffs · 20.13 Where the gap lives.

**Pedagogical devices worth porting.** *"The winning index IS the next token
id — no softmax"* + argmax is softmax-invariant for greedy. *Tree-reduction
stride-halving diagram* (Fig 20.3) with a barrier per pass; *two-phase
partials diagram* with exact ragged-chunk arithmetic (⌈151936/1024⌉=149, last
chunk 384). *Deterministic tiebreak* (strictly-greater ⇒ lower index) tied to
the gate — small, load-bearing for byte-identical diffs. *The 4-byte-read
diagram* (Fig 20.5): 594 KiB stays on GPU, 4 bytes cross, with the 13.2
µs/token counterfactual priced at the ceiling. *CPU-argmax-as-oracle*: second
implementation kept as diffable correctness reference (Muser has
`reference.rs` — same discipline). *Task-brief-number corrections in captions*
("~148 partials"/"~148 KB scratch" walk-backs): the book correcting its own
commissioning briefs in public — Muser-book DNA.

**Ferrite-specific to re-ground.** vocab 151936 everywhere; 1024-thread cap;
~32 KB threadgroup budget `[unverified]`; `FERRITE_GREEDY_CPU_ARGMAX`;
`argmax_f32_phase1/2` paths; W2A from-partials fast path.

**Port disposition. PORT-ADAPT + required DFlash extension.** Reduction
pedagogy is hardware-universal (re-ground vocab; Muser has
`encode_argmax_f32_rows`-style variants). §20.2's greedy-determinism argument
must be re-grounded: under DFlash the per-step selection is
draft-verify-accept, and parity targets greedy-*equivalence* including
acceptance semantics; §20.11's sampling note upgrades toward a real section if
Muser samples.

**Segues.** Opens from logits; closes "the last kernel is exonerated" — the
per-kernel exoneration chain lands here and hands to orchestration. Keep the
chain; Muser's includes kvpack, handoff, DFlash links.

---

## Ch 21 — The VM-exec scheduler (`21-vm-exec-scheduler.md`, 1084 lines)

**Teaching goal.** Why Ferrite's decode is a compiled `Vec<Op>` program
replayed per token: do the routing reasoning once at load, freeze program +
barrier plan, replay.

**Structure.** 21.1 The problem a scheduler solves · 21.2 The `Op` enum · 21.3
A concrete slice — one layer · 21.4 The compile pipeline (a–e) · 21.5 The
runtime walk · 21.6 Why a "VM" · 21.7 The dry-run interpreter · 21.8 Dispatch
resolution + route label · 21.9 Tradeoffs · 21.10 Where the gap lives.

**Pedagogical devices worth porting.** *"Do the expensive reasoning once,
freeze, replay."* *Program-as-data benefits checklist* (§21.6): inspectable,
reorderable, offline-countable, dry-runnable, variant-switchable by pointer
swap — an evaluation rubric for *any* decode driver. *Five-stage pipeline
diagram* (emit → fuse-to-fixed-point → locality reorder → 64-node lookahead
barrier-reducing reorder → byte-range barrier plan). *pc-walk flowchart* with
dual-CB split, plan-bit barrier test, three-macro dispatch. *`pos=10`
representative resolution* with honest `[unverified]` on why 10 — Muser's
routing-freeze points deserve the same scrutiny. *Fingerprint-corrected Figure
21.1*: diagram matched against the actual gate fingerprint (`qkv=…+mixed(2/28)
ffn=SplitQ4k attn=splitk-q8@…`) including both corrections (FfnNorm distinct
under SplitQ4k; an Op ≠ one dispatch). *~83-vs-~197 vs measured 43*:
model/config-specific numbers must never be hardcoded.

**Ferrite-specific to re-ground.** Everything: ~370 dispatches/token, ~150 op
entries, `vm-sync` + 4 sibling routes, ~53-variant `Op` enum, frozen-replay
history (8× M3U win banked; Phase-1B divergence reproduced; no current
upside), 1.4 % encode, 43 barriers.

**Port disposition. KEEP-AS-LINEAGE (verified divergence).** Muser kept
neither the VM program nor the compiled barrier plan — decode is direct encode
calls (verified above), Ferrite's *legacy* shape. Teach the compiled-program
design as the ancestor's answer; explain Muser's choice (correctness surface;
DFlash conditioning sensitivity to execution regimes per the `b9678d4`
anecdote; where the CPU-encode budget sits on Muser hardware); port only the
evaluative devices (program-as-data checklist, fingerprint discipline,
CPU-encode-share measurement framing). Do not present the five-stage pipeline
as Muser's.

**Segues.** Opens from Ch 8's black box + the 370-dispatch headline; closes
measuring the orchestration layer under 2 % of the token. The "easiest thing
to misread" guard ports in spirit.

---

## Ch 22 — Barriers (`22-barriers.md`, 1099 lines)

**Teaching goal.** The hazard taxonomy, Metal's two barrier APIs, and
Ferrite's compiled per-model plan under untracked hazards — including retired
garbage-producing policy layers, the cross-CB fence lesson, and the 0.4
%-of-token measurement.

**Structure.** 22.1 The hazard problem · 22.2 The three hazard types · 22.3
Metal's two barrier APIs · 22.4 Production system overview · 22.5 Layer (a)
`DispatchBarrierTracker` · 22.6 Layer (b) `OverlapAnalyzer` + sealed proof ·
22.7 Retired layer (c) · 22.8 Layer (d) `VmDispatchMode` · 22.9 Layer (e)
emission + frozen replay · 22.10 The measurement · 22.11 The `None` policy
caution · 22.12 Untracked-hazards interaction + cross-CB gap · 22.13
Tradeoffs.

**Pedagogical devices worth porting.** *The barrier hazard taxonomy* (Fig
22.1): RAW/WAW/WAR + RAR non-hazard with order, rule, decode-frequency ("RAW
dominates — the decoder is a producer→consumer chain"); universal Metal
pedagogy. *Scope-vs-per-resource timeline diagram* (Fig 22.2): global drain
stalls unrelated reads; named resources let C-reads-Y overlap. *Sealed
`OverlapProof` type certification*: `DispatchGroup` constructible only by the
analyzer — "the proof is in the type, not a runtime assertion"; a Rust idiom
Muser could adopt for hazard-certified concurrent dispatch. *Byte-range vs
whole-buffer identity*: sub-allocation requires range precision; equivalence
test proving the analyzer a free upgrade. *The cross-CB lesson*: untracked
visibility does **not** span command buffers even same-queue in commit order
(Apple DTS-confirmed); encoder-scoped barriers die at `end_encoding()`; fix is
one `MTLFence` update/wait pair — **chained through a single fence object**
(the two-fence bug is the second lesson). *Correctness-before-perf cautionary
tale*: retired `None` policy's 99.6 tok/s = real speed + real garbage; TLA+
claim falsified by bisect; "no throughput number is valid without a
correctness gate for that exact route/config" — Muser's release-lock culture,
cite as convergent lineage. *Barrier-cost box*: 43×2.7 µs=0.4 %; PSO/binding
free; the 170-barrier serialization stress test, `[A18-neo]`-only (M3U
disagrees) — "stress-test the ceiling of your own overhead" ports to Muser's
barrier/fence inventory.

**Ferrite-specific to re-ground.** `FERRITE_UNTRACKED_HAZARDS` premise
(**false as a llama.cpp comparison** — CORRECTIONS §3a); `msg_send!
memoryBarrierWithScope:1u64`; the "five layers" framing (corrected: one live
decision source); 43/71 counts; frozen-replay tags.

**Port disposition. PORT-ADAPT (taxonomy, APIs, fence lesson) /
KEEP-AS-LINEAGE (compiled plan).** Verified: Muser runs tracked buffers +
targeted resource barriers, and its own comment records *why* untracked was
rejected. Muser's chapter: port §22.1–22.3 wholesale (it literally calls
`memory_barrier_with_resources`); teach the untracked+compiled-plan gamble as
lineage with its measured payoff (−2 to −4 % tracking cost `[A18-neo]`) and
failure modes; document Muser's tracked-default + barrier-site inventory and
the `b9678d4` reversal as the local decision record. Sealed-proof is a
keep-as-technique if Muser grows concurrent independent dispatch.

**Segues.** Opens from record-then-play + untracked hazards; closes "solved
sub-problem… the lever is weight-fetch bandwidth." The bolded one-sentence
chapter summary device is worth adopting.

---

## Ch 23 — Prefill vs decode (`23-prefill-vs-decode.md`, 627 lines)

**Teaching goal.** Name the two serving regimes and flip the roofline: decode
is memory-bound (each weight byte ~one MAC), prefill compute-bound (each byte
B MACs) — why the book is about decode.

**Structure.** 23.1 Two phases · 23.2 Arithmetic intensity and the roofline
flip · 23.3 The regimes side by side · 23.4 The prefill code path · 23.5
Decode recap · 23.6 Why prefill 0.99× but decode 0.92× · 23.7 The ggml
metallib bridge · 23.8 TTFT · 23.9 The prefix cache · 23.10 Tradeoffs · 23.11
Where the gap lives.

**Pedagogical devices worth porting.** *The roofline ASCII picture* (Fig 23.2)
with both workload points marked (decode ~3.6 vs prefill ~455 FLOPs/byte at
B=128; knee ≈50 on A18 Pro) and "intensity scales linearly with B" derived
per-row both ways — the half's best single image; recompute for Muser's chip +
NVFP4 prefill shapes. *"Read it twice"* signposting. *Per-regime
winning-kernel bet* — "use the best kernel for the regime," no purity; Muser's
local-Metal-decode vs remote-GB10-prefill is the same bet one level up.
*Last-token byte-offset trick* (`(b−1)×hidden×4` into the batch buffer; no
memcpy, no second CB). *TTFT decomposition* (cold/warm load, GGUF parse-cache
458→31 ms; pure prefill 366 ms) with the "load once, prefill every prompt"
pivot to prefix caching. *Prefix-cache warm/cold table* (8920 ms → 0; TTFT →
62 ms) with device-scope caveats. *Silent-fallback wart + `lib=` fingerprint
fix*: a bench timing the wrong kernel family because a file was missing —
mandatory reading for Muser's handoff-lane bookkeeping.

**Ferrite-specific to re-ground.** All ratios (pp 0.99×/0.76×, tg 0.92×,
`[A18-neo]`-scoped with the M3U opposite-shape callout); 45.95 GB/s; 864 B/row
intensity arithmetic; ggml bridge ~12 % SGM delta; Qwen3.5 prefix log; PF07
card.

**Port disposition. PORT-REWRITE (the task's named hazard).** Muser adds the
*disaggregated* dimension — a three-corner chapter: (1) local-compute roofline
(port Fig 23.2 with Muser numbers, kquant lanes); (2) remote-prefill economics
— GB10 NVFP4 throughput, wire rate (~9.4 Gbps reference on the wired MikroTik
path, re-prove before citing per AGENTS.md), handoff pacing ceilings,
mTLS+HMAC overhead — the bandwidth-bound axis is the *network*; (3) the
boundary: when remote prefill beats local (TTFT gains a network term; the
`handoff_report` per-rep phase table is Muser's native decomposition device).
Prefix caching splits into durable kvpack reuse (local) + received-KV handoff
(remote). Ferrite's TTFT/prefix material ports as the local-lane baseline; the
rest is new ground with existing receipts and the sealing-plan doc.

**Segues.** Opens from Ch 8/11; closes establishing "only the memory-bound
regime can show the bandwidth gap," handing to Ch 24/25. Preserve that logic
for whatever gap Muser's ledger chases.

---

## Ch 24 — Measuring against llama.cpp (`24-measuring-against-llama.md`, 852 lines)

**Teaching goal.** The discipline that makes every other number trustworthy:
absolute tok/s is noise; same-session interleaved ratios are the only
cross-engine statistic; best-of-N for ceilings; the gate before perf.

**Structure.** 24.1 Absolute tok/s is noise · 24.2 Interleaved A/B ratios ·
24.3 The ceiling · 24.4 Best-of-N vs median · 24.5 Never block the samples ·
24.6 The correctness gate · 24.7 The flock locks + lock-order law · 24.8 The
`[A18-neo]` label · 24.9 The GATE-LOG · 24.10 Variance in pictures · 24.11
POWER mode · 24.12 Tradeoffs · 24.13 Why this matters · 24.14 The gate's blind
spots · 24.15 What's next.

**Pedagogical devices worth porting.** *Noise taxonomy*
(contention/DVFS/thermal) with "you changed nothing, the number changed
anyway" scenarios; ~30 % session spread (75–99 band) as motive.
*Interleaved-vs-blocked diagram* + the real contamination incident (concurrent
cargo, asymmetric, best-of-N rescue) as narrative; common-mode noise cancels
in the ratio. *"Ratio of per-engine medians"* as the citable statistic (not
mean-of-pair-ratios), real three-pair table reproduced.
*Best-of-N-for-ceilings / median-for-ratios* with reasoning and the near-miss
kill-bar story. *Two-step gate flowchart* (correctness → unconditional bench →
mechanical citability stamp "Do not cite…"). *Lock-order law* ("GPU first,
build second — reverse deadlocked once") +
don't-wrap-the-gate-in-its-own-lock; Muser analogue is `accelerator_safe.py` +
`/tmp/ferrite.gpu.lock`, same incident-learned-law framing. *Device-scope
labels* (`[A18-neo]` as scope claim, never merged with the M4 queue) → Muser
hardware tag + `[GX10]` remote tag. *Gate blind-spot honesty* (horizon
artifacts: IDENTICAL@1×64 → DIVERGES@4×256; inert flags;
PASS-can't-prove-the-lever-engaged) + landed-vs-planned split. *POWER mode
tradeoff*: sampling depresses tg128 ~12 % while J/tok stays ±6 % — cite energy
and throughput from separate runs.

**Ferrite-specific to re-ground.** `neo_gate.sh`/`mac_gate.sh`, flock paths,
GATE-LOG format, `[M1]`/`[T3bK3]`/`[D3]` cites, 45.95 GB/s protocol, J/tok
values, the fixed gate prompt, first-word-match rule.

**Port disposition. PORT-REWRITE (methodology swap wholesale — the task's
call).** Devices all port; every apparatus reference swaps to Muser's culture:
the parity ledger (`docs/goal-parity-ledger-2026-08.md`, append-only verdicts
with evidence paths), release lock + feature contract (no
seals/tags/candidates in containment; `release/findings-v1.json`), append-only
receipts under `muser-receipt://`, `accelerator_safe.py` gating
(dry-run default, gpu lock), the llama.cpp compatibility pin `89e0aa6f…`,
launch-claims discipline (`[precedent-7B-ferrite]`-style labeling of ancestor
numbers). New Muser protocol content Ferrite never had: network-lane hygiene
(wired en0 only, never WiFi en1; re-prove raw ceilings before citing ~9.4
Gbps), handoff-receipt retention (`handoff_report` as per-rep audit trail),
operational-state-on-internal-disk rule (the 2026-08-18 bimodal-stall lesson).
The parity gate's actual checks replace first-word-match — and document *its*
blind spots with §24.14's honesty.

**Segues.** Opens as "the long version of Ch 1's walk-back"; closes handing Ch
25 the protocol it depends on ("without interleaved ratios, 'the ratio did not
move' is meaningless"). The measurement→falsification dependency chain is the
half's spine — keep it explicit.

---

## Ch 25 — The gap, dissected (`25-gap-analysis.md`, 1139 lines)

**Teaching goal.** The culmination: run the falsification ledger on the ~8 %
deficit — six hypotheses killed, the survivor exposed as a tautology, the open
question kept open, and a second device tier's different answer kept
rigorously separate.

**Structure.** 25.1 The question restated · 25.2 The gap, measured · 25.3 The
method: falsification · 25.4 The ledger (H1–H7) · 25.5 Per-token breakdown ·
25.6 The DRAM ceiling · 25.7 The bandwidth comparison · 25.8 What we do *not*
yet know · 25.9 Path forward: attempted, blocked, routed around · 25.10 The M3
Ultra follow-up (25.10.1–25.10.6: per-op table, `FERRITE_QKV_SEPARATE` causal
test, submission overhead, full stack, guardrails, FA-decode) · 25.11 The
methodological lesson · 25.12 Two fronts · 25.13 Closing.

**Pedagogical devices worth porting.** *Falsification decision tree + ledger
table* (hypothesis/verdict/evidence/source, tombstones included) — the genre
Muser's parity ledger already speaks; the chapter teaches how to read and
write one. *H7-tautology exposure*: "achieved bandwidth = throughput × bytes"
— demote to description, keep the localization; the most valuable epistemic
device for any Muser "X % of ceiling" claim. *"Turn the method on the
ledger"*: it had *no row* for attention/KV, the LM head (17.5 % of bytes),
DVFS, the two-stream gate-up kernel — "its silences may be exactly where the
answer is hiding"; run the same audit on Muser's ledger (rows for handoff
phases? DFlash rejection cost? kvpack restore?). *Per-token breakdown bar*
(97.6 % GPU / 1.4 % CPU / 0.4 % barriers) with the 99.5 %-sum slack disclosed.
*Single-stream saturation table* (dual +0.43 %, quad −5.05 %) — "no 'just be
more parallel' escape hatch." *Causal-test template*: ranking →
predicted-mechanism flag → share collapses as predicted → throughput win +
byte-identical output. *Guardrails-as-loud-as-wins* (default-off levers,
provisional certification). *Two-fronts distinction* (bandwidth vs capacity) →
Muser's trio: local decode bandwidth, handoff wire economics, DFlash
acceptance rate. *Two-endings structure* (A18 open vs M3U answered, "kept
apart on purpose") + the meta-close (the method vindicated by exposing its own
incompleteness).

**Ferrite-specific to re-ground.** Every number: 0.90–0.93× band; 77 %/85 % of
46.1 GB/s; 35.4/39.4 GB/s; 994 MB; 166-buffer −6.7 % retraction; M3U stack
(1.133×/1.199×/1.120×; vs llama ~1.24×/1.14×/1.03×); FA-decode tables
(+53.7…+64.6 %; vs `-fa 1` 0.915×/0.827×/0.736×); all log tags.

**Port disposition. PORT-REWRITE.** Muser's gap story is different: parity is
Muse Glimmer vs the llama.cpp reference pin, with different levers (dual
weight lanes, ring-KV attention share at depth, DFlash acceptance economics,
handoff-phase overhead, kvpack restore cost). Port the genre and devices
wholesale; rebuild every row from Muser's ledger entries with evidence paths
under `muser-receipt://`. The ledger-blindness audit becomes a
first-class section against Muser's actual ledger.

**Segues.** Opens with Ch 1's hanging question ("where does that 8 % live?");
closes the book with the two-endings honesty and the method-as-receipt coda.
The culmination position (prerequisites: every chapter before) is the
structural device — Muser's equivalent collects its per-chapter "where the gap
lives" notes the same way.

---

## CORRECTIONS-2026-07.md — full correction register

The ancestor corrections file (DOC-01, 2026-07-10) is a
claim→replacement→evidence map over audit findings F1–F23 plus
session-verified facts; most rows were applied via DOC-02..DOC-06 and the FA
addendum (the chapters read above already carry the applied text). **The port
must not re-inherit any pre-correction framing.** Entries, one line each:

**Book-wide / meta**
- §1 — SUMMARY.md marked all 25 chapters "status: draft" while every chapter
  header says "polished"; needs one coordinated sweep (left undone by DOC-04).
- §0 — live-repo-state: DOC-09 (fabricated source-comment claims) and cull-6
  (orphaned files) already landed; DOC-02 landed mid-map; a second copy of the
  fabricated untracked-hazards language survives in `buffer.rs:158-161`; the
  `19a5037` hash cited for the LM-head barrier bug no longer resolves (cite
  the handout section).

**§3 — six facts verified beyond F1–F23**
- §3(a) — "llama.cpp uses untracked hazard tracking by default" is FALSE (zero
  `HazardTracking` hits; llama is Metal-default tracked); the false claim
  root-caused the real N_CB=2 cross-CB visibility bug.
- §3(b) — AllHeadsQ8 is *not* "the default decode attention kernel": ladder
  `splitk-q8@0-7,allheads-q8@8-63,splitk-q8@64+` (AllHeadsQ8 = 56 positions);
  the old fingerprint sampled pos=0 and was structurally blind; fixed format
  brackets 0/1/2/7/8/63/64/127/128/512.
- §3(c) — `FERRITE_NO_PAGED_KV=1` is a Q8_0→F16 **precision swap**, not pure
  addressing (2×2: ≈80–85 % precision / ≈15–20 % layout, multiplicative);
  store bytes bit-identical when precision held; divergence = reduction
  granularity past pos≥64 (2/12) plus dominant early quantization fork (11/12,
  median ≈24); the 2:1 F16 reference-match lean (n=3) is "directionally
  supported, not conclusively proven."
- §3(d) — the `paged_kv.rs:12` "commit 20812393 (Bishop paged KV benchmark)"
  citation was fabricated (that commit is a bugfix); removed by DOC-09.
- §3(e) — the residency "code None vs doc ON (335 allocations)" contradiction
  is CLOSED: the field populates in `warmup()`; `residency_installed()` reads
  real post-warmup state.
- §3(f) — restates §1 (polished/draft).

**DOC-02 — Ch 3 (applied)**
- D02-1 — arena-view "isn't this slow?" CLEARED: `set_buffer` views =
  llama.cpp's own model; inside the 1.4 % encode share.
- D02-2 — untracked-hazards "validated by Bishop" story fabricated; design
  itself fine; measured cost of tracking ON is −2 to −4 % `[A18-neo]`.
- D02-2b — the llama comparison was inverted: llama is tracked by default;
  Ferrite's untracked mode is deliberate divergence, not industry norm.
- D02-3 — B4 zero-init: mechanics benign, but "96 of 128 lanes read garbage"
  describes a kernel-contract bug zero-init hides; `zero_raw_gpu` routes
  around a CPU-memset-flakes-~40 % sync bug instead of root-causing it.
- D02-4 — "Bishop" markers fabricated; rule: no prose claim cites an advisor
  persona or a number without a GATE-LOG/perf-log pointer.
- D02-5 — contradictory barrier-count literals (71/43/142/~83/~197/193) were
  unlabeled (model,device,era)-specific; replace hardcodes with the runtime
  diagnostic pointer.
- D02-6 — mmap page-align "hackish" CLEARED: the only way
  `newBufferWithBytesNoCopy` works; llama does the same.
- D02-7 — GpuHeap judgment inverted: the heap path is the principled one; the
  broken `FERRITE_PACKED_ACTIVATIONS` (wrong-offset bug) stayed alive.
- D02-8 — `buf_id` sound-but-coarse; latent landmine: two MTLBuffers over
  overlapping mmap page ranges get different ids while aliasing the same bytes
  (live the day anything writable is wrapped no-copy).

**DOC-03 — Ch 14 + appendix row 11 (applied)**
- D03-1 — llama-bench's KV is real (autoregressive decode, F16 contiguous,
  `kernel_flash_attn_ext_vec`), not synthetic — but the cross-engine KV
  comparison is not apples-to-apples and nobody has measured ferrite-vs-llama
  attention time on any device.
- D03-2 — the "KV isn't the gap" byte-count argument is dubious: bytes bound
  bandwidth, not time; short-ctx attention is latency/occupancy-bound.
- D03-3 — the addressing-only framing of `FERRITE_NO_PAGED_KV` is FALSE (§3c);
  chapter now carries mechanism, Table 14.1 (flag alone reaches the F16
  kernel), the 2×2 figure, the reduction-granularity caveat, quoted verdict
  language.
- D03-4 — "best-correct route" downgraded to an open precision-for-speed
  tradeoff pending a real decode-path NLL number.

**DOC-04 — Ch 15/21/22 + SUMMARY + glossary + env-flags (applied)**
- D04-1 — SUMMARY path line presented allheads-4sg as *the* attention path.
- D04-2 — Ch 15's title/framing presented one kernel as the subject
  unconditionally; SplitKVecQ8 is the long-run majority path.
- D04-3 — "production route skips `Op::KvStore`" conflated the config boolean
  with the dispatched kernel; suppression is compile-time keyed on the
  AllHeadsQ8 plan (`SplitKVecQ8` not on `mha_stores_current_kv`'s list).
- D04-4 — "the VM smells" PARTLY: design defensible (llama re-encodes per
  token, 1.4 % encode); real smells are 9+ decode routes and a ~53-variant Op
  enum; residency contradiction closed per §3(e).
- D04-5 — Nop/Barrier/ConditionalSkip "super dubious" → runtime-harmless IR
  debt, cleanup territory.
- D04-6 — frozen replay: default OFF (Phase-1B invalidation incomplete, "masks
  decode correctness issues"); A18 caps at ≤1.4 %; the historical M3U 8× win
  is banked by the unretained-CB baseline; Qwen2.5 re-test reproduced the
  divergence bug with no win (+0.9 %/−0.2 %).
- D04-7 — Figure 21.1 showed `Mha{AllHeadsQ8}` unconditionally; needs the pos
  window.
- D04-8 — Figure 21.1 showed the fused FFN dispatch; resolved default is
  `ffn=SplitQ4k`.
- D04-9 — "× 28 layers" omitted the `qkv …+mixed(2/28)` per-layer
  non-uniformity.
- D04-10 — frozen-replay's second occurrence (Ch 22) gets the same single
  answer.
- D04-11 — "so many barriers / TLA+ vestigial / no per-model graph": the
  per-model compiled plan exists; 43/token is not "so many"; the retired
  category masks and non-live runtime tracker were the vestiges; one live
  decision source.
- D04-12 — "170-barrier serialization = same speed as 43" is `[A18-neo]`-only,
  untested on M3U where overlap matters (None policy +34 % garbage; 8× frozen
  replay); scope every "X is not the gap" claim to the device.
- D04-13 — "TLA+ should be per model": cleared in the specific (the plan *is*
  per-model), confirmed in the general (global correctness knobs are at the
  wrong altitude).
- D04-14 — glossary `allheads-4sg` entry needed the pos∈[8,64) window.
- D04-15 — `FERRITE_MHA_IMPL` row: the dispatched kernel is position-gated and
  override-collapsed, not merely "falls back if explicitly set."
- D04-16 — `FERRITE_QKV_FUSED` row: FAILs as **inert on 1.5B** (the only model
  gate-tested on M3U); do not extrapolate to "0.5B only" or "all three
  models."

**DOC-06 — Ch 2/4/13 (applied)**
- D06-1 — string-keyed PSO lookup: CLEARED on perf (inside 1.4 % encode;
  llama's is worse and still fast), CONFIRMED as design smell (two registries;
  `pipeline(name)` panics on miss) — with a correction-to-the-correction: the
  map's mechanism claim was backwards (`matvec_q4k_v4` *is* HashMap-looked-up;
  the `Option<ComputePipelineState>` fields are for optional kernels).
- D06-2 — "512 idle threads" guard-exit cost: CLEARED, effectively nothing.
- D06-3 — ggml-kernel reproducibility hole CONFIRMED: silent metallib fallback
  can time the ~12 %-slower native path unknowingly; fixed by the
  `[fingerprint] lib=` field + inert-flag guard.
- D06-4 — `Op::RopeEncode` verify-only: zero construction sites anywhere
  (dead-but-wired); Ch 13's text already kernel-level-correct (grep recount: 8
  lines, not 6).
- D06-5 — Ch 13's literal `!!!…!!!` is quoted historical prose, excluded from
  marker-count validation; plus the per_layer/single_cb attribution was scoped
  to the `Qwen2LayerwiseNative` route, and the audit's own
  `q4k_fused.rs:155-192` cite pointed at non-default code.

**DOC-05 — Ch 23/24/25 (applied)**
- D05-1 — "why not port llama's kernels": ports tried piecemeal kept tying or
  losing (block attention −4.9…−22.8 %, killed); the *composition* was never
  ported; the irony — Ferrite ships llama's prefill GEMMs (pp 0.99×) while
  decode is 100 % Ferrite (0.90–0.92×).
- D05-2 — Ch 23's 0.99×/0.92× figures carried no device tag; `[A18-neo]`
  added; M3U's shape is opposite (0.5B 1.097×, 1.5B 0.960×, 7B 0.926×).
- D05-3 — the correctness gate is weaker than the weight put on it: first-word
  match + no-6-identical-chars at 1 prompt × 64 tokens cannot see post-token-1
  drift, inert flags, or unengaged levers (N_CB=2+NO_PAGED_KV IDENTICAL@1×64 →
  DIVERGES@4×256).
- D05-4 — state landed vs planned accurately: fingerprint + inert-flag guard
  ARE live; the fixed-prompt NLL/top-k harness (Q1–Q9) is NOT.
- D05-5 — Ch 25's "root cause: GPU DRAM bandwidth 77 % vs 85 %" is a
  **tautology** (F23: throughput × bytes restated); keep as description; the
  ledger had no row for attention/KV, the LM head, DVFS, or the gate-up
  two-stream kernel.
- D05-6 — the xctrace experiment was attempted and blocked for specific
  non-privilege reasons (names: 396/396 "unknown," MTSP Ctt records for 4/16
  kernels; timing: `metal-gpu-intervals` attributes 0/2292 rows to ferrite — a
  compositor-scoped table; not a SIP wall).
- D05-7 — add the working alternative: in-process per-op attribution names
  `qkv+rope` (+7.8 pp on 7B); `FERRITE_QKV_SEPARATE` causal test +6.6 % (7B) /
  +4.4–5.3 % (1.5B), byte-identical, share 31.7 %→18.0 %; M3U full stack beats
  llama at every size/depth tested — with the single-prompt-gate correctness
  caveat.
- D05-8 — every absolute number in the gap chapter must carry
  `[A18-neo]`/`[M3U]`.

**Appendices + FA addendum**
- DB-1 — appendix-kernel-table row 5 (Mha) needed `pos∈[8,64)` scoping +
  ladder note.
- App D — SUMMARY linked a `bibliography/` directory that never existed;
  replaced by flat `bibliography.md` collected by grepping all chapters' arXiv
  cites.
- FA-1 — the three-rung ladder was outdated: fourth rung `FaVecDecode`
  (`FERRITE_FA_DECODE`, M-class default-on at pos≥8192, opt-out
  `FERRITE_NO_FA_DECODE`) landed; Fig 15.0 rewritten, §15.14 added.
- FA-2 — "beats llama at every model size and every depth tested"
  scope-narrowed: true for prefill depth ≤8192 under the three-lever stack,
  but ferrite trails `llama.cpp -fa 1` at d≥8192 by a *widening* margin
  (0.915×→0.827×→0.736×); §25.10.6 added; closing count updated. (Flagged,
  uncorrected: stale "pending sign-off" comments in `resolve.rs`/`routing.rs`
  beneath the landed default-on logic.)

**Port rule:** never port a claim without its correction context — the
pre-2026-07-10 framing of Ch 14/15/21/22/23/25 (hero kernel, pure-addressing
flag, five live barrier layers, untagged ratios, bandwidth-as-root-cause) is
exactly what the Muser book must not inherit even as analogy.

---

## Appendix format audit (for direct reuse)

**Appendix B — kernel dispatch table.** Columns, in order: `#` · `Op (VM IR)`
· `Metal kernel` · `Shader file:line` · `Rust encode_*` · `Grid × Threads` ·
`Reads → Writes`. Blocks: *Per-layer (repeated × N)*, *Tail (after layer
N−1)*, *Barrier mechanisms (cross-cutting)* (layer/type/lives-at), plus a
device-tagged *Production barrier facts* bullet box. Header pins the route
("the exact Metal kernel sequence for one decode token on … Verified from
source at the time of writing"). Conditionally-dispatched kernels get an
inline note block, not a silent single row (the DB-1 pattern). **Muser
reuse:** seven columns work verbatim; add a lane column (or per-lane tables)
for kquant vs NVFP4 vs DFlash draft/target, a ring/full marker on attention
rows; cite `file:line` at the muser-book pin per PINNED.md.

**Appendix C — env flags.** Main table: `Flag` · `Gate status` · `Effect on
the measured path` — gate status embeds the verdict (default ON / PASS /
FAIL-inert / killed regression) with evidence pointer. Secondary tier-split
table: `Flag` · `[M3U]` default · `Opt-out` · `Effect`; server/diagnostic
flags split with `Flag` · `Default` · `Effect` + `file:line`; numbered
correction subsections pinned to specific rows ("Row 11 correction
(2026-07-10)…"); closing block restates the measurement-protocol rules.
**Muser reuse:** keep the gate-status pattern re-anchored to parity-ledger /
release-lock verdicts; tier-split maps to local-Metal vs GB10-remote vs DFlash
lanes; build the fingerprint/resolved-signal (inert-flag guard) discipline
into the format itself — it was the ancestor's most-exploited blind spot.

**Appendix D — bibliography.** Entry: bold key (`[arxiv:XXXX.XYYYY]`) —
authors, *title* (venue year) — one-sentence why-it-matters — "Cited in [Ch
N]" back-references. Sections: Papers / Vendor specifications (`[Metal-SS]`,
`[Metal-PG]`) / Everything else — an explicit *non*-duplication policy (source
cites, internal docs, cross-chapter refs stay per-chapter: a master list of
repo-state citations is "one more place to go stale"). Housekeeping note
records the generating grep and the manual-update rule. **Muser reuse:**
format ports as-is; seed with
attention/GQA/RoPE/quantization/spec-dec/NVFP4/kvpack/disaggregated-prefill
papers actually cited; keep the non-duplication policy verbatim.

---

## Cross-half summary

**Five most portable devices:** (1) the roofline flip with two marked workload
points and per-row intensity arithmetic (Ch 23 §23.2) — recompute for Muser
decode + NVFP4 remote prefill; (2) the falsification ledger + tautology check
+ ledger-silences audit (Ch 25) — the parity-ledger genre, then run it on
Muser's own ledger; (3) the head_dim=2 fully-worked attention example +
max-subtraction running-max table (Ch 15) — extend with the sigmoid gate term;
(4) the 2×2 layout×precision decomposition with multiplicative check (Ch 14
§14.8) — the confound-splitter for any Muser KV-format/lane A/B; (5) the
barrier hazard taxonomy + scope-vs-per-resource timeline + single-fence
cross-CB lesson (Ch 22) — universal Metal truth, already load-bearing in
Muser's tracked-default reality.

**Three biggest port hazards:** (1) **Ch 14's KV story** — porting paged-Q8 as
"the" KV design misrepresents Muser (39-layer 2048-token ring + 13 growing
NoPE layers + durable kvpack + remote KV ingestion); even "grows with context"
is now per-layer-class. (2) **Ch 21–22's scheduler/barriers** — Muser verified
to have no VM program and tracked-by-default buffers; compiled-plan material
must be lineage/contrast, and the untracked-hazards framing must not leak in
(it was the ancestor's biggest fabricated-claim site, CORRECTIONS §3a). (3)
**Ch 23–25's measurement/gap methodology** — apparatus swaps wholesale to
parity ledger / release lock / accelerator_safe / receipts, and the gap
chapter rebuilds around Muser's levers (lanes, DFlash acceptance, handoff
phases); inheriting any ancestor ratio or the pre-correction "bandwidth root
cause" framing violates the launch-claims precedent discipline.
