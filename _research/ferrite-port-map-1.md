# Ferrite book → Muser book port map, part 1 (front matter + chapters 1–13)

Audit of `<ferrite-rs checkout>/inference-book/` (read-only) to feed the
Muser port. Covers README.md, SUMMARY.md, STYLE.md, glossary.md, and chapters 01–13
(Part I Metal model, Part II quantization, Part III model, first hero-kernel
chapters of Part IV). Chapters 14–25 are out of scope here (part 2 of this map).

Dispositions used below:
- **PORT-ADAPT** — concept survives; re-ground in Muser source/numbers.
- **PORT-REWRITE** — concept survives but Muser does it differently; say how.
- **KEEP-AS-LINEAGE** — present as Ferrite-ancestor lesson (cite `[ferrite-book Ch N]`).
- **DROP** — does not apply to Muser.

---

## 0. Front matter

### README.md / SUMMARY.md
- README establishes: exact production decode path (`vm-sync`, paged-Q8 KV,
  ladder attention), "code wins over book", every number tagged `[A18-neo]`,
  recurring question = "where does the 8% gap to llama-bench live". SUMMARY adds
  per-chapter one-liners + status lines, and a header note correcting the "hero
  attention kernel" myth (a position ladder, not one kernel) — summarizing
  measured reality over marketing.
- **Port:** muser-book README/SUMMARY already re-ground scope (Muse
  Glimmer/kquant/NVFP4/GX10, 10-part arc). The one unported device: a single
  **recurring question stated on page one**. Ferrite's (the 8% llama gap) cannot
  port; Muser must pick its own from the ledger (decode ceiling of the 30B at
  local bandwidth; DFlash acceptance; what remote prefill buys in TTFT). Decide
  before drafting Ch 1 — every kernel chapter's mandatory "where the gap lives"
  section hangs off it. Keep the SUMMARY header-note honesty: state the actual
  dispatch reality (ladders, spec-dependence) up front.

### STYLE.md (the writing contract — the most portable artifact in the repo)
Not on the required list but load-bearing; digest: §0 define-every-term-on-first-
use (one sentence + tiny diagram/worked number). §1 status line per chapter.
§2 kernel-chapter skeleton: (1) what it computes, (2) why it exists, (3) the
matrix op explained (2×2 worked example first time), (4) Metal kernel quoted,
(5) Rust dispatch, (6) access pattern ("where the bandwidth story lives"),
(7) tradeoffs (≥2, measured), (8) where the gap lives, (9) references. §3 mermaid
for flows, **ASCII for anything with byte offsets**; captioned figures. §4 quote
real source with `file:line`; elide with `// …` + note; never paraphrase in a
code fence. §5 per-chapter bibliography; every number tagged; uncited "why" is
`[unverified]`. §6 tradeoffs cite measurements, banned intuition patterns listed.
§8 absolute tok/s variance-warned or given as ratio; show byte arithmetic so the
reader re-derives. §9 glossary maintenance + anchor links. §10 reviewer veto
triggers. **Port: PORT-ADAPT essentially verbatim** — diff against muser-book's
STYLE.md and import the four highest-value clauses: ASCII-for-byte-offsets, the
veto list, skeleton items 6–8, show-the-arithmetic.

---

## 1. Chapter 1 — Why inference is a memory problem (`01-…md`)

**Teaching goal.** Establish the book's thesis in one sentence (one token ≈ stream
the whole model through the GPU; math finishes long before bytes arrive) and
derive it on the back of an envelope so the reader can reproduce it.

**Structure.** §1.1 thesis in one sentence · §1.2 the cast (six one-line
definitions) · §1.3 one token, costed by hand (6 steps) · §1.4 the bandwidth
wall (roofline) · §1.5 why "8 GB" is a lie and 45.95 GB/s is the real budget ·
§1.6 measured reality for Qwen2.5-1.5B · §1.7 the book's recurring question ·
§1.8 where the time of one token goes (pie at the floor) · §1.9 what's next.

**Devices worth porting.**
- **"One token, costed by hand" (§1.3)** — six numbered steps (params → bytes on
  disk → bytes read per token → FLOPs → GPU time → DRAM time → ratio); every
  number derived, approximates flagged. This is the chapter; port with Glimmer
  numbers.
- **The DRAM→GPU starvation diagram (Fig 1.1)** — weights arena streaming at the
  ceiling into a GPU that "finishes in 0.7 ms and idles ~21 ms waiting for bytes".
- **The roofline derivation (§1.4)** — arithmetic intensity 3.5 FLOP/byte vs
  machine balance ~109 FLOP/byte; "wrong side of the roofline by 30×", derived
  in place rather than cited.
- **"8 GB is a lie / bandwidth is the real budget" reframe (§1.5)** — capacity
  answers "does it fit", bandwidth answers "how fast"; ceiling-vs-target.
- **Single-stream saturation table (Table 1.1)** — 2 streams +0.43%, 4 streams
  −5.05%: "no 'just be more parallel' escape hatch". Concept ports; re-measure.
- **The three-lever strategy space** — read fewer bytes / read faster / avoid
  re-reading; "there is no fourth option". Muser's version *gains* a fourth
  lever — move the work elsewhere (remote prefill, speculation) — state it.
- **Energy aside (§1.6)** — fixed ~4 W ⇒ energy/token = tokens/s ⇒ bandwidth
  efficiency *is* energy efficiency. Fragile off A18 (see §17).
- **Pie at the theoretical floor, not the measured token (Fig 1.2)** — the
  floor-to-real gap is "a property of the engine, not the math".
- **Variance warning + ratio rule in chapter 1** — absolute tok/s squishy;
  same-session interleaved ratio is the only cross-engine statistic.

**Ferrite-specific (replace/re-ground).** 1.77 B params; 4.5 bits/weight Q4_K;
~1.0 GB stream (994 MB measured cross-check); A18 Pro ~5 TFLOP/s
`[unverified]`; 45.95 GB/s ceiling `[PERF §2]`; 33.20 tok/s; 0.92×; 36.9 GB/s =
80.3%; 0.104 J/tok; 3.94/4.13 W; 22 ms vs 0.7 ms; `scripts/neo_gate.sh`;
`docs/PERF.md`; the entire 8%-gap recurring question (§1.7).

**Port disposition.** PORT-ADAPT §1.1–1.4 and 1.8 (re-derive with Muse Glimmer:
~30 B params, kquant/NVFP4 effective bits, actual model GB, Muser-Mac DRAM
ceiling, measured tok/s with/without the spec lane). §1.5's saturation table
and §1.6's energy figures: KEEP-AS-LINEAGE (A18 measurements) unless
re-measured. §1.7 recurring question: PORT-REWRITE — must become Muser's own
open question. The 30× memory-bound conclusion likely survives qualitatively
for dense 30B decode; quantify with Muser numbers.

**Segues.** Opens the book cold (no prereqs) with the one-sentence thesis —
strongest opening in the book. Closes with the parts I–VI roadmap and the
promise that §1.1 "will be something you have proven, kernel by kernel".

---

## 2. Chapter 2 — The Metal compute model (`02-…md`)

**Teaching goal.** Teach the Metal host API and the three nested parallelism
units, so any later shader is readable; establish the record-then-play model.

**Structure.** §2.1 what a GPU actually is · §2.2 Metal: Apple's GPU API · §2.3
"record a tape, then press play" · §2.4 the three handles (MetalContext) · §2.5
threads, threadgroups, and the SIMD group · §2.6 binding memory: set_buffer and
the offset trick · §2.7 the full encode/commit/wait cycle · §2.8 encode_* vs
dispatch_* · §2.9 a tiny end-to-end example · §2.10 what's next.

**Devices worth porting.**
- **The tape-recorder analogy (§2.3)** — "record many songs onto a cassette,
  press play once"; command-buffer amortization made intuitive before any perf
  argument.
- **The four-line dispatch sequence** — bind kernel / bind buffers / bind
  constants / launch = "one dispatch". A memorable chant that reduces Metal.
- **First real kernel + annotated wrapper (residual_add)** — dispatch_* wrapper
  with six numbered inline steps, then the encode_* inner half, closed with
  "what it does NOT do" (no buffer, no commit, no wait).
- **The three-nested-units ASCII diagram (Fig 2.1)** — grid ⊃ threadgroups ⊃
  SIMD groups of 32 lockstep lanes.
- **"The SIMD group is the unit that actually matters" callout** — hardware
  executes in 32-wide chunks; log₂(32)=5 barrier passes collapse to one
  `simd_sum`; the `_4sg` suffix tell for reading kernel names.
- **Two contrasting dispatch geometries** — RMSNorm 1×128 vs matvec `rows`×32;
  "grid is the output shape" taught concretely.
- **Skeptical-reader Q&A sidebars** — "Isn't reference-by-string slow?" (1.4%
  encode incl. lookups; llama does snprintf per dispatch) and "What about the
  512 idle threads?" (guard-exit is a branch; production fused the op away).
  Models interrogating plausible objections with data.
- **`encode_*` vs `dispatch_*` vocabulary (§2.8)** — hot-path vs test-only
  prefixes; used by the whole book afterwards.

**Ferrite-specific.** `MetalContext` struct (context.rs:240) incl. ggml/dk64
optional libraries; 215 shader files; ~171 dispatches/token; 28 layers; hidden
1536; pipeline-HashMap-panics smell; 400 µs/1.4% encode; llama.cpp
`std::unordered_map` PSO cache comparison; M4 compiler-bug isolation library;
`t4_hazard_dump.rs` geometry.

**Port disposition.** PORT-ADAPT — Metal fundamentals (device/queue/tape,
thread/threadgroup/SIMD-group, set_buffer/set_bytes, grid math) port nearly
verbatim. Re-ground every quote in Muser source (context struct, shader count
incl. the `ferrite/` lineage dir, dispatch count for 52 layers + spec lane).
The `_4sg` naming tell is a Ferrite convention — replace with Muser's own
naming tell. M4-bug and llama-PSO-cache asides: KEEP-AS-LINEAGE or drop.

**Segues.** Opens cold. Closes: "you now know every Metal concept needed to read
any Ferrite kernel" → Ch 3 memory, Ch 4 compilation. Clean.

---

## 3. Chapter 3 — Unified memory and GpuBuffer (`03-…md`)

**Teaching goal.** One physical DRAM for CPU+GPU; storage modes; the arena
offset-view trick; zero-copy mmap of the GGUF; why Ferrite turns hazard tracking
off and what makes that safe.

**Structure.** §3.1 one DRAM, two brains · §3.2 the three storage modes · §3.3
the GpuBuffer struct · §3.4 the offset-view / arena trick · §3.5 allocation and
the B4 zero-init fix · §3.6 turning off Metal's hazard tracking · §3.7 zero-copy
mmap of the GGUF · §3.8 the placement-heap path (GpuHeap) · §3.9 buffer identity
buf_id() · §3.10 the MetalContext · §3.11 tradeoffs · §3.12 what's next.

**Devices worth porting.**
- **Unified-vs-discrete split diagram (Fig 3.1)** — two DRAM pools + PCIe
  memcpy vs one shared pool; "uploading weights is a no-op".
- **Storage-mode triage (§3.2)** + **six-field struct walk (§3.3)** — Shared/
  Private/Managed with why each is irrelevant on Apple Silicon; one sentence
  per field makes the arena trick inevitable before it appears.
- **The arena diagram (Fig 3.2)** — four `GpuBuffer{offset, view_byte_len,
  inner: arena}` views into one MTLBuffer; "Metal never notices".
- **Page-alignment worked arithmetic (§3.7)** — tensor at 0x1_4A00 → round down
  to 0x1_4000, adjustment 0xA00 becomes the view offset; 16 KB pages.
- **Zero-copy flow diagram (Fig 3.3)** — GGUF → mmap →
  new_buffer_with_bytes_no_copy; "no memcpy, ever"; demand paging.
- **The evidence-box genre (§3.5/3.6/3.9)** — boxed verdicts ("Resolved",
  "Partly verified — the framing undersells a real risk", "Corrected 2026-07")
  dissecting a fabricated/stale source comment clause by clause (the
  "llama.cpp uses untracked by default" myth; six files with six different
  "the" barrier counts; the buf_id overlapping-page landmine). The book's
  signature honesty device; maps perfectly onto Muser's fail-closed culture.
- **Arena vs heap contrast (§3.8/3.11)** — collapsing identity (views) vs
  preserving identity (heap sub-buffers); when each is safe. General Metal
  lesson.
- **Barrier-sufficiency argument (§3.6)** — "untracked is safe iff something
  else enforces ordering", with the real A/B (tracked −2..−4%) and the N_CB=2
  MTLFence bug as the price of trusting a comment's false guarantee.

**Ferrite-specific.** GpuBuffer/buffer.rs/views.rs/maintenance.rs paths;
FERRITE_UNTRACKED_HAZARDS plumbing; 43 barriers / 2.7 µs / 0.4%; B4 fix;
zero_raw_gpu 40% flake; packed-activations tombstone (+0.5% BW falsification);
llama.cpp `ggml_metal_buffer_map` corroboration; buf_id overlap hazard.

**Port disposition.** PORT-ADAPT the Apple-Silicon facts (unified memory,
storage modes, 16 KB pages, mmap zero-copy — true on Muser's Mac). PORT-REWRITE
the hazard-tracking section: it depends on Muser's actual ordering design —
keep the analytical frame ("what makes turning tracking off safe") and re-run
it on Muser's barrier design read from source. Audit-box genre and B4-style
warts: adapt if Muser has analogues, else KEEP-AS-LINEAGE. Note: unified memory
is *the* contrast to the GB10's discrete memory — this chapter is the natural
place to seed the local/remote memory split Part VI will need.

**Segues.** Opens recalling Ch 2's `set_buffer`. Closes toward Ch 4 (compile)
and Ch 5 (GGUF bytes). The llama.cpp "standard practice, not a wart"
corroborations are a repeated strength worth imitating (Muser's comparanda:
llama.cpp, vLLM, kvpack).

---

## 4. Chapter 4 — PSO and shader compilation (`04-…md`)

**Teaching goal.** How `.metal` text becomes a runnable kernel: library vs PSO
stages, metallib precompile, binary-archive cache, function-constant
specialization, fast-math.

**Structure.** §4.1 three-stage compile pipeline · §4.2 embedding shaders
include_str! · §4.3 the concat! trick · §4.4 library creation, two paths
(4.4.1 runtime JIT; 4.4.2 build-time metallib; 4.4.3 three-tier resolution;
4.4.4 ggml bridge) · §4.5 the PSO cache (MTLBinaryArchive) · §4.6 PSO creation
through the archive · §4.7 the GpuKernels constructor · §4.8 function-constant
specialization · §4.9 the PSO HashMap · §4.10 tradeoffs.

**Devices worth porting.**
- **Compile-pipeline mermaid (Fig 4.1)** — source →(include_str!|xcrun metal)→
  library → function → PSO → HashMap, with the archive hit/miss side-loop.
- **"Why two compiles" box** — frontend (parse/typecheck/lower to AIR; per
  library; device-independent) vs backend (AIR→ISA; per kernel; per device);
  pay each at a different time.
- **The four-component cache-key breakdown (Fig 4.2)** — ASCII anatomy of
  `pso_v3_<exe_mtime>_<shader_hash>_<devname_hash>.metalbin` and what each
  field guards (stale on rebuild/shader edit/device change; manual
  format-version bump).
- **Descriptor-vs-function PSO API choice** — descriptor form chosen *only*
  because it accepts `binaryArchives`; cost = an FFI layer, payoff = the cache.
- **Function-constant specialization walkthrough** — slots 10/11; the
  `is_function_constant_defined` ternary; one source serving generic +
  specialized PSOs; map-key ≠ function-name gotcha.
- **Fast-math contract + `precise::cos/sin` (§4.10)** — fast-math trig ~11-bit;
  at RoPE base 1e6 and pos 200+ angle error corrupts attention; *accuracy, not
  determinism* is the reason. Directly relevant to Muser if Glimmer's theta is
  large — port prominently.
- **The dk64 wart** — a compiler bug worked around by library isolation, named
  as a wart, not a design.
- **Fingerprint discipline (§4.4.4)** — gate logs print *resolved* signals
  (`lib=metallib+ggml-bridge`), not env echoes; caught an inert flag via
  fingerprint diffing. Muser receipts/ledger should adopt the same rule.

**Ferrite-specific.** All paths (ferrite-metal-shaders lib.rs, build.rs,
pso_cache.rs, fast_metal_pso_cache.m, constructor/*); FERRITE_METALLIB /
RUNTIME_SHADERS / SHADER_OVERRIDE_DIR / GGML_METALLIB flags; MSL V3_1; Metal-4
tensor-ops policy; receipt-v4; ~12% ggml-bridge prefill claim; ARCHIVE_LOCK
SIGSEGV workaround; slot numbers.

**Port disposition.** PORT-ADAPT — compilation mechanics are Metal-universal;
re-ground every quote/flag/slot in Muser's shader build (incl. the `ferrite/`
lineage dir — attribute where the lineage is literal). SIGSEGV lock and dk64
bug: KEEP-AS-LINEAGE unless Muser hit them too. The precise::trig lesson:
PORT-ADAPT with Glimmer's theta.

**Segues.** Opens by paying off Ch 2's hand-wave ("bundle of compiled kernels,
addressable by name"). Closes pivoting to quantization (Ch 5). Fine.

---

## 5. Chapter 5 — The Q4_K super-block (`05-…md`)

**Teaching goal.** Teach block quantization byte-by-byte: why naive 4-bit fails,
the min+offset local-range idea, the 144-byte super-block, the 6-bit scale
packing, a full hand dequant.

**Structure.** §5.1 the weights do not fit · §5.2 naive 4-bit is not enough ·
§5.3 the min+offset idea · §5.4 the super-block layout · §5.5 sub-blocks and
6-bit scale packing · §5.6 the dequant formula, one element by hand · §5.7 f16:
why 16 bits is enough for the scale · §5.8 the worked example — read this twice
· §5.9 the block-alignment rule · §5.10 why 256-element super-blocks · §5.11
Q4_K_M mixing in Q6_K · §5.12 tradeoffs.

**Devices worth porting.**
- **"Quantize locally, not globally" (§5.3)** — a small block's range is far
  narrower than the tensor's; spend the 4 bits inside the narrow band; "the
  single idea behind every K-family format". Universal quant insight.
- **Dequantizer-first-then-byte-map order (§5.4)** — quote the complete dequant,
  read it in four strips, *then* show the byte map; code explains layout.
- **The 144-byte byte map (Fig 5.1)** — offset/size/field/meaning table; ASCII
  per STYLE §3. Plus **sub-block interleaving (Fig 5.2)** — low nibbles = even
  sub-block, high = odd, matching the dequant loop's two output lines.
- **The 96-bit scale-packing bit table (Fig 5.3)** — every bit of the 12-byte
  strip accounted; sanity check by substituting j=5 into the extractor.
- **The hand-built worked example (§5.8, "read this twice")** — clean f16
  headers, all 8 scales/mins packed by hand (per-byte arithmetic shown),
  round-trip verified for sub-block 5, three elements dequantized with every
  multiply written out ("one byte, two sub-blocks, two completely different
  scales — that is the whole point"). Strongest worked example in the book.
- **Format comparison table (Fig 5.4)** — seven formats by block/bits/min+offset;
  punchlines: "Q4_K and Q4_0 cost the SAME bitrate; the win is structural" and
  "the real comparison is Q4_K vs Q4_1 — the super-block recovers the 0.5 bit
  the min costs".
- **Block-alignment rule (§5.9)** — divisibility enforced at quantize time;
  lm_head clears it via the *product* of dims though vocab isn't a multiple.
- **Bits-per-element accounting always shown as arithmetic.**

**Ferrite-specific.** The entire format: 144 B/256 elems, 8×(6+6)-bit scales,
`get_scale_min`, k_block.rs/q4k.rs/gguf_quantize.rs paths; 1.77 B/8 GB/7.08 GB
arithmetic; Q4_K_M mix (down-proj Q6_K alternating, lm_head Q6_K; 24.5% of
decode time); 6–10%-faster-than-llama kernel verdict; 7B-cannot-decode-on-8GB.

**Port disposition.** The chapter's pedagogy is the template for Muser's Part
II; the content is Ferrite's format. Split three ways: (1) kquant DFlash block
chapter — PORT-REWRITE using every device above (byte map, hand-packed worked
example, format table, local-vs-global framing) on Muser's actual kquant
layout read from source; (2) NVFP4 chapter — PORT-REWRITE likewise (e4m3 +
scales; re-run the "same-bytes-better-representation" analysis for FP4-vs-INT4,
a point the ancestor couldn't make); (3) Q4_K itself — KEEP-AS-LINEAGE (a
compressed aside; the shader tree has a `ferrite/` dir that may still speak it).
The 6–10%-faster-than-llama verdict: lineage only.

**Segues.** Opens from Ch 3 ("never launches a kernel — pure byte layout;
kernels that consume these bytes start in Ch 11") — an explicit promise that
pays off 6 chapters later. Closes toward Ch 6 (the other quantization: KV).

---

## 6. Chapter 6 — Q8_0: quantizing the KV cache (`06-…md`)

**Teaching goal.** Why the KV cache is quantized differently from weights:
online vs offline budgets, symmetric vs asymmetric, block size from access
pattern.

**Structure.** §6.1 the cache is a third memory beast · §6.2 why Q8_0, not Q4_K
· §6.3 the block layout · §6.4 symmetric vs asymmetric · §6.5 the online
quantization math (+ worked example) · §6.6 the store kernel in Metal · §6.7
the dequant inside the attention gather · §6.8 why int8, not int4 · §6.9 why
block size 32, not 256 · §6.10 total footprint · §6.11 tradeoffs · §6.12 where
the gap lives.

**Devices worth porting.**
- **"Third memory beast" framing (§6.1)** — weights (static, streamed) /
  activations (tiny, per-token) / KV cache (dynamic, growing, re-read).
- **Offline-vs-online quantization dichotomy (§6.2)** — weights quality-optimal
  and amortized at build time; cache latency-sensitive per-token in-kernel;
  "the format choice follows from the budget". The conceptual heart; universal.
- **Symmetric-vs-asymmetric side-by-side (Fig 6.3)** — range, quant ladder,
  formulas, and a quantize-vs-dequantize cost table; punchline "the cost
  difference is in quantizing, not dequantizing". Plus the **DC-offset price**
  (values in [5,7] halve symmetric precision; asymmetric captures it in min).
- **Quantize-one-block-by-hand (Fig 6.4)** — amax=2.0, d=2/127, inv_d=63.5,
  the 63.5 tie (MSL round-away-from-zero), dequant errors (+0.8%, exact at
  amax); "zero error at the extremes, maximum in the middle — the
  symmetric-quantization signature".
- **Store-kernel read as the 4 steps (§6.6)** — each line mapped to the math;
  the d==0 guard; "no min scan, no two-level fit, no 6-bit packing — that is
  the online budget Q4_K could not meet".
- **Fused-dequant-in-dot diagram (Fig 6.5)** — f16 + 32×i8 → widen → ×d_k →
  ×q[j] → acc; "dequant happens in registers, in flight with the dot product".
- **Read-frequency / compounding-error asymmetry (§6.8)** — weight error is a
  fixed deterministic bias; KV error is fresh every read with no chance to be
  absorbed; tighter error budget. Flagged `[unverified]` where folklore.
- **Block-size-32 three-reason argument (§6.9)** — weights read bulk (big block
  amortizes) vs cache read scattered (per-position rows); head_dim 128 = 4×32;
  SIMD lane width 32 = one block per `simd_sum`. Geometry-from-access-pattern
  reasoning at its best.

**Ferrite-specific.** Qwen KV geometry (28×2×2048×128 → 31.2 MB Q8_0 at ctx
2048); `mha_kv_store_q8_paged` / `attention_mla_paged.metal` kernels;
block_arena.rs constants; llama.cpp block_q8_0 lineage; gap tail (77%/85%).

**Port disposition.** Concepts (offline/online, symmetric/asymmetric,
read-compounding, scattered-access block sizing): PORT-ADAPT. Content:
PORT-REWRITE — Muser's KV story is paged local KV **plus kvpack serialization
plus remote handoff**; the "third beast" becomes "the third beast and its
transported twin". Verify from `crates/muser-kvpack/` + `third_party/kvpack/`
what the resident and on-wire KV formats are, then write: local store format,
kvpack container, handoff format, and the precision-parity argument between
them (what precision does NVFP4-prefilled KV arrive in; does the Mac
re-quantize?). Q8_0: KEEP-AS-LINEAGE if lineage kernels still use it, else
drop the byte detail and keep the dichotomies.

**Segues.** Opens recalling Ch 5 (weights quantized; now the second pile).
Closes with "KV is not the gap at tg128; the long-context regime is where KV
quant pays" — an honest scope statement; Muser's version must re-derive this
with Glimmer's head counts (52 layers changes KV arithmetic a lot).

---

## 7. Chapter 7 — The Qwen2.5-1.5B architecture (`07-…md`)

**Teaching goal.** Build the complete model map — hyperparameters verified from
source, block anatomy, GQA, untied embeddings, theta, SiLU — with every later
chapter anchored to its tables.

**Structure.** §7.1 what a transformer is (one paragraph) · §7.2 exact
hyperparameters (verified from source) · §7.3 one transformer block · §7.4 the
full model · §7.5 components tour · §7.6 GQA · §7.7 head_dim and why the 0.5B
differs · §7.8 RoPE theta 1e6 · §7.9 activation SiLU · §7.10 untied embeddings
· §7.11 per-layer tensor inventory · §7.12 memory accounting · §7.13 tradeoffs
· §7.14 what's next.

**Devices worth porting.**
- **Double/triple-cited hyperparameter table (Table 7.1)** — each value cited
  from a Rust constant *and* a validation test *and* the live GGUF extractor
  with the metadata key annotated per field ("if a checkpoint lied, Ferrite
  would error out"). Kills drift-by-hardcoding; port with Muser's startup
  identity validation (revision/size/SHA-256 per PINNED.md).
- **"The single most important diagram in the book" (Fig 7.1)** — the block
  mermaid (norm→QKV→RoPE→cache→score→O-proj→+; norm→gate/up/SiLU⊙→down→+) with
  two dashed residual skips and chapter pointers per box.
- **Three-things-to-internalize list** — RMSNorm twice; attention is the only
  cross-token mixer; the residual adds are the spine (`h_out = h_in +
  Attn(…) + FFN(…)` written out).
- **GQA 6:1 ASCII (§7.6)** — 12 Q-heads in 2 groups over K0,V0 / K1,V1; the
  bandwidth win quantified (KV 6× smaller); "n_kv_heads is the number that
  matters for cache sizing".
- **Parameter accounting by hand (§7.12)** — per-layer counts summed to
  1,777,030,656; split 65% FFN / 8.7% attention / 26% embed+lmhead; "every
  later bandwidth number divides through a total like this one".
- **Tradeoffs framed as model-author decisions (§7.13)** — "none is a Ferrite
  decision… but they shape everything the kernels must be good at".
- **`[unverified]` on every quality rationale** (why 1e6, why untied) while
  structural facts stay cited — the epistemic split made visible.

**Ferrite-specific.** All numbers: 28/1536/12/2/128/8960/151936/1e6/1e-6; q_dim
1536, kv_dim 256; untied with output.weight Q6_K; config.rs /
multi_model_validation.rs / shard/config.rs / shape_check.rs / layer_config.rs
paths; FERRITE_QKV_FUSED 0.5B-only story; Gemma4 contrasts.

**Port disposition.** PORT-ADAPT the skeleton wholesale — this is the template
for Muser's "Muse Glimmer architecture" chapter: build the hyperparameter table
from Muser's config/loader with the same double-citation (plus PINNED.md's
SHA-256 identity check), clone Fig 7.1 with Muser's chapter pointers, GQA
section with Glimmer's real ratio, tensor inventory from Muser's shape checker,
hand accounting to ~30 B with the FFN/attention/embed split. Re-derive the
"three things to internalize" list if Glimmer has any architectural novelty
(norm placement, QK-norm, sliding window). Downstream chapters reference these
tables — get them right first.

**Segues.** Opens "never read a transformer paper — we define every term in
place; here we only build the map". Closes: "every later chapter opens by
pointing back at a row in Tables 7.2/7.3" — the table-anchoring convention that
keeps 16 kernel chapters coherent. Port that convention.

---

## 8. Chapter 8 — The forward pass at a glance (`08-…md`)

**Teaching goal.** The spine: one diagram of the whole decode loop, the residual
stream as the central data structure, the activation arena, the precompiled
VM program, one command buffer per token, and the measured time budget.

**Structure.** §8.1 the two modes: prefill and decode · §8.2 the decode
forward-pass diagram (centerpiece) · §8.3 the residual stream · §8.4 the
GpuArena · §8.5 the VM-exec route · §8.6 one command buffer per token · §8.7 a
concrete walkthrough, real numbers · §8.8 the per-token time budget · §8.9
tradeoffs.

**Devices worth porting.**
- **Prefill/decode one-sentence contrast (§8.1)** — "prefill has parallelism
  across tokens so it is compute-bound; decode is one token so zero cross-token
  reuse and bandwidth-bound. Same matrices, opposite bottleneck."
- **Figure 8.1, "the single most-reused figure"** — boxes ①–⑩ with shapes,
  FUSED annotations, ×28 loop collar; every kernel chapter zooms one box. Plus
  "read the fusions explicitly": five named fusions (a)–(e) with gates and
  citations — the fusion list is half the architecture of the book.
- **Residual-stream-as-pipe ASCII (Fig 8.2)** — one 6 KiB buffer; 28 layers
  each += a delta; "the same bytes are read and rewritten 56 times".
- **Activation-pool table (Fig 8.3)** — every named buffer, shape, bytes, role;
  "≈700 KiB — utterly negligible next to 994 MB"; allocated once, zero
  hot-path allocation.
- **The Op enum excerpt + compiler docstring read as "the menu"** — production
  "deletes lines from it"; Op↔box mapping.
- **Per-token time budget bar (Fig 8.5)** — 97.6% GPU / 1.4% encode / 0.4%
  barriers; "the token time IS GPU time"; therefore the lever is kernel
  bandwidth, not orchestration. Governing measurement of the book.
- **Three structural bets (§8.9)** — one CB per token / precompiled Vec<Op> /
  arena reuse, each with the measured alternative; "decode is bandwidth-bound,
  so the orchestration layer's job is to get out of the way".
- **The `[unverified]` tension box (§8.2)** — diagram draws fused gate-up but
  the barrier contract says split; "Ch 17 will settle this with a live
  Op-dump". Honest open-question bookkeeping inside the spine chapter.

**Ferrite-specific.** vm-sync route; position ladder; all five fusion names and
gates; arena_state.rs / ops.rs / compile.rs / vm_forward.rs line-tagged quotes;
~140 Ops, 43 barriers; 994 MB; 30.1 ms; FERRITE_LM_REQUANT; n_cb; t4 contract.

**Port disposition.** PORT-ADAPT the skeleton (two modes; centerpiece diagram;
residual stream; arena table; time budget; bets). PORT-REWRITE the content in
two big ways: (1) **speculation is absent from Ferrite's spine** — Muser's
Figure-8.1 equivalent must add the DFlash draft loop (draft generation, target
verify — a batched matvec that *reintroduces* GEMM-like reuse into decode —
accept/reject, KV rollback); the "decode is batch=1" thesis becomes "batch=1
per accepted step, batch=k at verify", i.e. the fourth lever from Ch 1 made
real. (2) **disaggregated prefill changes §8.1's dichotomy** — on the remote
lane the GB10 prefills NVFP4 and the Mac *receives* KV over Handoff V2; the
two-modes section must become three-regime (remote prefill / local prefill /
decode) or re-open the question in the lane's own chapter. Time budget and bets
must be re-measured on Muser (52 layers, spec lane, handoff receive); the
fusion list is Muser's to discover from source.

**Segues.** Opens defining prefill/decode formally (payoff of Ch 1's promise).
Closes handing off to the kernel chapters — this chapter is the hinge of the
whole book; port it early and carefully.

---

## 9. Chapter 9 — Token embedding lookup (`09-…md`)

**Teaching goal.** Why the embedding is a CPU-side one-row lookup, not a GPU
gather; it seeds the residual stream.

**Structure.** §9.1 what this chapter is about · §9.2 what an embedding is ·
§9.3 why this is the start of the residual stream · §9.4 why CPU — the call
site · §9.5 the dequant, one row · §9.6 the flow end to end · §9.7 tradeoffs:
CPU lookup vs GPU gather · §9.8 where the gap lives · §9.9 the tie-vs-untie
note · §9.10 what's next.

**Devices worth porting.**
- **"A memcpy with a stride — the intelligence lives in the table" (§9.2)** —
  deflates the scariest-sounding op to nothing.
- **The drawn matrix (Fig 9.1)** — with the honesty note that it is drawn
  transposed for readability while memory is vocab-major (`offset = token_id ×
  row_bytes`).
- **"The one number that makes it obvious: 6 KiB per token"** — the whole
  chapter hangs on one derived quantity.
- **CPU-vs-GPU-gather tradeoff (§9.7)** — "the CPU wins because the work is too
  small to amortize a GPU dispatch over"; dispatch ordering cost or barrier;
  batch=1 vs prefill-batch criterion for when the gather *is* right.
- **"Large in memory, trivial in bandwidth — both facts are true
  simultaneously; do not collapse them" (§9.8)** — 131 MB table, 6 KiB/token
  read. A reusable analytical distinction.
- **Quantized-not-f32 table trade** — 131 MB vs 933 MB f32; dequant per row is
  trivial; residency is forever.

**Ferrite-specific.** vm_launch_setup.rs / embedding.rs / dequant.rs paths;
row_bytes = 864 B (Q4_K); dtype facts read from the GGUF header; Gemma4
embed_scale; 0.5B-vs-1.5B comment archaeology (the 138/519 MB comment is the
*0.5B's* figures — the book catches it).

**Port disposition.** PORT-ADAPT — same argument shape with Glimmer's hidden
size, vocab, actual embedding dtype; verify Muser also embeds on CPU, and check
the spec-lane draft model's embedding (draft steps may change the calculus).
Tie/untie: re-ground on Glimmer.

**Segues.** Opens "the first chapter that actually touches the residual stream…
short because the operation is short" — sizing chapters to their op is itself a
portable editorial rule.

---

## 10. Chapter 10 — RMSNorm (`10-…md`)

**Teaching goal.** What normalization does; why root-mean-square not stddev; the
4-simdgroup reduction kernel line by line; rsqrt precision.

**Structure.** §10.1 what normalization computes · §10.2 RMSNorm by hand ·
§10.3 the Metal kernel rms_norm_vec4_4sg · §10.4 the Rust dispatch · §10.5 the
fused residual+RMSNorm kernel · §10.6 the final norm before the LM head · §10.7
tradeoffs · §10.8 where the gap lives.

**Devices worth porting.**
- **Multiplication-drift motivation (§10.1)** — 1.2^28 ≈ 165 vs 0.8^28 ≈ 0.002;
  normalization is the valve. Two numbers, whole justification.
- **LayerNorm/RMSNorm formula pair + identity** — mean(x²) = σ² + μ², so rms ≥
  σ with equality iff mean zero; "the sense in which the two scaling steps
  coincide". Plus **two reasons to drop the mean** — cost (two reductions →
  one; reductions are the expensive part on GPU; portable physics) and
  empirical equivalence (paper-cited, `[unverified]` for the model).
- **The `[3,4,0,0]` worked example (Fig 10.1)** — sumsq 25, rms 2.5, inv_rms
  0.4, output [1.2,1.6,0,0]; check mean(y²)=1; then γ=[2,.5,1,1] variant
  showing γ re-learns the shape after normalization fixes scale.
- **Kernel line-by-line as named ideas** — float4 cast (4× fewer loads, 16 B
  transactions), strided loop, `simd_sum` (one-cycle 32-way), threadgroup
  handoff, two barriers and why each exists, normalize-and-scale loop.
- **Lane-geometry diagram (Fig 10.2)** — 1536 = 4×32×12 factorization driving
  the whole kernel shape.
- **rsqrt vs 1/sqrt bit-exact tradeoff (§10.7)** — ULP defined; compounding-ULP
  motivation; the gated llama-bit-exact port kernel. Precision-vs-parity
  discipline Muser's cross-engine diffing will want.
- **Fused residual+norm transaction counting** — unfused 2R+1W of hidden vs
  fused 1R+1W, one fewer encoder-boundary gap.
- **"This kernel is not the gap" byte arithmetic (§10.8)** — 18 KiB/dispatch,
  <2 MB/token; latency-motivated, not bandwidth.

**Ferrite-specific.** 57 norms/token (2×28+1); rmsnorm.metal /
rms_norm_llamacpp.metal paths; fc3584 prebaked PSO; FERRITE_RMSNORM_BITEXACT /
FC3584 typed controls; ~7 µs idle comment; has_qk_norm=false.

**Port disposition.** PORT-ADAPT — RMSNorm is (verify) in Muse Glimmer; math,
worked example, and reduction-kernel pedagogy port verbatim. Kernel quotes
become Muser's (possibly the ferrite-lineage shader — a nice lineage-citation
moment). 52 layers ⇒ 105 norms/token if no QK-norm. Re-run the fused-vs-
unfused and 1sg-vs-4sg tradeoffs on Muser's dispatch policy. The `[3,4,0,0]`
example and the drift motivation: port as-is.

**Segues.** Opens from the residual stream (Ch 8) and matvec promise (Ch 11).
Closes "if you are hunting the DRAM gap, look at the matvec family, not here" —
the first execution of the every-chapter gap verdict.

---

## 11. Chapter 11 — The Q4_K GEMV family (`matvec_q4k_f32_v4`) (`11-…md`)

**Teaching goal.** The hero chapter: the matvec math from zero, the v4 access
pattern (4-way block interleave, register-cached x, deferred scaling), the
function-constant specialization — and the measured verdict that this inner
loop is NOT the engine's gap.

**Structure.** §11.1 what it computes · §11.2 why it dominates decode · §11.3
the matrix operation, explained from zero · §11.4 the kernel (7 subsections:
lane decomposition; 4 rows/TG; x register cache; block-strided loop; deferred
scaling; the dual-row MAC helper; simd_sum + write-back) · §11.5 grid ×
threadgroup dispatch arithmetic · §11.6 Rust dispatch + FC specialization ·
§11.7 the access pattern · §11.8 the 6–10% measurement · §11.9 the gap root
cause · §11.10 the matvec_q4k_fast llama port · §11.11 the residual-fused
variant · §11.12 tradeoffs · §11.13 what's next.

**Devices worth porting.**
- **The 4×4 matvec by hand (Fig 11.2)** — four dot products with all products
  written out; "one dot product of length 4 per output row". Plus the
  **row-major ⇒ contiguous-span-per-row** observation tying layout to DRAM
  efficiency in one sentence.
- **The 32-lane → 4 groups of 8 → (iq, ir) decomposition diagram (Fig 11.3)** —
  and the note that the split *matches the 8-sub-block structure* so scale
  decoding lines up.
- **The x register-cache diagram (Fig 11.5)** — load 32 elems once, reuse
  across 2 rows; "a 2× cut on the smaller of the two streams".
- **The block-strided loop diagram (Fig 11.6)** — four 144-byte blocks in
  flight per iteration vs one; in-flight requests as "the currency of
  bandwidth-bound kernels"; kernel comment quoted ("critical for saturating
  Apple Silicon's high-latency memory bus").
- **Deferred scaling, decomposed in three steps (§11.4.5)** — bit-position
  normalization folded into the final 1/256·1/16 multipliers (bit extraction
  becomes free); scale applied once per sub-block; min term from cached `sumy`
  in closed form (−dmin·m·Σx). Then the full helper quoted and the kmask
  dissection (0x3F3F/0x0F0F/0xC0C0 mapped back to Ch 5's packing).
- **Grid arithmetic for three real shapes (§11.5)** — Q-proj 384×64; gate/up
  2240×64; down 384×64 with 35 blocks/row; "grid is rows, never cols" and why.
- **Access-pattern walk (§11.7)** — 864 B/row, 1.27 MB/Q-proj, ceiling math
  (994 MB / 45.95 GB/s ≈ 21.6 ms floor).
- **The 6–10% verdict quoted verbatim + "read that twice" (§11.8)** — Ferrite's
  kernel *beats* llama's in isolation; therefore the gap is elsewhere. The
  rhetorical pivot of the whole book.
- **Gap-attribution decision tree (Fig 11.7)** — mermaid with every branch
  labeled FALSIFIED + data (encode 1.4%; serialization ≈ free; buffer count
  +0.5%; all variants at parity); REMAINS → pipeline DRAM structure; NEXT →
  measurement not optimization. Plus **"the trap this section exists to
  prevent"** — don't polish the inner loop.
- **v4-as-fusible-substrate argument (§11.12)** — 16 call sites share
  `q4k_v4_dual_row_mac`; v4 exists so other ops fuse into the weight-fetch
  pattern; the llama port is a one-off diagnostic.

**Ferrite-specific.** Everything kernel-level: v4 geometry, helper, FC slots
10/11, FERRITE_FAST_Q4K, parity sweep numbers (34.70 vs 34.52 tok/s), 994 MB /
77% vs 85% / 70% vs 83%, packed-activations falsification, llama
kernel_mul_mv_q4_K_f32.

**Port disposition.** The chapter *template* (deliberately-longest
hero chapter; math-from-zero → decomposition diagrams → helper dissection →
dispatch arithmetic → access pattern → verdict) is the port's master pattern:
PORT-ADAPT the template and the general GEMV pedagogy (4×4 by hand, row-major,
register caching, in-flight loads, deferred scaling as a *concept* — it applies
to any block-quant format with per-block scales, kquant and NVFP4 included).
PORT-REWRITE the content: Muser's hero kernel is the kquant/NVFP4 matvec
(possibly ferrite-lineage — cite if so); deferred-vs-pre-decoded scaling and
shape-dependent kernel selection survive as the design space. The "inner loop
is not the gap" verdict is a Ferrite measurement: KEEP-AS-LINEAGE as the
ancestor punchline; Muser needs its own verdict on its own question. The
FALSIFIED-branch decision tree is the single best figure to imitate for
Muser's gap/parity chapters.

**Segues.** Opens declaring its own importance ("longest kernel chapter on
purpose: ~90% of decode time"). Closes to the 4sg sibling (Ch 12). Strong.

---

## 12. Chapter 12 — The 4sg variant: K and V projections (`12-…md`)

**Teaching goal.** The sibling GEMV: one row per threadgroup, blocks split
across SIMD groups, pre-decoded scales + fma — and why a second access pattern
exists at all.

**Structure.** §12.1 what it computes — the recap · §12.2 why K and V get their
own kernel (GQA row count) · §12.3 the matrix operation · §12.4 the kernel (5
subsections: one TG per row; two SIMDs striding blocks; byte-wise lane read;
ac[8] accumulator; two-stage reduction) · §12.5 the helper (packed header
reads; pre-decoded scales; fma) · §12.6 the v4-vs-4sg contrast table + llama
attribution · §12.7 the Rust dispatch + discrepancy · §12.8 the access pattern
· §12.9 tradeoffs · §12.10 where the gap lives — not here.

**Devices worth porting.**
- **Occupancy argument (§12.2)** — GQA makes K/V 256 rows; v4 would launch 64
  threadgroups = under-occupied GPU; flip the geometry to 1 row/TG ⇒ 256 TGs.
  "Neither kernel dominates; the right one is shape-dependent" — the general
  lesson.
- **The v4-vs-4sg contrast table (Fig 12.4)** — 13 axes side by side (rows/TG,
  what SIMDs split, stride unit, nibble read width, x handling/reuse, scale
  strategy, per-element dequant, final reduction, used-for, grid). "The heart
  of the chapter."
- **Pre-decoded-scales diagram (Fig 12.3)** — d_sc[j]=d·sc_j, neg_dm[j]=−(dmin·m_j)
  computed once per super-block; inner loop uses constants.
- **fma single-rounding explanation (§12.5.3)** — fused multiply-add, no
  intermediate rounding; the dequant is one instruction; the one structural
  place 4sg buys a shorter dependency chain.
- **The discrepancy flag box (§12.7)** — dispatch launches 128 threads but the
  kernel is written for 64; tg[2]/tg[3] OOB writes; "output still comes out
  right… unverified whether intentional or a latent over-launch bug". Teaching
  readers to *find and flag* bugs in code they are quoting — peak credibility.
- **Careful attribution scoping (§12.6)** — the header asserts the geometry
  matches llama's N_SG_Q4_K=2; the *stronger* claim (llama chose it for short
  matrices) is marked unverified; what is Ferrite's own is the row-count split.
- **"Not the gap" byte math (§12.10)** — K+V = 0.42 MB/layer ≈ 11.8 MB ≈ 1.2%
  of the stream; ideal 0.26 ms of a 30 ms token.

**Ferrite-specific.** All kernel paths (matmul_q4k_core.metal, decode_all_q4k_scales_h,
encode_dispatch.rs); the 128-thread discrepancy; N_SG_Q4_K; parity sweep.

**Port disposition.** PORT-ADAPT the lesson (aspect-ratio-driven kernel
selection; deferred vs pre-decoded scales; fma; two-stage simd_sum+threadgroup
reduction) as a pattern chapter if Muser has an analogous second geometry; if
Muser's K/V run through a fused QKV or a different geometry, PORT-REWRITE the
content but keep the contrast-table device. The discrepancy-box device:
PORT-ADAPT as a standing genre — audit Muser's dispatch-vs-kernel-geometry
agreements and flag mismatches in the same voice.

**Segues.** Opens "a short companion chapter… everything about the dequant
math is inherited from Ch 11; only the access pattern is new" — explicit
deduplication contract between sibling chapters. Port this contract.

---

## 13. Chapter 13 — Rotary embeddings (RoPE) (`13-…md`)

**Teaching goal.** What position means to a transformer; the rotation trick and
its relative-position proof; the NEOX convention trap; theta; the kernel; why
RoPE is standalone on this path.

**Structure.** §13.1 what RoPE computes · §13.2 why position must be injected ·
§13.3 three ways to inject position (survey) · §13.4 the RoPE idea (pair up;
rotate; worked example head_dim=4; the magic (m−n)) · §13.5 the half-split
(NEOX) convention — read this twice · §13.6 theta_base = 1e6 · §13.7 the kernel
line by line · §13.8 the grid · §13.9 the Rust dispatch · §13.10 access pattern
+ where the gap lives · §13.11 tradeoffs (incl. the fused-QKV saga) · §13.12
what's next.

**Devices worth porting.**
- **Permutation-invariance demo (§13.2)** — "the cat sat on the mat" vs "mat
  the on sat cat the" look identical to raw attention; order-blind model makes
  gibberish; position injection is not optional. Best why-does-this-exist
  opener in the kernel chapters.
- **Three-scheme one-paragraph survey (§13.3)** — absolute-add / relative-bias /
  rotate; RoPE's pitch: relative for free without touching the attention kernel.
- **head_dim=4 worked example with theta 1e6 (§13.4.3)** — θ_0=1, θ_1=0.001;
  pos=2; [1,0,1,0] → [cos2, sin2, cos0.002, sin0.002] ≈ [−0.4161, 0.9093,
  0.999998, 0.002]; "the fast pair thrown almost onto the negative x-axis; the
  slow pair barely moved".
- **The (m−n) derivation (§13.4.4)** — full expansion of q_rot·k_rot, the two
  trig identities, everything cancels into (q0k0+q1k1)·cos((m−n)θ) +
  (q0k1−q1k0)·sin((m−n)θ); "absolute positions have vanished… relative
  position falls out for free". The mathematical climax of Part IV's opening.
- **NEOX-vs-interleaved side-by-side (Fig 13.3) + "read this twice"** — same
  math, different addressing; "must match the convention the checkpoint was
  trained with — mixing them silently corrupts the position signal" while the
  model still runs. The classic silent-failure warning.
- **The 64-clock frequency table (Fig 13.4)** — θ_i spans seven orders of
  magnitude; pair 0 full turn every 6 positions, pair 63 needs 5.1×10⁶; "RoPE
  is a bank of 64 clocks running at wildly different speeds"; aliasing/
  wavelength explanation of why large theta extends context.
- **No-pow-in-kernel pattern** — freq table built once at startup (`powf` in
  Rust), kernel does one multiply; the `_cached` naming.
- **precise::cos/sin accuracy lesson** — ties to Ch 4's fast-math contract;
  accuracy not determinism.
- **The fused-QKV fail-closed saga (§13.11)** — fused head_dim=128 kernel
  emitted repeated-`!` garbage from step 1; never root-caused, fail-closed to
  the 0.5B shape; "a verified-slow path beats an unverified-fast path that
  emits garbage"; plus the later measurement that the fused dispatch is
  actually *slower* (+4.4–6.6% for separate) — fusing is not automatically
  faster; register pressure vs dispatch overhead. Two portable lessons.
- **The dead-but-wired `Op::RopeEncode` footnote** — real variant, zero
  construction sites (grep-counted); "built for a plan that was never
  finished". The *technique* ports.
- **The route-disambiguation box** — "this book used to conflate" two
  three-matvec implementations on mutually exclusive routes; correction walked
  through with routing predicates. Models untangling quoted code from the
  wrong route.

**Ferrite-specific.** theta 1e6 for Qwen2.5; rope.metal / arena/new.rs /
encode_batch paths; the four-flag QKV-fusion taxonomy
(FERRITE_QKV_FUSED / FUSED_QKV_ROUTE / SPLIT_QKV / QKV_SEPARATE) and their A/B
numbers; 896 pairs / 28 TGs; correctness_7b.rs expected-value.

**Port disposition.** Math pedagogy (permutation invariance, survey, worked
example, (m−n) proof, convention warning, clock table): PORT-ADAPT verbatim —
re-ground theta and head_dim from Glimmer's config. Kernel: PORT-ADAPT to
Muser's rope kernel (if ferrite-lineage, cite). The fused-QKV saga:
KEEP-AS-LINEAGE as an ancestor lesson, but *re-run the experiment* on Muser's
own QKV fusion — "fused ≠ faster" is exactly the tradeoff Muser should measure
and record in its ledger. precise::trig: PORT-ADAPT (tie to Muser's fast-math
policy).

**Segues.** Opens "never seen a positional encoding before". Closes: Q and K
rotated in buffers → "before attention can use them, K and V must be written
into the paged KV cache" → Ch 14. Clean hand-off.

---

## 14. Glossary structure (so the port can extend it)

`glossary.md` = "Appendix A — Glossary". Preamble rule: every term defined in
place at first use in a chapter, then indexed here with a back-reference to the
introducing chapter; chapter writers must add new terms (STYLE §9 enforces).

**Sections (in order):** Metal / GPU · Quantization · Transformer / model ·
Performance / bandwidth · Methodology / performance analysis · Ferrite-specific.

**Entry format:** `- <a id="anchor"></a>**Term** *(optional qualifier)* — 1–4
sentence definition, usually carrying the concrete format/number (e.g. "256
elements per 144-byte super-block, ~4.5 bits/element"). Introduced in
[Ch N](chapters/NN-slug.md).` HTML anchors enable `glossary.md#term` deep links
(add for terms chapters link to); some entries cross-reference ("see
[ceiling]"); the Methodology section carries a prose preamble naming the
chapters that exercise its terms; duplicates across sections are reconciled by
cross-link.

**Port guidance:** keep the skeleton, rename the last section "Muser-specific",
add two sections the ancestor lacks — **Disaggregated lane / transport**
(handoff, mTLS, kvpack, wire format, pacing) and **Speculative decoding**
(draft, verify, acceptance, rollback, DFlash) — and keep a small "Ferrite
lineage" bucket for ancestor terms cited as lineage (Q4_K, vm-sync,
`[A18-neo]`). Preserve the anchor convention so chapter→glossary links work
from day one.

---

## 15. First-definition order (chapters 1–13) — preserve in the port

Terms as the book *first defines* them (not merely mentions), chapter by
chapter. The port should keep this ordering discipline: Metal vocabulary before
quantization before model before kernels; every term defined before use.

- **Ch 1:** parameter, weights, token, decode (vs prefill named), matvec/GEMV
  (named; derived Ch 11), FLOP, bandwidth, GB/s, DRAM (informal), quantization
  (informal), DRAM ceiling, roofline/arithmetic intensity, tg128/pp128, the
  gap, `[A18-neo]` tag, GGUF (first mention; formal Ch 5).
- **Ch 2:** GPU/thread, Metal, MSL, kernel, command buffer, compute command
  encoder, dispatch (four-line sequence), MTLDevice, MTLCommandQueue,
  MTLLibrary, MetalContext, threadgroup + threadgroup memory, SIMD group
  (simd_sum/simd_shuffle), grid, ALU, set_bytes/inline constant, encode_* vs
  dispatch_*, pipeline-by-name lookup.
- **Ch 3:** unified memory, DRAM/VRAM/PCIe (formal), SoC, storage modes
  (Shared/Private/Managed), blit, mmap, page fault, zero-copy, MTLBuffer,
  GpuBuffer, arena, TLB, page alignment (16 KB), hazard tracking, RAW/WAW/WAR
  (named; deep Ch 22), barrier (named; deep Ch 22), untracked hazards, B4
  zero-init, GpuHeap, buf_id.
- **Ch 4:** PSO, AIR, .metallib, JIT compilation, include_str!/concat!
  embedding, function constant (+ is_function_constant_defined), fast-math,
  MTLBinaryArchive/PSO cache, GpuKernels, fingerprint line, `precise::`
  namespace (used; motive deep Ch 13).
- **Ch 5:** quantization (formal), f16 bit layout, nibble, super-block,
  sub-block, scale/min, min+offset (local quantization) idea, Q4_0/Q4_1
  (comparative), Q4_K, Q6_K, Q4_K_M, block alignment.
- **Ch 6:** KV cache (quick def; deep Ch 14), attention (quick def; deep Ch
  15), Key/Value (quick), online vs offline quantization, symmetric vs
  asymmetric quantization, Q8_0, amax, DC offset, int8 quant.
- **Ch 7:** transformer, layer, hidden state/residual stream (named; deep Ch
  8), embedding, vocab/BPE, Q/K/V mental model, head, head_dim, GQA, RoPE
  (named; deep Ch 13), theta, FFN/MLP, intermediate_dim, activation, SiLU
  (formula; deep Ch 17), GELU (comparative), tied/untied embeddings, dot
  product (intro; derived Ch 11), argmax (intro; kernel Ch 20), causal mask,
  softmax (named; derived Ch 15), logits (named; deep Ch 19), LM head (named),
  SwiGLU (named; deep Ch 17).
- **Ch 8:** prefill (formal) + GEMM, residual stream (deep: the pipe),
  GpuArena/activation buffer, `Vec<Op>`/VM-exec/Op, command-buffer
  amortization, kernel fusion, vm-sync route, position ladder (SplitKVecQ8/
  AllHeadsQ8 named), paged-Q8 KV cache (named; deep Ch 14), barrier plan.
- **Ch 9:** lookup/gather, embedding-table orientation, QuantEmbedding
  (raw-bytes storage).
- **Ch 10:** normalization, LayerNorm, mean/variance/standard deviation,
  RMS/RMSNorm, gamma, epsilon, float4 vectorized load, strided loop, rsqrt,
  ULP, bit-exact path.
- **Ch 11:** matvec/GEMV (formal + derived), row-major, register caching /
  register tiling, memory-level parallelism / in-flight loads, L2 cache,
  deferred scaling, residualized matvec, GEMM (formal contrast).
- **Ch 12:** pre-decoded scales, fma (fused multiply-add), two-SIMD-group
  handoff, occupancy/under-occupation, N_SG geometry.
- **Ch 13:** positional encoding, permutation-invariant, RoPE (formal),
  rotation matrix, frequency/theta base, half-split (NEOX) vs interleaved
  (GPT-J) convention, wavelength/aliasing, cached frequency table,
  fused-vs-separate dispatch tradeoff.

Muser-only terms the port must slot in (suggested placement): NVFP4 / e4m3,
kquant / DFlash, draft & target model, speculative acceptance/rollback,
kvpack, page/block table (Ch 14 scope), Handoff V2, mTLS, pacing, TTFT
(ancestor defers to Ch 23), GB10/GX10, wire rate, parity ledger.

---

## 16. Cross-cutting port hazards (chapters 1–13)

1. **The recurring question is baked into every chapter.** STYLE §2 item 8
   mandates a "where the gap lives" section per kernel chapter, and the gap is
   Ferrite's A18 8%-vs-llama bandwidth deficit. Muser cannot inherit the
   verdicts. Establish Muser's own measured question first (see §0), then
   re-run each verdict against it — or the sections become stale ancestor
   citations and the book loses its spine.
2. **Two measurement systems in collision.** Ancestor numbers come from
   `[A18-neo]`/PERF.md/perf-log-book with the interleaved-ratio rule; Muser
   numbers must come from the parity ledger + `muser-receipt://`
   receipts under release-lock discipline. Re-tag every imported number
   (`[ferrite-book Ch N]` lineage vs `[ledger §N]`/`[receipt …]`); an untagged
   A18 number in the Muser book is a defect.
3. **Part II and the KV story are different physics on Muser.** Q4_K/Q8_0 byte
   layouts vs kquant/NVFP4 + kvpack + transported KV. Ch 5/6 pedagogy ports;
   content is a rewrite, and the KV chapter's scope expands from "a third
   memory beast" to "local store + container + wire format + precision parity
   across the handoff" — the ancestor has no notion of KV crossing a trust
   boundary.
4. **The decode-loop spine lacks Muser's two headline features.** Ch 8's
   Figure 8.1 and fusion list know nothing of speculative decoding (draft/
   verify/accept changes batch=1, adds rollback, changes the time budget) or
   disaggregated prefill (prefill written as always-local). Port the spine
   diagram last among early chapters, after dumping Muser's real dispatch
   order from source.
5. **Hardware-behavior claims are A18-specific** — single-stream DRAM saturation
   (dual +0.43%/quad −5.05%), the ~4 W fixed-power energy argument, the 7 µs
   encoder-gap estimate, the M4 compiler bug. Re-measure or label lineage.
6. **Unresolved Ferrite discrepancies must not be inherited silently** — the 4sg
   128-vs-64-thread launch discrepancy (Ch 12 §12.7) and the fused-vs-split FFN
   tension (Ch 8 §8.2) are open in the ancestor. If Muser's lineage shaders
   carry the same code, re-audit; don't quote the ancestor's shrug.

## 17. Factually fragile claims (A18- or Ferrite-behavior-dependent)

- 45.95 GB/s pure-read ceiling, 98.4% pure-read, 46.1 GB/s blit ceiling, the
  single-stream saturation table — A18 Pro measurements.
- "~5 TFLOP/s" GPU peak — explicitly `[unverified]` in the book itself.
- All throughput/energy figures: 33.20 tok/s, 0.92×, 36.9 GB/s (80.3%),
  0.104 J/tok, 3.94/4.13 W, "~4 W regardless of model size" (almost certainly
  false on a larger Mac where power tracks work).
- 77%-vs-85% (tg32 best/best) and 70%-vs-83% (production probe) framings;
  97.6% GPU-busy; 1.4% encode; 43 barriers/2.7 µs/0.4%; 6–10% kernel parity —
  Ferrite-vs-llama on A18; not portable truths.
- "~7 µs GPU idle" per encoder boundary — an unmeasured source comment.
- Model-load "seconds, not tens of seconds" — `[unverified]` in-text.
- M4 Metal compiler poisoning (dk64) — device- and OS-build-specific.
- Quality rationales the book itself marks `[unverified]`: naive-4-bit
  insufficiency, Q4_K-vs-Q4_0 quality recovery, Q6_K promotion choices,
  RMSNorm≈LayerNorm equivalence, KV error-compounding law, DC-offset impact on
  Qwen KV, theta=1e6 short-range cost. Keep the markers when porting concepts;
  do not promote them to fact.
- The 4sg dispatch thread-count discrepancy — unresolved in the ancestor.

*Discrepancies & scope:* runs ~960 lines rather than the 500–700 guide — the
per-chapter rubric (goal, full heading list, named devices, concrete
Ferrite numbers/paths, disposition, segues) plus glossary format, definition
order, hazards, and fragility does not compress further without dropping
mandated substance. Density was prioritized over brevity.

— end of part 1 (chapters 1–13). Part 2 should cover chapters 14–25 +
appendices with the same per-chapter rubric.
