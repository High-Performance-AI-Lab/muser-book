# Chapter 40 — What we measured and rejected
> **status:** polished  ·  **path:** Muse Glimmer, pinned Muser tree
>
> *Prerequisites: the whole book. This chapter is the ledger of its dead
> ends; each one assumes you know the machinery it tried to move. The load-
> bearing priors are [Ch 33](33-speculation-and-the-distributed-verdict.md),
> [Ch 35](35-ordering-hazards-and-the-dispatch-gap.md), and
> [Ch 32](32-precision-across-the-handoff.md).*

---

## 40.1 The last chapter is a graveyard

Chapter 39 ended on the evidence culture: locks, registers, receipts — the
machinery that decides what may be *said*. This final chapter is about the
saying of "no." A from-scratch engine is a pile of attractive hypotheses,
and the only reason the pile stands at the end is that most of them were
measured and killed. The public benchmarks file says it plainly, in a
section literally titled "What we measured and rejected" `[docs/benchmarks.md
§5]`.

The genre is inherited: the ancestor book closed with a falsification
ledger — hypotheses, verdicts, tombstones, and a survivor exposed as a
tautology `[ferrite-book Ch 25]`. Muser's version is stricter, because
Muser's campaign had receipts. Every entry in this chapter has five parts:

1. **The hypothesis** — what we hoped was true.
2. **The experiment** — what was actually measured, under what scope.
3. **The receipt** — where the evidence lives.
4. **The verdict** — what was concluded, in the record's own words where
   possible.
5. **What the rejection preserves** — because a well-run rejection is not a
   loss. It buys a boundary, a guard, or an insight that ships.

Read the fifth column carefully; it is the point of the chapter. Rejections
are how this engine learned where its exactness contract actually binds.

## 40.2 Rejection 1 — the linear distributed-verifier lane

**Hypothesis.** The GX10 sits mostly idle during Mac decode. Let the remote
node be the *authoritative speculative verifier*: Mac DFlash drafts, the
GB10 runs the target's verification pass on tensor cores, and the pair
beats the local 107.9 tok/s kquant spec bar `[docs/nvfp4-distributed-
speculative-frontier-20260818.md §Decision]`.

**Experiment.** A composite M16 screen first: 31 warm prefix-cached GX
Dudeman runs with all five f32 DFlash target layers copied into pinned host
memory measured **107.152 ms median** target wall; charging the already
measured Mac draft (26.9 ms), RTT (0.78 ms), and capture transport
projected **114.93 tok/s** at median — a real opening, which set the
preregistered bar: ≥ **99.151 % IID per-edge acceptance** to beat 107.9
tok/s `[frontier §Decision]`. Then four end-to-end traces through the real
authenticated lane: one positive control plus three organic content strata
(docs, python, rust).

**Receipt.** The verdict table, with per-trace receipt SHA-256 pairs
`[frontier §End-to-end linear-lane verdict]`:

| Trace | Acceptance | Measured tok/s | Verifier-only ceiling |
|---|---:|---:|---:|
| Standard (all-accept control) | 100.00 % | 110.59 | 125.61 |
| Documentation | 9.23 % | 15.53 | 20.15 |
| Python | 26.31 % | 11.17 | 40.04 |
| Rust | 38.07 % | 15.41 | 55.96 |

**Verdict.** "They reject the linear M16 candidate for general product
serving" `[frontier §End-to-end linear-lane verdict]`. The decisive number
is the **verifier-only ceiling** — output tokens divided by GX verifier
wall alone, "granting zero time to DFlash, feature decode, transport,
installation, or scheduling" — and even under those physically impossible
assumptions all three organic traces stay below 107.9 tok/s. The 110.59
figure is the control, and the claims register fixes its meaning in
wording: "We measured remote speculation across the wire and rejected it for
general serving — the verifier cost eats the gain. The shipped
disaggregated lane is fast remote prefill plus plain parity decode … Never
cite the all-accept control number as a serving result" `[claims #14]`.

**What the rejection preserves.** Three things. First, a *theorem-shaped
insight*: lossless speculative decoding does not require the drafter and
target to share a checkpoint — "It requires one endpoint to execute the
authoritative target transition. The other endpoint may use any
approximation" `[frontier §Decision]`. Second, the rejection bound itself:
any future distributed scheme must beat 20.15/40.04/55.96 tok/s *before*
it even pays for transport. Third, one live research thread, unwired: a
"hardware-aware token tree" that would turn otherwise idle GX batch
arithmetic into path coverage — "That experiment must beat the ceilings
above with measured emitted tokens per evaluated tree node" `[frontier]`.
The protocol machinery (authenticated verifier log, carried-frontier state)
is retained in `crates/muser-cluster/src/verifier*.rs` as unwired research
substrate.

## 40.3 Rejection 2 — native NVFP4 speculative decode (Fallback B)

**Hypothesis.** The native NVFP4 lane is the fast lane; give it speculative
decoding too and compound the wins.

**Experiment.** Measured directly: W4A4 batched target execution of the
speculative verify pass ran at **6.81 tok/s** against the 107.9 tok/s
kquant bar — "Of its 37.619 s decode span, target verification consumes
35.915 s (95.5 %) … each 16-row target cycle is about 2.24 s versus
128.4 ms in the L-series kquant reference" `[ledger "F-series remediation
context"]`. The W4A4 batched verify matmul — the one shape where FP4 tensor
arithmetic should shine — is the very place the lane collapses
([Ch 33](33-speculation-and-the-distributed-verdict.md)). A remediation
lane ("Fallback A": a Mac weight-only E2M1 verifier) was then built and
measured to its own no-go: best result **227.864 ms GPU** per 16-row cycle,
"still 13.9 % over the preregistered 200 ms GPU admission gate and 1.77x
the 128.400 ms kquant reference … its hard throughput ceiling is 70.2
verified rows/s, below the shipped kquant lane's 107.9 tok/s" `[ledger
"Fallback A follow-up — weight-only verifier final no-go"]`.

**Receipt.** `[receipt goal-native-spec-local-verify-v7/]` and siblings
under `goal-native-spec-*` for the local no-go; `[docs/nvfp4-fast-lane-
evidence-20260817.md]` records the 6.805 tok/s diagnostic and its scope.

**Verdict.** Fallback B, in the operator's recorded words: "Fallback B is
selected. The product ships the native NVFP4 lane without speculative
decoding: 3.881s-class disaggregated prefill, ~35.5 tok/s plain decode,
~64ms warm prefix hits, determinism-pinned seam, published drift envelope.
Speculative decoding remains kquant-lane-only at 107.9 tok/s; the native
lane's fail-closed rejection of speculative configs stays structural."
`[ledger "F-series shipping qualification amendment — Fallback B
authorization", verbatim]` (The ledger sentence continues with the Fallback A
follow-up authorization, elided here — Fallback A is measured to its own
no-go above. And read "~64ms warm prefix hits" with its scope: 64.631 ms is
the *shallow*, 2,048-token warm-hit figure `[ledger P4 cell; claims #11]`;
deep warm hits are 0.6132 s at 65,536 and 1.0566 s at 130,815 tokens, each
a single sample `[ledger "Kvpack ladder stage-5 isolated-depth verdict"]`,
[Ch 25](25-warm-reuse.md).)

**What the rejection preserves.** A *structural* guard, not a convention:
`producer_mode: native` plus DFlash fails closed at server construction
`[crates/muser-server/src/state.rs:1666-1675]` and in the qualifier
`[scripts/qualify_nvfp4_fast.py:333-336]` — the configuration cannot be
expressed, let alone measured, in a serving path. It also preserves the
interpretive lesson of [Ch 32](32-precision-across-the-handoff.md):
quantization's cost is never global. NVFP4 is parity-within-noise at plain
decode (35.491 vs 35.440 tok/s `[ledger P1.3]`) and catastrophic in the
batched verify shape; the gate exists to localize which one you are in.

## 40.4 Rejection 3 — the ANE/CoreML route

**Hypothesis.** Apple's Neural Engine (ANE) — the fixed-function
accelerator beside the GPU, programmed through Core ML — could run the
DFlash draft cheaper than Metal and lift speculative decode.

**Experiment.** A lineage of focused, target-exact POCs: split v4–v6
reached only 0.644×/0.704×/0.711× Metal (best ANE 238.637 ms vs Metal
153.681 ms on the comparable cell) `[docs/release-provenance.md, ANE POC
history]`; the final v9 fused-attention POC — warm, three repetitions,
256 tokens, identical target-token digest across reps, ANE/Metal draft
acceptance 238/259 (91.89 %) — measured ANE raw times 5.118/5.113/5.073 s
(CV 0.40 %) vs Metal 4.185/4.204/4.260 s (CV 0.75 %): "The resulting
ANE/Metal throughput ratio was 0.8266x" `[docs/release-provenance.md §ANE
v9 fused-attention POC]`.

**Receipt.** `[receipt ane-v9-fused-sg4-256x3-20260814/]` for the stable
result; earlier POC receipts retained as dated research evidence.

**Verdict.** "No v0.1 launch claim. ANE is experimental/post-release,
excluded from qualification and candidate contents, and never selected by
`auto`" `[claims #5]`. The release provenance carries the standing scope
override: public-CoreML ANE "is not a mandatory lane, release identity
input, seal member, or candidate artifact; v0.1 `auto` routing is
permanently Metal" `[docs/release-provenance.md, v0.1 scope override]`.

**What the rejection preserves.** A scope boundary that protects the claim
surface (the telemetry labels ANE counters experimental, and the metrics
schema forbids any ANE speed card `[docs/metrics-schema.md §DFlash and
optimization claims]`) — and a clean example of *rejecting on measured
ratio, not on vibes*: the route was exact and functional; it was merely
0.827× slower, so it lost. The Ferrite-lab 1.42× ANE+GPU concurrency figure
that circulates in ancestor context is explicitly quarantined as
`[precedent-7B-ferrite]`, never a Muser result `[docs/launch-claims.md
§Ground rules]`.

## 40.5 Rejection 4 — the 104-group norm-boundary fusion

**Hypothesis.** From [Ch 35](35-ordering-hazards-and-the-dispatch-gap.md):
the production decode graph carries 104 separated norm-boundary closure
groups that the legacy graph fuses. Fusing them should remove dispatch
overhead and close the (then) 22 % decode deficit.

**Experiment.** Three implementations were measured against the pinned
2,048-token fixture with full-logit hashing: the existing dual-norm fusion
(logit SHA *changed* — rejected outright); a "pinned-reduction" dual norm
reproducing llama's 32-SIMD-group reduction twice (bit-exact, 760→655
groups, 40.330→39.274 ms GPU, but historical self-consistency only — this
was before J0 made llama's bytes the gate); and a hybrid retained-activation
schedule selecting fast fused boundaries.

**Receipt.** The hybrid postmortem, quoted from the record: full-logit
maximum absolute error `4.6300888e-4`; normalized logprob maximum absolute
error `3.197146176834309e-4`, "above the `1e-4` contract"; **201,970 of
202,048 logits differed**; "the first KV difference was layer 1, value
plane element 524,115, with f16 bits 39,892 versus 39,893" `[docs/
decode-dispatch-gap-20260815.md §Rejected hybrid postmortem]`, receipts
`[receipt pinned-token-parity-20260814-v{3,4}/]`.

**Verdict.** "The 104-group fusion is not eligible regardless of its wall
sample because its logits changed" `[docs/decode-dispatch-gap-20260815.md
§Landed and rejected reductions]` — and the disposition line in the
reconciliation table is two words: "Existing fusion is not exact; reject."
The one removal that *was* exact (one last-row copy, 6,656 elements) bought
−0.136 ms GPU (−0.34 %) and no wall claim `[docs/decode-dispatch-gap-
20260815.md]`.

**What the rejection preserves.** The bit-exactness contract itself, with a
priced tombstone: any future fusion must "reproduce the standalone
reduction and store order bit for bit. The current 104-group fusion is a
negative fixture, not a candidate" `[docs/decode-dispatch-gap-20260815.md
§Ranked remaining exact work]`. One f16 ULP in a layer-1 value is enough to
lose 201,970 logits — and the culture's response was to *remove the hybrid*
rather than widen the tolerance `[docs/decode-dispatch-gap-20260815.md
§Rejected hybrid postmortem]`. This is the exactness contract of
[Ch 32](32-precision-across-the-handoff.md) enforcing itself at home, not
just across the wire.

## 40.6 Rejection 5 — full send-during-prefill streaming

**Hypothesis.** The handoff moves gigabytes after CUDA finishes; if
segments could leave *during* prefill, TTFT would drop by the overlap.

**Experiment and analysis.** The original wire schedule was tile-major with
strict ordering, and the early analysis deferred streaming: the schedule's
strict tile order meant no segment could leave before the last NoPE layer
computed — a structural property, not a tuning knob. The register keeps
the deferral on its post-launch list: "connector streaming during prefill
was analyzed and deferred (the wire schedule's strict tile order means no
segment can leave before the last NoPE layer computes)" `[docs/launch-
claims.md §Explicitly post-launch]`. The 2026-08-19 rework then found the
surgery smaller than feared — switching both sides to a layer-major
schedule made "each SWA group … sendable as soon as its 13 layers exist
mid-prefill and only the NoPE tiles trail," with a dedicated sender thread;
receipt `stream4-p4-20260819` records all 16 segments enqueued during
prefill, the first on the wire at +470 ms of a ~1.18 s prefill, and TTFT
1.596 → 1.500 s median (CV 0.14 %) `[docs/disaggregated-prefill-sealing-
plan-20260818.md §W2]`.

**Receipt.** `[receipt stream4-p4-20260819/]`; the register's post-launch
bullet and the sealing plan's W2 entry carry the analysis.

**Verdict.** Split, and honestly so: SWA-group streaming during prefill
*shipped*; full streaming remains deferred because the **NoPE bulk cannot
start before the last NoPE layer computes** — 95.7 % of a deep payload
waits for layer 51, which is exactly why a 130,815-token handoff idles the
link 41–47 s and then bursts 1.74 GB onto it `[docs/kvpack-merge-handoff-
20260820.md §6 "Pacing reality"]` (the EEE story of
[Ch 31](31-the-wire-discipline.md)). The register's conservative wording
stands: neither full streaming nor its benefits may be implied
`[docs/launch-claims.md §Explicitly post-launch]`.

**What the rejection preserves.** The dependency is now *written down as a
schedule invariant with a verifier* — "The verifier enforces the new
invariant (first segment before D2H completion) with positive and negative
tests" `[docs/disaggregated-prefill-sealing-plan-20260818.md §W2]` — and
the analysis survives as the design note for anyone who wants the NoPE
trailing edge moved: the blocker is compute order, not transport
enthusiasm.

## 40.7 Rejection 6 — remote multimodal handoff

**Hypothesis.** (Implicit in any full matrix): the disaggregated lane could
qualify a multimodal arm — images shipped to the GX10, vision prefill done
remotely like text prefill.

**Experiment.** The 2026-08-23 release-readiness attempt tried to bind the
mandatory remote packet and hit a hard boundary: the native arming wrapper
`scripts/qualify_nvfp4_fast.py` admits only `text` and
`target-plus-dflash` — `--variant multimodal` is rejected with exit 2
before touching the node (the choice set is fixed at the argument parser
`[scripts/qualify_nvfp4_fast.py:47]`) — and the live image's source-matched
request parser "accepts exactly a token-only top-level request" `[ledger
"Release preparation — native multimodal matrix blocker", 2026-08-23]`. A
direct capability probe that tried to route around it timed out waiting
for a producer the direct qualifier cannot arm, and was recorded as
**INVALID_WRONG_REASON** — "not a multimodal correctness or performance
verdict, and it was not retried" `[ledger, same entry]`. The operator then
chose the sealing plan's explicit-gating disposition: commit `df2a0f9`
appends the boundary and adds claim #17 `[ledger "Release readiness
attempt 2", 2026-08-23]`.

**Receipt.** `[receipt release-readiness-campaign-20260823/attempt-1/phase4/
native-multimodal-wrapper-commandability-attempt-1/…command.log]` and
`PHASE4_MULTIMODAL_STOP_VERDICT.json`.

**Verdict.** Claim #17 (OPERATOR REVIEW REQUIRED wording): "Multimodal
requests are served with local prefill; remote multimodal handoff is
unqualified. … Do not imply remote image transfer, remote vision prefill,
or a remotely qualified multimodal path" `[claims #17]`.

**What the rejection preserves.** The **local vision lane remains
mandatory** — it is one of the fifteen seal lanes (`vision`) and it stays
in scope `[docs/private-release.md §2]` — so the rejection costs nothing
user-facing while keeping an unqualified remote path out of the contract.
And it preserves a piece of measurement hygiene worth naming: the
INVALID_WRONG_REASON label. A run that fails for reasons outside its
hypothesis is not evidence about the hypothesis; it is retained, named,
and not retried into a fake verdict.

## 40.8 The smaller tombstones

The record holds more measured rejections than fit sections; each in one
line with its receipt:

- **Multi-stream TCP slicing** — evaluated and rejected by the W0
  measurement: "a single stream already saturates" the 9.40 Gbps link
  `[docs/disaggregated-prefill-sealing-plan-20260818.md §W1]`.
- **The universal 15 % NVFP4 quality gate** — retired after the E1
  quant-vs-quant yardstick showed disagreement bands are content- and
  depth-local (calibrated gates 8.796–15.299 %); the published form became
  a content-sensitive envelope with the docs@65,536 exceedance (15.134 % vs
  13.339 %) stated, not footnoted `[claims #10]`.
- **The Inferact NVFP4 checkpoint** — rejected after a full E2 sweep:
  worse in all 30 cells, confident flips 52/95/176 vs RedHat's 23/33/56 at
  docs 8k/16k/32k, McNemar p ≤ 5.7e-05 `[ledger "Checkpoint bake-off"]`,
  receipt `[receipt nvfp4-bakeoff-20260817/checkpoint-decision.json]`.
  Prefill keeps RedHatAI; decode keeps Dudeman.
- **Six Stage-B spec levers** (b16 tile, mul_mm, multicol, two custom
  split-K matvecs, SGM K-split) — probed exact and rejected; the L-series
  n32 tile won instead `[ledger K0, K2]`.
- **The historical 5.83× exact-mirror comparison** — retired as a product
  baseline; superseded by the accepted 3.881 s / ~6.5 s comparison and the
  4.149× EEE-off median `[claims #6]`.

A pattern should be visible by now: every tombstone is either an
exactness violation (40.5), a measured performance miss against a
preregistered bar (40.2, 40.3, 40.4), or an unqualified capability kept
out of contract by an explicit boundary (40.6, 40.7). No rejection in this
record is "seemed like a bad idea." All of them ran.

## 40.9 The recurring question, answered

This book opened with one sentence masquerading as three questions, and
promised to collect answers. It is time to pay that debt — one page, across
all eight parts, every number already cited in its chapter.

**What does one token cost?** Sixteen point seven six gigabytes of reads
and about 53 GFLOP of arithmetic that hides completely underneath them —
28.22 ms at the kquant lane's measured 35.440 tok/s, an effective read
rate of ~594 GB/s derived from that measurement
([Ch 1](01-why-inference-is-a-memory-problem.md), `[ledger P1.3]`). The
cost is memory, almost purely: the GPU finishes the math long before the
bytes finish arriving, which is why the whole engine is organized around
bytes-per-token, and why quantization buys *capacity* (4.81 bits/param)
rather than decode speed — NVFP4 lands at parity-within-noise, 35.491 vs
35.440 tok/s `[ledger P1.3]`.

**Where does the time go?** Into the weight stream, and into structure you
can name. The +196-closure dispatch gap reconciled *exactly* into 104
norm-boundary groups, 39 SWA staging groups, 52 KV-publication splits, and
one copy `[Ch 35]` — and the fusions that would remove the 104 groups
change bits (§40.5), so the time went into the exactness contract, paid
knowingly. On the wire, the time goes into a bill of charges that turned
out to be self-inflicted until proven otherwise: our own pacing pin (3.9 of
9.4 Gbps), our own ledger's fsync tail, EEE's blackouts on our own burst
schedule ([Ch 31](31-the-wire-discipline.md)). At depth, TTFT is
compute-bound locally and wire-bound remotely: 570.122 s local vs 137.405 s
remote at the 130,815 class, 4.149× `[claims #6]`.

**What may be moved without breaking the exactness contract?** Three
moves, each with its measured proof and its fence. *Into a draft model:*
DFlash speculation, exact by CPU verification against the full target
distribution, currently 1.23692× synthetic at 2,048 and — the honest edge
— 0.931 on high-acceptance shallow natural text `[claims #15]`
`[docs/benchmarks.md §2]`. *Into a cache:* kvpack warm reuse returning the
first token in 0.6132 s at 65,536 where cold is 68.6 s, bit-identical
`[claims #11]`, and delta handoff moving 54.2851 % of full bytes with
byte-equal output `[claims #12]`. *Across the wire:* the entire
disaggregated lane, anchored by the integer-dot verification producer and
the kquant reference lock ([Ch 32](32-precision-across-the-handoff.md)).
And what may *not* be moved: the anchor bytes themselves (J0), the
reduction DAGs behind them (J1), and any fusion that disturbs so much as
one f16 ULP in a layer-1 value (§40.5). The rejected lane of §40.2 marks
the far fence: even a *provably lossless* move fails if its verifier cost
eats the gain.

That is the book's answer, and notice its shape: it is not a number but a
*contract with receipts* — what moved, what it cost, what it could not
touch, and where each fact is written down.

## 40.10 Where a reader goes from here

The book ends; the program does not. The open threads, each labeled by its
evidence status in the register's own vocabulary:

- **Natural-workload gates as standing apparatus** — *enrolled requirement,
  not yet a full matrix.* The synthetic fixture certified a broken draft
  lane for an entire campaign; natural-text cells are now a standing part
  of the spec matrix, but broad natural-workload performance remains
  explicitly unclaimed `[claims #2]` `[ledger "ROOT CAUSE FOUND AND FIXED",
  consequence 2]`.
- **The unwired V2 verifier protocol and the token-tree experiment** —
  *research, unwired.* The carried-frontier protocol exists
  (`crates/muser-cluster/src/verifier_v2.rs`); the hardware-aware token
  tree must beat the measured 20.15/40.04/55.96 ceilings with "measured
  emitted tokens per evaluated tree node" `[frontier]`.
- **Scale-out beyond 1× Mac + 1× GX10** — *roadmap, must not be implied.*
  "1x Mac + 1x GX10 today. Do not imply a multi-GX10 cluster is running;
  scale-out is roadmap" `[claims #8]`; no multi-producer scheduler, no
  node discovery, no revocation flow yet `[docs/launch-claims.md
  §Explicitly post-launch]`.
- **Raw-dispatch attribution (the xctrace class of question)** — *planned,
  not run at the pin.* "A future `.gputrace` capture should count raw
  kernel dispatches and encoder CPU cost" `[docs/decode-dispatch-gap-
  20260815.md §Ranked remaining exact work]`; the ancestor's equivalent
  experiment was attempted and blocked for named, non-privilege reasons
  `[ferrite-book Ch 25]` — the question survives both trees.
- **Sustained deep-load stability** — *open finding.* One producer died on
  the ninth consecutive deep handoff during the EEE-off sequence; the
  eight-handoff soak passed, but "sustained-deep-load stability remains
  open" `[claims #13]`.
- **Deep reuse coverage** — *partial by record.* Warm reuse measured at
  65,536/130,815, delta at 65,536; "reuse and delta are not measured at
  every depth, and every deep multimodal cell remains unrun" `[docs/
  launch-claims.md §Explicitly post-launch]`.

And the meta-lesson to carry into whatever you build next: the answer to
"what does one token cost?" rots the moment it leaves its receipt. Keep the
ledger append-only, keep the gate ahead of the benchmark, keep the lock
ahead of the claim — and when a beautiful number arrives from an
all-accepting control, a half-windowed draft, or an asymmetric clock,
write the tombstone yourself before someone else has to.

That is how you write an inference engine. You measure it until it tells
you the truth, you reject everything that fails to, and you leave the
receipts where the next person can find them.

---

## References

- `[frontier]` — `docs/nvfp4-distributed-speculative-frontier-20260818.md`:
  §Decision (checkpoint-unification insight, the 114.93 tok/s screen and
  the 99.151 % bar), §End-to-end linear-lane verdict (the four-trace
  table, verifier-only ceilings, receipt hashes).
- `[claims #4]`, `[claims #5]`, `[claims #6]`, `[claims #8]`, `[claims #10]`,
  `[claims #11]`, `[claims #12]`, `[claims #13]`, `[claims #14]`,
  `[claims #17]` — `docs/launch-claims.md` rows (OPERATOR REVIEW status
  stated where present).
- `[ledger …]` — F-series remediation context and Fallback A no-go;
  Fallback B verbatim authorization; "ROOT CAUSE FOUND AND FIXED";
  "Checkpoint bake-off"; "Release preparation — native multimodal matrix
  blocker"; "Release readiness attempt 2".
- `[docs/decode-dispatch-gap-20260815.md]` — §Landed and rejected
  reductions; §Rejected hybrid postmortem; §Ranked remaining exact work.
- `[docs/release-provenance.md]` — v0.1 scope override; §ANE v9
  fused-attention POC (0.8266×, three-rep packet); earlier ANE POC lineage.
- `[docs/disaggregated-prefill-sealing-plan-20260818.md]` — §W0 (9.40 Gbps,
  multi-stream rejection), §W1, §W2 (layer-major streaming receipt), the
  2026-08-23 multimodal amendment.
- `[docs/kvpack-merge-handoff-20260820.md §6]` — the NoPE-trailing-edge
  pacing analysis behind §40.6.
- `[docs/private-release.md]` — the fifteen mandatory lanes (the `vision`
  lane stays).
- `[crates/muser-server/src/state.rs:1666-1675]`,
  `[scripts/qualify_nvfp4_fast.py:47, 333-336]` — the fail-closed guards of
  §40.3 and §40.7.
- `[crates/muser-cluster/src/verifier_v2.rs]` — the unwired carried-frontier
  protocol of §40.10.
- `[receipt …]` — under `muser-receipt://`:
  `goal-native-spec-*/`, `ane-v9-fused-sg4-256x3-20260814/`,
  `pinned-token-parity-20260814-v{3,4}/`, `stream4-p4-20260819/`,
  `nvfp4-bakeoff-20260817/checkpoint-decision.json`,
  `release-readiness-campaign-20260823/attempt-1/phase4/…`.
- `[ferrite-book Ch 25]` — the ancestor's falsification-ledger closer this
  chapter ports; its numbers are A18-class ancestor context, never Muser
  results.
- [glossary](../glossary.md) — terms introduced this chapter:
  falsified-hypothesis ledger, all-accept control, verifier-only ceiling,
  Fallback B, INVALID_WRONG_REASON, preregistered bar.
