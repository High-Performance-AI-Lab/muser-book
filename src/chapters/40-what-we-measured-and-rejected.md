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

So the question here is not "what shipped?" The preceding chapters answered
that at length. The question is the harder one behind it: how did we find
out where the exactness contract actually binds? Not by reasoning about it.
We walked into it — repeatedly, at speed, with the instruments running —
and each collision left a mark you can still read. This chapter is those
marks.

The genre is inherited: the ancestor book closed with a falsification
ledger — hypotheses, verdicts, tombstones, and a survivor exposed as a
tautology `[ferrite-book Ch 25]`. Muser's version is stricter, because
Muser's campaign had receipts. Each rejection below runs in the same five
beats, and it is worth having them in mind before the first one starts:

1. **The hypothesis** — what we hoped was true.
2. **The experiment** — what was actually measured, under what scope.
3. **The receipt** — where the evidence lives.
4. **The verdict** — what was concluded, in the record's own words where
   possible.
5. **What the rejection preserves** — because a well-run rejection is not a
   loss. It buys a boundary, a guard, or an insight that ships.

Read the fifth beat carefully; it is the point of the chapter. A rejection
that leaves nothing behind was a waste of a week. A rejection that leaves a
guard, a bound, or a named lesson is the cheapest engineering there is: you
paid once, in measurement, and the boundary holds for everyone who comes
after.

## 40.2 Rejection 1 — the linear distributed-verifier lane

Start with the most tempting idea of the whole campaign, because it is the
one that took longest to kill. During Mac decode the GX10 sits mostly idle:
a tensor-core machine watching a laptop work. So make the remote node the
*authoritative speculative verifier* — Mac DFlash drafts, the GB10 runs the
target's verification pass on tensor cores, and the pair beats the local
107.9 tok/s kquant spec bar `[docs/nvfp4-distributed-
speculative-frontier-20260818.md §Decision]`. On paper that is free money:
idle silicon, a draft model we already have, a link we already trust.

We screened before we built. The composite M16 screen took 31 warm
prefix-cached GX Dudeman runs with all five f32 DFlash target layers copied
into pinned host memory and measured a **107.152 ms median** target wall.
Charging the already measured Mac draft (26.9 ms), the RTT (0.78 ms), and
the capture transport against that median projected **114.93 tok/s** at
median — a real opening. The same screen fixed the price of entry, written
down before any end-to-end run: the lane needed ≥ **99.151 % IID per-edge
acceptance** to beat 107.9 tok/s `[frontier §Decision]`. Sit with that
figure a moment, because it is the whole story in advance. It demands that
the drafter be right on essentially every edge — and acceptance is a
property of the *content* being generated, not of the hardware. That is
exactly why the traces below are split by content type.

So we ran the real thing: four end-to-end traces through the authenticated
lane, one positive control plus three organic content strata (docs, python,
rust). We expected the control at the top, the organic strata somewhere
beneath it, and the interesting question to be whether "beneath" still
cleared the bar. Here is what came back, each row carrying its own
receipt SHA-256 pair `[frontier §End-to-end linear-lane verdict]`:

| Trace | Acceptance | Measured tok/s | Verifier-only ceiling |
|---|---:|---:|---:|
| Standard (all-accept control) | 100.00 % | 110.59 | 125.61 |
| Documentation | 9.23 % | 15.53 | 20.15 |
| Python | 26.31 % | 11.17 | 40.04 |
| Rust | 38.07 % | 15.41 | 55.96 |

The control row is the all-accept case, and it is beautiful: 110.59 tok/s,
past the bar with room to spare. Real text is nowhere near it. Acceptance on
organic content did not miss the preregistered demand by a few points; it
missed by a factor, and measured throughput went with it.

The record's verdict is flat — "They reject the linear M16 candidate for
general product serving" `[frontier §End-to-end linear-lane verdict]` — but
the sentence is not what closed the lane. The last column did. The
**verifier-only ceiling** divides output tokens by GX verifier wall alone,
"granting zero time to DFlash, feature decode, transport, installation, or
scheduling"; it is the score the lane would post if everything except the
remote verifier were free and instantaneous. Even under those physically
impossible assumptions, all three organic traces stay below 107.9 tok/s.
That is what makes this a tombstone rather than a to-do. The lane does not
lose to overhead we could go and optimize. It loses to arithmetic.

One beautiful number therefore needed a leash, and the claims register put
one on it in wording: "We measured remote speculation across the wire and
rejected it for general serving — the verifier cost eats the gain. The
shipped disaggregated lane is fast remote prefill plus plain parity decode …
Never cite the all-accept control number as a serving result"
`[claims #14]`.

What did the dead lane leave behind? Three things, and the first outlives
the lane by a distance. It is a *theorem-shaped insight*: lossless
speculative decoding does not require the drafter and target to share a
checkpoint — "It requires one endpoint to execute the authoritative target
transition. The other endpoint may use any approximation"
`[frontier §Decision]`. That is worth restating in flatter words, because it
contradicts what most people assume when they hear "speculative decoding":
the two models do not have to match. Exactly one side must own the
authoritative step; the other may be any approximation you like, and the
output is still lossless. Second, the rejection left a bound: any future
distributed scheme must beat 20.15/40.04/55.96 tok/s *before* it even pays
for transport. Third, it left one live research thread, deliberately
unwired — a "hardware-aware token tree" that would turn otherwise idle GX
batch arithmetic into path coverage, admitted only on a standing condition:
"That experiment must beat the ceilings above with measured emitted tokens
per evaluated tree node" `[frontier]`. The protocol machinery (authenticated
verifier log, carried-frontier state) was kept rather than deleted, in
`crates/muser-cluster/src/verifier*.rs`, as unwired research substrate.

## 40.3 Rejection 2 — native NVFP4 speculative decode (Fallback B)

Remote verification lost to arithmetic, so the obvious next move is to keep
speculation at home and put it on the fastest local lane we own. The native
NVFP4 lane *is* that lane; give it speculative decoding too and the two wins
should compound. Nothing about that reasoning is careless. It is simply
wrong, and the first measurement said so.

We ran the W4A4 batched target execution of the speculative verify pass
directly. It came back at **6.81 tok/s** against the 107.9 tok/s kquant
bar — not a regression, a collapse — and the ledger records where the time
went: "Of its 37.619 s decode span, target verification consumes 35.915 s
(95.5 %) … each 16-row target cycle is about 2.24 s versus 128.4 ms in the
L-series kquant reference" `[ledger "F-series remediation
context"]`. Now look at *which* operation ate the span, because that is the
surprise. The W4A4 batched verify matmul is the one shape in the engine
where FP4 tensor arithmetic ought to shine — many rows at once, against
low-precision weights — and it is the very place the lane collapses
([Ch 33](33-speculation-and-the-distributed-verdict.md)).

A collapse localized to one shape is a fork, not an ending. If batched W4A4
verification is what breaks, verify in some other precision: we built
"Fallback A", a Mac weight-only E2M1 verifier, and measured it to its own
no-go. The best result was **227.864 ms GPU** per 16-row cycle — "still
13.9 % over the preregistered 200 ms GPU admission gate and 1.77x the
128.400 ms kquant reference … its hard throughput ceiling is 70.2 verified
rows/s, below the shipped kquant lane's 107.9 tok/s" `[ledger
"Fallback A follow-up — weight-only verifier final no-go"]`. Much closer,
and still on the wrong side of a gate that had been written down before the
attempt began. That ordering is what turns a near miss into a verdict
instead of a negotiation. We kept the evidence for
both: `[receipt goal-native-spec-local-verify-v7/]` and its siblings under
`goal-native-spec-*` hold the local no-go, and `[docs/nvfp4-fast-lane-
evidence-20260817.md]` records the 6.805 tok/s diagnostic with its scope.

With both remediation lanes measured out, the decision became a choice
between fallbacks rather than a retreat, and the operator recorded it word
for word: "Fallback B is selected. The product ships the native NVFP4 lane
without speculative decoding: 3.881s-class disaggregated prefill, ~35.5
tok/s plain decode, ~64ms warm prefix hits, determinism-pinned seam,
published drift envelope. Speculative decoding remains kquant-lane-only at
107.9 tok/s; the native lane's fail-closed rejection of speculative configs
stays structural."
`[ledger "F-series shipping qualification amendment — Fallback B
authorization", verbatim]`

Two things about that quotation, both of which matter more than they look.
The ledger sentence continues into the Fallback A follow-up authorization,
elided here — and that follow-up is the no-go measured above, so the cut
hides nothing. More importantly, "~64ms warm prefix hits" must be read with
its scope attached: 64.631 ms is the *shallow*, 2,048-token warm-hit figure
`[ledger P4 cell; claims #11]`. Deep warm hits are a different animal
altogether: 0.6132 s for 65,536 tokens and 1.0566 s for 130,815 tokens, each
of them a single sample
`[ledger "Kvpack ladder stage-5 isolated-depth verdict"]`,
[Ch 25](25-warm-reuse.md). A round figure with a tilde in front of it is
exactly the kind of thing that walks out of a ledger and into a slide, so
the scope travels with it here.

What the rejection preserves is a *structural* guard rather than a
convention. `producer_mode: native` together with DFlash fails closed at
server construction `[crates/muser-server/src/state.rs:1666-1675]`, and
again in the qualifier `[scripts/qualify_nvfp4_fast.py:333-336]`: the
configuration cannot be expressed, let alone measured, in a serving path.
Nobody downstream has to remember why the combination is bad, because the
machine will not let them rediscover it in production.

It preserves an interpretive lesson too, the one from
[Ch 32](32-precision-across-the-handoff.md): quantization's cost is never
global. The same NVFP4 weights are parity-within-noise at plain decode
(35.491 vs 35.440 tok/s `[ledger P1.3]`) and catastrophic in the batched
verify shape. Put the claim the other way around, since this is the idea
readers most often carry away broken: quantization does not make a model
uniformly worse, it makes particular *shapes* worse. The gate exists to
localize which shape you are standing in.

## 40.4 Rejection 3 — the ANE/CoreML route

This is the rejection that felt least like a failure, because nothing about
it ever broke. Apple ships a second accelerator beside the GPU: the Neural
Engine (ANE), a fixed-function unit programmed through Core ML. The draft
model is the part of speculation you pay for on every token, accepted or
not, so if the ANE could run the DFlash draft more cheaply than Metal, the
whole speculative lane gets faster for nothing. That was the question, and
it is a question only a stopwatch can answer.

The answer arrived slowly, in a lineage of focused, target-exact POCs. The
split v4–v6 generation reached only 0.644×/0.704×/0.711× of Metal — the best
ANE cell at 238.637 ms against Metal's 153.681 ms on the comparable cell
`[docs/release-provenance.md, ANE POC
history]`. Read as a series rather than a verdict, those three ratios are
encouraging: each revision closed part of the gap, and extrapolation is
seductive. So we ran one more, and we ran it the way you run an experiment
you intend to believe. The v9 fused-attention POC was warm, three
repetitions, 256 tokens, with an identical target-token digest across reps
and ANE/Metal draft acceptance of 238/259 (91.89 %). It produced the
steadiest numbers of the whole lineage — ANE raw times 5.118/5.113/5.073 s
at CV 0.40 %, against Metal's 4.185/4.204/4.260 s at CV 0.75 % — and a
result needing no interpretation at all: "The resulting ANE/Metal throughput
ratio was 0.8266x" `[docs/release-provenance.md §ANE
v9 fused-attention POC]`. Exact, stable, reproducible, and slower.

We kept the receipt for the stable result —
`[receipt ane-v9-fused-sg4-256x3-20260814/]` — and the earlier POC receipts
are retained too, as dated research evidence.

Because the route worked, the disposition had to be a scope decision rather
than a bug report: "No v0.1 launch claim. ANE is experimental/post-release,
excluded from qualification and candidate contents, and never selected by
`auto`" `[claims #5]`. The release provenance carries the standing override
beside it — public-CoreML ANE "is not a mandatory lane, release identity
input, seal member, or candidate artifact; v0.1 `auto` routing is
permanently Metal" `[docs/release-provenance.md, v0.1 scope override]`.

Two things survive the decision. One is a boundary that protects the claim
surface: telemetry labels ANE counters experimental, and the metrics schema
forbids any ANE speed card outright `[docs/metrics-schema.md §DFlash and
optimization claims]`, so nobody can accidentally publish a chart of a lane
that lost. The other is the example itself, which is why this section is in
the book at all. This is a rejection on a measured ratio, not on taste: the
route was exact and functional, it was merely 0.827× slower, and that was
the entire argument. Holding that line matters most when the ancestor
context is loud — the Ferrite-lab 1.42× ANE+GPU concurrency figure that
circulates in that lineage is quarantined as `[precedent-7B-ferrite]`, an
A18-class ancestor measurement and never a Muser result
`[docs/launch-claims.md
§Ground rules]`.

## 40.5 Rejection 4 — the 104-group norm-boundary fusion

This one comes straight out of
[Ch 35](35-ordering-hazards-and-the-dispatch-gap.md), which left a number
sitting on the table. The production decode graph carries 104 separated
norm-boundary closure groups where the legacy graph fuses them, and the
decode deficit at the time stood at 22 %. Fuse the boundaries, remove the
dispatch overhead, close a good part of the deficit. It is the most
attractive class of optimization there is, because it appears to change no
math at all — only *when* the math is scheduled.

That appearance is the trap, and it took three implementations to see the
bottom of it. All three were measured the same way: against the pinned
2,048-token fixture, with full-logit hashing.

The first was the existing dual-norm fusion, already written and waiting.
Its logit SHA *changed*, so it was rejected outright; the wall sample was
never worth reading.

The second was more careful. A "pinned-reduction" dual norm reproduced
llama's 32-SIMD-group reduction twice instead of reorganizing it, and it
worked: bit-exact, 760→655 groups, 40.330→39.274 ms GPU. For a while this
looked like the answer. It was not, and the reason is project history rather
than arithmetic — the exactness it demonstrated was self-consistency with
our own earlier bytes, established before J0 made llama's bytes the gate.
Being exact against the wrong reference is not being exact.

The third was a hybrid retained-activation schedule, selecting the fast
fused boundaries only where they looked safe. We expected small, bounded
drift in exchange for real time. What came back is the most instructive
postmortem in the whole record: full-logit maximum absolute error
`4.6300888e-4`; normalized logprob maximum absolute error
`3.197146176834309e-4`, "above the `1e-4` contract"; **201,970 of 202,048
logits differed**. And then the line that explains all of them: "the
first KV difference was layer 1, value plane element 524,115, with f16 bits
39,892 versus 39,893" `[docs/
decode-dispatch-gap-20260815.md §Rejected hybrid postmortem]`, receipts kept
at `[receipt pinned-token-parity-20260814-v{3,4}/]`.

Follow that chain slowly, because it is the lesson of the section. A single
f16 value, one layer deep, differed by one representable step — and 201,970
logits moved with it. The hybrid did not introduce error; it introduced a
different *rounding order*, and a transformer is a long enough amplifier to
carry one such step out to nearly every logit in the vocabulary by the time
the stack ends.

The verdict was written to be unarguable: "The 104-group fusion is not
eligible regardless of its wall sample because its logits changed"
`[docs/decode-dispatch-gap-20260815.md
§Landed and rejected reductions]`. The disposition line in the
reconciliation table is two words long: "Existing fusion is not exact;
reject." What survived the whole exercise was one exact removal — a single
last-row copy of 6,656 elements — worth −0.136 ms GPU (−0.34 %) and no wall
claim at all `[docs/decode-dispatch-gap-
20260815.md]`.

So the rejection preserves the bit-exactness contract itself, and prices it.
Any future fusion must "reproduce the standalone reduction and store order
bit for bit. The current 104-group fusion is a negative fixture, not a
candidate" `[docs/decode-dispatch-gap-20260815.md
§Ranked remaining exact work]`. Notice which way the tension resolved when a
fast, nearly-correct schedule met a tolerance it just missed: the hybrid was
removed, and the tolerance was left where it was
`[docs/decode-dispatch-gap-20260815.md
§Rejected hybrid postmortem]`. That is the exactness contract of
[Ch 32](32-precision-across-the-handoff.md) enforcing itself at home — on
our own machine, against our own optimization — and not only across the wire
where someone else's kernels are the suspect.

## 40.6 Rejection 5 — full send-during-prefill streaming

The handoff moves gigabytes only once CUDA has finished. So why must the
segments wait? If they could leave *during* prefill, TTFT would fall by
whatever overlap we managed to buy. Everyone asks this question eventually.
We asked it early, answered "no", and then had to answer it a second time —
and the second answer is why this entry is here.

The first answer was structural rather than lazy. The original wire schedule
was tile-major with strict ordering, and strict tile order means no segment
can leave before the last NoPE layer has computed. That is a property of the
schedule, not a knob, so streaming was deferred and the register put the
deferral in the open where it could be argued with: "connector streaming
during prefill was analyzed and deferred (the wire schedule's strict tile
order means no segment can leave before the last NoPE layer computes)"
`[docs/launch-
claims.md §Explicitly post-launch]`.

The second time round, the thing that got questioned was the schedule
itself. The 2026-08-19 rework switched both sides to a layer-major order,
and the surgery proved far smaller than anyone had feared: group by layer
instead of by tile and "each SWA group … sendable as soon as its 13 layers
exist mid-prefill and only the NoPE tiles trail," carried by a dedicated
sender thread. The receipt shows it running — all 16 segments enqueued
during prefill, the first on the wire at +470 ms of a ~1.18 s prefill, TTFT
1.596 → 1.500 s median at CV 0.14 % `[docs/disaggregated-prefill-sealing-
plan-20260818.md §W2]`. That run is retained as
`[receipt stream4-p4-20260819/]`, with the register's post-launch bullet and
the sealing plan's W2 entry carrying the analysis on either side of it.

The verdict is therefore split, and labelled that way on purpose. SWA-group
streaming during prefill *shipped*. Full streaming stays deferred, for the
original reason now narrowed to the place it actually lives: the **NoPE bulk
cannot start before the last NoPE layer computes**. At depth that is not a
footnote. 95.7 % of a deep payload waits on layer 51, which is precisely why
a 130,815-token handoff idles the link for 41–47 s and then bursts 1.74 GB
onto it `[docs/kvpack-merge-handoff-
20260820.md §6 "Pacing reality"]` — the EEE story of
[Ch 31](31-the-wire-discipline.md). The register's conservative wording
stands unchanged: neither full streaming nor its benefits may be implied
`[docs/launch-claims.md §Explicitly post-launch]`.

The half that did not ship left the more durable artifact. The dependency is
now written down as a schedule invariant with a verifier standing behind
it — "The verifier enforces the new invariant (first segment before D2H
completion) with positive and negative tests"
`[docs/disaggregated-prefill-sealing-plan-20260818.md §W2]` — so the
property cannot quietly regress the next time somebody reorders the wire.
And the analysis survives as the design note for whoever wants the NoPE
trailing edge moved: the blocker is compute order, not transport
enthusiasm.

## 40.7 Rejection 6 — remote multimodal handoff

This hypothesis is the one nobody bothered to write down, which is precisely
what a full matrix is for: if text prefill can be done remotely, why not
vision? Ship the images to the GX10, run vision prefill there like any other
prefill, and the disaggregated lane grows an arm.

The 2026-08-23 release-readiness attempt set out to bind the mandatory
remote packet and walked into a hard boundary — two of them, stacked. The
native arming wrapper `scripts/qualify_nvfp4_fast.py` admits only `text` and
`target-plus-dflash`, so `--variant multimodal` is rejected with exit 2
before the wrapper ever touches the node; the choice set is fixed right at
the argument parser `[scripts/qualify_nvfp4_fast.py:47]`. Behind that, the
live image's source-matched request parser "accepts exactly a token-only
top-level request" `[ledger
"Release preparation — native multimodal matrix blocker", 2026-08-23]`. The
capability was not slow, and it was not lossy. It was not expressible.

Then comes the part of this story worth stealing. We tried a direct
capability probe to route around the wrapper, and it timed out waiting for a
producer the direct qualifier cannot arm. That run could easily have been
filed as evidence about remote multimodal handoff — it *looks* like the
thing under test failing. It was recorded instead as
**INVALID_WRONG_REASON**: "not a multimodal correctness or performance
verdict, and it was not retried" `[ledger, same entry]`. With no valid
measurement to be had, the operator took the sealing plan's explicit-gating
disposition: commit `df2a0f9` appends the boundary and adds claim #17
`[ledger "Release readiness
attempt 2", 2026-08-23]`. The evidence for the stop is retained whole, at
`[receipt release-readiness-campaign-20260823/attempt-1/phase4/
native-multimodal-wrapper-commandability-attempt-1/…command.log]` and
`PHASE4_MULTIMODAL_STOP_VERDICT.json`.

Claim #17 carries OPERATOR REVIEW REQUIRED wording, and it draws the line
exactly where the measurement stopped: "Multimodal requests are served with
local prefill; remote multimodal handoff is unqualified. … Do not imply
remote image transfer, remote vision prefill, or a remotely qualified
multimodal path" `[claims #17]`.

So what does this rejection cost? Nothing a user can see. The **local vision
lane remains mandatory** — `vision` is one of the fifteen seal lanes and it
stays in scope `[docs/private-release.md §2]` — so an unqualified remote
path is held out of the contract without a capability leaving the product.
What it preserves beyond that is a piece of measurement hygiene worth
carrying into your own work: the INVALID_WRONG_REASON label. A run that
fails for reasons outside its own hypothesis is not evidence about that
hypothesis. It is retained, named for what it actually was, and not retried
into a verdict it never earned.

## 40.8 The smaller tombstones

The record holds more measured rejections than fit into sections of their
own. What follows is deliberately compressed — inventory rather than story,
because the stories differ only in the apparatus. Every one of them ran, and
every one kept its receipt:

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

A pattern should be visible by now, and it is worth saying out loud, because
it is this chapter's actual argument. Every tombstone here is one of exactly
three things: an exactness violation (40.5), a measured performance miss
against a preregistered bar (40.2, 40.3, 40.4), or an unqualified capability
kept out of contract by an explicit boundary (40.6, 40.7). No rejection in
this record is "seemed like a bad idea." All of them ran, and each had a
number before it had a verdict.

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
— 0.931 on high-acceptance shallow natural text; the record states the win
and the loss in the same breath `[claims #15]`
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

The book ends; the program does not. What follows are the open threads, each
labeled by its evidence status in the register's own vocabulary. Read the
labels as carefully as the threads themselves: *enrolled requirement*,
*research, unwired*, *roadmap* and *open finding* grant very different
permissions to whoever picks the work up, and keeping them distinct is what
stops an open question from quietly becoming a claim.

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
