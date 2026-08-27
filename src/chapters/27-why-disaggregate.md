# Chapter 27 — Why disaggregate prefill and decode
> **status:** polished  ·  **path:** Muse Glimmer, pinned Muser tree
>
> *Prerequisites: [Ch 1](01-why-inference-is-a-memory-problem.md) (the
> bandwidth wall and the four levers), [Ch 7](07-nvfp4-native-lane.md)
> (NVFP4), [Ch 22](22-the-price-of-context.md) (KV bytes per token),
> [Ch 26](26-delta-handoff-and-migration.md) (delta handoff). This is the
> first chapter of Part VI.*

---

## 27.1 Where we are: KV as a movable asset

[Ch 26](26-delta-handoff-and-migration.md) ended with the delta handoff: with
32,768 of 65,536 prompt tokens already held, the wire moved 54.2851 % of the
full payload and the decoded output was exactly the full-handoff reference
`[claims #12]`. The lesson generalized quietly: **the KV cache is not a
private data structure of the machine that computed it — it is an asset that
can be moved, stored, and resumed**, bit-exactly, somewhere else.

This part of the book takes that lesson to its logical end. If KV can move,
then the machine that *computes* prefill and the machine that *uses* the
result need not be the same machine. That is the disaggregated lane. This
chapter is the argument for why you would bother — an argument that returns,
at systems scale, to the memory-bound case [Ch 1](01-why-inference-is-a-memory-problem.md)
built for one token.

The book's standing question — *what does one token cost, where does the
time go, and what may be moved without breaking the exactness contract?* —
now has a fourth clause in play: *moved across the wire.* Everything in
Part VI is about paying for that clause honestly.

## 27.2 Two regimes, stated precisely

You have met the two regimes before, briefly
([Ch 1 §1.2](01-why-inference-is-a-memory-problem.md)); here they are as the
engineering facts the whole lane rests on.

**Decode** — generating tokens one at a time. Each token does one matvec
against every weight matrix ([Ch 13](13-the-qkv-gate-matvec-family.md) is the
hero example). One multiply-add per weight byte read; ~53 GFLOP of arithmetic
against ~16.76 GB of reads; arithmetic intensity ~3.2 FLOPs per byte
(derived in [Ch 1 §1.3](01-why-inference-is-a-memory-problem.md)). **Decode
is bandwidth-bound, serial, and proportional to the number of generated
tokens.** You cannot make it faster by having more arithmetic units sitting
idle.

**Prefill** — processing the whole prompt before the first token can be
emitted. Here the same weight matrices are multiplied by *many* rows at
once: the engine chunks prompts into 512-position batches
(`PREFILL_BATCH_TOKENS = 512`,
`[crates/muser-engine/src/decode.rs:53]`), and each weight byte read from
DRAM is *reused across the whole chunk* — the module doc says it plainly:
"prefill of T tokens ≈ one token's DRAM traffic"
`[crates/muser-engine/src/prefill.rs:6-8]`. The arithmetic per byte
therefore scales with the batch. **Prefill is compute-bound, parallel, and
proportional to the prompt length.**

Same weights, same kernels' worth of math, opposite bottleneck. That
asymmetry is the entire subject of this chapter.

### The roofline, with both workload points

[Ch 1](01-why-inference-is-a-memory-problem.md) put decode on one side of
the machine's roofline — the "balance point" where a workload's FLOPs-per-
byte exactly matches the machine's FLOPs-per-byte of bandwidth. With the
~800 GB/s memory class `[ledger L0]` and the ~2.6 TFLOP/s of FP32 that keeps
ALUs busy at decode intensity (both derivations from
[Ch 1 §1.3](01-why-inference-is-a-memory-problem.md)), the balance point
sits at ≈ 3.2 FLOPs/byte. Now put both regimes on the same chart:

```text
   arithmetic intensity (FLOPs per byte read), log scale

   10^4 ┤                                            ● prefill, 512-chunk
        │                                            │ ≈ 1,619 F/B
        │                                            │ (derived below)
   10^3 ┤
        │
   10^2 ┤
        │                     roofline: compute ceiling
        │                    ╱  (need ~1.3 PFLOP/s FP32 to feed a
        │                   ╱   512-chunk at 800 GB/s — no Mac has that)
    10 ┤                  ╱
        │                ╱     ← machine balance point ≈ 3.2 F/B
     4 ┤ ─ ─ ─ ─ ─ ─ ─╱─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        │            ╱
     1 ┤ ● decode ≈ 3.2 F/B
        │ │      bandwidth-bound      compute-bound
        │ └ the entire token time is the weight read   (intensity → right)
      └─┴────────┴──────────────────────────────────────────────
        1 B/token        ~5 B/token                ~1,000+ B/token
```
*Figure 27.1: The roofline flip, ported from the Ferrite book's Ch 23 device
`[ferrite-book Ch 23]` and recomputed for Muse Glimmer on the M3 Ultra. The
ancestor's marked points were decode ~3.6 vs prefill ~455 FLOPs/byte at
batch 128 on an A18 Pro — Ferrite-lineage numbers that do not transfer; the
method does.*

Derive the prefill point yourself, the way [Ch 1](01-why-inference-is-a-memory-problem.md)
derived the decode one. Per token, the matmul parameter count is 26.5 × 10⁹
and per-token FLOPs ≈ 2 × 26.5e9 ≈ 53 GFLOP (the arithmetic in
[Ch 1 §1.3 step 4](01-why-inference-is-a-memory-problem.md)). A 512-token
chunk therefore costs:

```text
FLOPs per chunk  ≈ 53.0e9 × 512        ≈ 2.714e13 FLOP
bytes read       ≈ 16.757e9            (each weight byte read once, reused)
intensity        ≈ 2.714e13 / 16.757e9 ≈ 1,619 FLOPs / byte
```

Roughly **500× further right than decode**. On the compute side of the
roofline, the only way to go faster is more arithmetic per second — or
cheaper arithmetic per FLOP. A Mac's FP32 units are what they are; a GPU
with FP4 **[tensor cores](../glossary.md#tensor-core)** (matrix-multiply
units that consume 4-bit operands natively; see
[Ch 7](07-nvfp4-native-lane.md) for the NVFP4 format) offers both. Hold that
thought for two more sections.

### The measured shape of the two regimes on one Mac

The roofline predicts the measured behavior. Local decode is 35.440 tok/s
(kquant, CV 0.037 %, its full scope in
[Ch 1 §1.3](01-why-inference-is-a-memory-problem.md)) — 28.22 ms per token,
dominated by the weight stream. Local *prefill* per prompt token is much
cheaper — weight reads amortize across the chunk — but it is pinned against
the machine's *compute* ceiling instead. Do the arithmetic with the deep
cell: 570.122 s mean for a 131,008-token prompt
`[claims #6, local baseline]` is

```text
570.122 s / 131,008 tokens ≈ 4.35 ms per prompt token
53.0e9 FLOP/token / 4.35e-3 s ≈ 12.2 TFLOP/s sustained
```

Twelve TFLOP/s of sustained arithmetic on this machine's FP32-class
throughput — the compute-bound regime, hit at depth, on one Mac. (The 12.2
figure is an arithmetic derivation from the measured 570.122 s cell, not an
instrumented compute measurement; the honest roofline conclusion survives
the derivation's roughness by two orders of magnitude.)

Now look at what the *user* experiences in that regime. Nothing. For nine
and a half minutes, nothing. That is the TTFT cliff.

## 27.3 The TTFT cliff at depth

**TTFT** — time to first token — is the latency from "prompt submitted" to
"first generated token visible." At shallow depth on one Mac it is fine. At
depth it is catastrophic, and it is catastrophic *precisely because prefill
is proportional to prompt length and compute-bound on silicon shaped for
bandwidth-heavy decode*.

The measured cliff, local lane, five exact-token reps per depth
`[docs/benchmarks.md §3]` `[ledger "Phase 4 disaggregated GX10→Mac context
matrix", 2026-08-20]`:

| Depth | Local TTFT (mean) |
|---:|---:|
| 2,048 | 6.48 s |
| 8,192 | 26.77 s |
| 16,384 | 54.79 s |
| 32,768 | 114.31 s |
| 65,536 | 247.88 s |
| 131,008-class | 570.12 s |

*Table 27.1: Local prefill TTFT by depth — linear growth in the prompt,
compute-bound on one Mac `[docs/benchmarks.md §3]`.*

An agent workload that refreshes a 100k-token context is not an edge case
of this table; it is the table's whole reason to exist. This is the demand
side. The supply side is the next section's question: what would it cost to
get that KV from somewhere else?

## 27.4 What the wire would have to carry

Here is the quiet fact that makes disaggregation plausible *for this model
specifically*: Muse Glimmer's KV is small. [Ch 22](22-the-price-of-context.md)
derived 1,024 bytes per token per layer (2 KV heads × head_dim 128 × 2 bytes
× K+V). If all 52 layers shipped in full, that would be 52 KiB per token.
But only the **13 NoPE full-attention layers** ([Ch 14](14-qk-norm-and-rope.md))
grow with context; the **39 SWA layers** ([Ch 23](23-the-swa-ring-and-the-growing-cache.md))
live in a 2,048-token ring. What crosses the wire at depth is therefore:

```text
NoPE:  13 layers × 1,024 B/token × position
SWA:   39 layers × 1,024 B/token × 2,048 tokens (the window, not the past)
```

At the 130,815-token cell that product is **1,823,184,896 B** (≈ 1.82 GB) —
the measured wire payload of the deep cell, reconciled to the byte against
this per-class arithmetic in [Ch 22 §22.7](22-the-price-of-context.md)
`[docs/kvpack-merge-handoff-20260820.md §3 D1]` `[receipt
phase4-disagg-20260820/130815-g900091/out-p4/
f-p4-text-g900091-client.json]`. (Two conventions ride in that
reconciliation: the receiver holds back the boundary token — NoPE rows ship
for prompt − 1, decoded locally — and the SWA rings travel as three
13-layer groups.) Note the ratio: ≈ 13.9 kB (decimal; 13.6 KiB) shipped per token
versus ~53 GFLOP recomputed per token. The sealing plan's external-research
section makes the same point against the literature: ~52 KiB/token effective
full-model footprint versus ~2.2 MB/token for DistServe's OPT-66B — a 42×
smaller artifact class — and DéjàVu's viability rule (transfer cheaper than
recompute) holds with room to spare `[docs/disaggregated-prefill-sealing-plan-20260818.md §4]`.

And the wire is fast *relative to that payload*. The lab's raw reference
ceiling is ~9.4 Gbps single-stream on the pre-migration direct 10GbE link
`[ledger T0]` `[docs/disaggregated-prefill-sealing-plan-20260818.md §W0]`;
after the 2026-08-23 topology migration to the switched fabric the
re-probed product direction measured 9.256 Gbps (reverse 6.161 Gbps,
retained as a deviation) `[ledger GX10 return 2026-08-23, attempts 3–4 + readiness entries]`. Even at the
release floor of 3.0 Gbps installed-payload median
(`[crates/muser-cluster/src/lib.rs:14-15]` — installed payload being the
handoff's payload bytes over the kernel's measured send busy-time, the wire
clock [Ch 31](31-the-wire-discipline.md) defends), the transfer floors are ~224 ms
at 2k, ~1.06 s at 32k, ~3.75 s at 131k `[docs/disaggregated-prefill-sealing-plan-20260818.md §4]`.
Chapter 31 owns the full wire-discipline story (pacing, EEE, why the
product rate sits below raw); the point *here* is the shape: **the wire
cost is seconds; the local recompute cost is minutes.**

So the economics write themselves — *if* you have a prefill machine whose
compute is up to the job, and *if* moving the KV does not break the
exactness contract. Everything after this section is about those two ifs.

## 27.5 The measured payoff

The disaggregated lane, as qualified: a resident vLLM NVFP4 producer on one
GX10 node prefills the prompt and hands the KV to Mac Metal decode over
authenticated Handoff V2. The next two chapters are the producer and the
transport; the measurement is here.

The **shallow, final-image cell** — the one the claims register scopes
carefully — at 2,048 prompt / 256 output tokens on the final image, with
one uncounted warmup handoff then five counted reps:

> **1.493 s median TTFT, 0.22 % counted CV, ≥ 6.23 Gbps installed payload,
> deterministic output** — `[claims #6]`, receipt
> `nvfp4-pacing8g-20260818/p4-wrapper23/`.

The counted-warmup convention is *part of the claim* (rep 0 is ~8 % hot from
CUDA warmup; the ruling that made it uncounted is
`[docs/disaggregated-prefill-sealing-plan-20260818.md §7.3]`). A post-router
re-qualification on the switched fabric reproduced the class: 1.535889499 s
median, CV 0.322 %, payload 6.4592–7.2065 Gbps `[ledger "Post-router GX10
lane requalification"]`.

The **deep cell** — the headline — at 130,815 tokens, EEE-off arm, same
night, same producer, same fixture, one warmup + five counted:

> **137.405 s median remote TTFT, CV 0.576 %, ≥ 6.995 Gbps per-rep payload
> floor, deterministic output, versus 570.122 s local 131,008-token mean:
> 4.149×** — `[claims #6]`, receipts under
> `kvpack-ladder-20260820/stage2-130815-rerun/`.

Scope discipline, both sides, because the number is useless without it: the
remote side is a **median** over five counted reps at **130,815** tokens
with **EEE disabled** (the enrolled link invariant,
[Ch 31](31-the-wire-discipline.md)); the local side is a **mean** at
**131,008** tokens from the same claims row. The local baseline is 0.15 %
deeper than the remote cell, so the payoff is, if anything, understated
`[ledger "EEE A/B at 130815"]`. The earlier Phase-4 matrix — five reps per
depth, an earlier packet lineage — put the whole band at **3.75–4.26×**
across 2,048 → 130,815 `[docs/benchmarks.md §3]`:

| Depth | Local TTFT | Remote TTFT | Payoff |
|---:|---:|---:|---:|
| 2,048 | 6.48 s | 1.520 s | 4.26× |
| 32,768 | 114.31 s | 30.489 s | 3.75× |
| 130,815 | 570.12 s | 137.405 s | **4.149×** (EEE-off cell) |

*Table 27.2: The disaggregated payoff band `[docs/benchmarks.md §3]`
`[ledger "Phase 4 disaggregated GX10→Mac context matrix"]`. Payoff here is
local ÷ remote TTFT — note this is the one ratio family in the book that is
not the llama ÷ muser convention.*

Two honesty notes the claims register insists on, repeated here so the
chapter cannot be quoted without them. First, an earlier **integrated
engineering headline** — 3.881 s cold disaggregated TTFT for a 2,048-token
prompt including 1.87 s of native producer compute, versus ~6.5 s local
serving prefill, paced wire 3.925 Gbps — is a **dated single-cell packet**
from 2026-08-17, "operator-accepted engineering headline, not a
five-repetition stability claim"
`[docs/nvfp4-fast-lane-evidence-20260817.md §Measured product numbers]`.
Second, the historical **5.83×** figure (exact Spark producer versus a 275 s
Mac exact mirror) is **retired** and must never be cited
`[docs/nvfp4-fast-lane-evidence-20260817.md §Measured product numbers]`
`[claims #6]`. Both packets remain non-notarial; the release lock governs
what may be said publicly `[docs/launch-claims.md §Ground rules]`.

And reuse — the Part V machinery — collapses the bill further on repeat
traffic: warm first token 0.6132 s at 65,536 and 1.0566 s at 130,815,
bit-identical text, no producer drive at all `[claims #11]`; delta handoff
54.2851 % of bytes `[claims #12]`. Disaggregation and reuse compose.

## 27.6 Why v0.1 is honestly ONE Mac + ONE producer

Here is a place where the architecture document is blunt, and the book will
not soften it:

> "The v0.1 topology is one Mac decoder and one Spark/GX10 producer. …
> Multi-producer scheduling and node discovery are not implemented."
> `[docs/muser-architecture.md §Durable and remote KV]`

The claims register puts the launch-side form: **"1× Mac + 1× GX10 today"**;
scale-out is roadmap, and no wording may imply a multi-GX10 cluster is
running `[claims #8]`. The receiver admits **one producer at a time — one
control endpoint, one HMAC key id, and a replay ledger keyed per key id**
(`[crates/muser-cluster/src/lib.rs:9-12]`). Onboarding a second node
registers it; it does not create a second concurrent producer
`[docs/one-button-onboarding.md §v1 limits]`.

Why this is the *right* v0.1, not a cop-out: the roles are the architecture;
the placement is a technicality. The producer/consumer split is defined
between *processes*, not vendors or hosts — the handoff protocol, identity
binding, and kvpack state format hold for a colocated producer over loopback
just as for a remote one `[docs/disaggregated-prefill.md §Roles, not
machines]`. What the single-remote-producer placement adds is exactly the
one thing this part of the book has to teach: tensor-core NVFP4 prefill and
unified-memory decode each win on different silicon
`[docs/disaggregated-prefill.md §The idea]`. Colocated producers remain
unqualified, not unarchitected `[docs/release-todo-20260823.md §10]`.

The failure model follows from the topology and is worth stating as a
design principle, because it decides what the producer is allowed to be:

> **The producer is a single point of failure for TTFT, never for
> correctness.**

If the GX10 is down, the lane falls back to local Mac prefill — the engine
keeps a complete local path for exactly this reason
(`[crates/muser-engine/src/prefill.rs:10-12]`: prefill is "the Mac-local
fallback path when GX10 disaggregated prefill (`muser-cluster`) isn't
available"). Output tokens stay exact; TTFT degrades to Table 27.1's column.
The claims register's own boundary language: "one producer is a TTFT SPOF
with local-prefill fallback, never a correctness SPOF" `[claims #13]`. A
disaggregated system whose *correctness* depends on a remote node's uptime
has failed at system design; Muser's contract keeps every correctness gate
on the Mac side of the wire (that is [Ch 32](32-precision-across-the-handoff.md)'s
subject in full).

## 27.7 The decision, as a tradeoff

Every chapter in this book owes a tradeoff with measured consequences. This
one has three:

**Why split at the prefill/decode boundary and not somewhere else?**
Because that is where the roofline flips (Figure 27.1) — the two regimes
want opposite machines, and the artifact that crosses the boundary (KV, ~14
KiB/token at depth) is three orders of magnitude cheaper to move than the
work it saves (~53 GFLOP/token). The measured consequence is the 3.75–4.26×
band `[docs/benchmarks.md §3]`; the counterfactual is the 570.122 s local
cell `[claims #6]`.

**Why not just make local prefill faster?** The obvious alternative — better
local batch kernels — attacks the compute-bound side on silicon whose
computing budget is already spent elsewhere. The measured consequence of
*not* having that option is visible in the six-depth plain matrix: local
prefill means 1.0139–1.0397× versus llama `[claims #2]` — competitive, not
transformative; no local lane turns 570 s into 137 s. The FP4-tensor-core
route exists on the GB10, not on the Mac
(`[docs/disaggregated-prefill-sealing-plan-20260818.md §4]`, the W4A4 /
FlashInfer CUTLASS notes — [Ch 28](28-the-gx10-and-vllm-nvfp4-prefill.md)
gets the details).

**What does the split cost?** A wire, with everything a wire brings: a
pacing story ([Ch 31](31-the-wire-discipline.md) — pacing is the
sender-side rate cap on its own sockets), a durability story (the
replay ledger's fsync dance, [Ch 30](30-handoff-v2-transport.md)), a
security story (mTLS + HMAC, [Ch 30](30-handoff-v2-transport.md)), and a
precision story (trusting someone else's prefill, [Ch 32](32-precision-across-the-handoff.md)).
Every one of those costs is a chapter in this Part because every one of
them *bit* during the campaign — the pacing self-cap (3.9 of 9.4 Gbps was
our own pin `[ledger T1]`), the fsync tail in our own commit path
(`[docs/disaggregated-prefill-sealing-plan-20260818.md §W1]`), EEE's
retransmission blackouts on our own burst schedule `[ledger "EEE link
ruling — operator decision (2026-08-20)"]`. The lane survived them because
it fails closed, not because it was lucky.

## 27.8 What comes next

The argument is done: prefill is a throughput job that wants tensor cores
and FP4; decode is a latency job that wants unified memory close to the
user; Muse Glimmer's KV is small enough to move; and the measured payoff at
depth is a 4.149× TTFT reduction with deterministic output — under scopes
this book will keep restating. To disaggregate you need a prefill machine.
Ours is one ASUS GX10 — an NVIDIA GB10 — running a resident vLLM NVFP4
producer in a docker container, with a fail-closed culture all its own.
That machine, its producer process, and its exit code 75 are the next
chapter.

---

## References

- `[crates/muser-engine/src/decode.rs:53]` — `PREFILL_BATCH_TOKENS = 512`,
  the chunk size behind the prefill intensity arithmetic.
- `[crates/muser-engine/src/prefill.rs:6-12]` — "prefill of T tokens ≈ one
  token's DRAM traffic"; the local-fallback role of the same driver.
- `[crates/muser-cluster/src/lib.rs:9-22]` — 1× Mac + 1× GX10 launch
  config, one-producer-at-a-time admission, 3.0 Gbps release floor.
- `[docs/benchmarks.md]` — §Methodology (repetition and floor conventions),
  §3 (the disaggregated payoff table and the EEE-off 130,815 row).
- `[docs/muser-architecture.md §Durable and remote KV]` — v0.1 topology,
  lane matrix, multi-producer-not-implemented.
- `[docs/disaggregated-prefill.md]` — roles-not-machines; the two-jobs
  argument; honest limitations (single producer, link dependence).
- `[docs/disaggregated-prefill-sealing-plan-20260818.md]` — §4 (KV-size
  math vs DistServe/DéjàVu, transfer floors, GB10 tensor-core notes), §W0
  (raw 9.4 Gbps), §W1 (pacing and the fsync-tail lesson), §7.3 (the
  counted-warmup ruling).
- `[docs/nvfp4-fast-lane-evidence-20260817.md §Measured product numbers]` —
  the dated 3.881 s / 1.87 s / ~6.5 s integrated cell and the retired
  5.83×.
- `[docs/launch-claims.md]` — #2 (local matrix), #6 (the disaggregated
  claims and their scope language), #8 (topology wording), #11/#12 (reuse
  and delta), #13 (SPOF boundary), §Ground rules.
- `[ledger …]` — "Phase 4 disaggregated GX10→Mac context matrix" (the
  payoff band), "EEE A/B at 130815" (the 137.405 s / 4.149× cell), T0/T1
  (raw ceiling, pacing ladder), "Post-router GX10 lane requalification".
- `[receipt phase4-disagg-20260820/130815-g900091/out-p4/f-p4-text-g900091-client.json]`
  — `payload_bytes = 1,823,184,896`, the deep wire payload.
- `[docs/kvpack-merge-handoff-20260820.md §3 D1]` — the payload
  reconciliation (NoPE + SWA arithmetic).
- `[ferrite-book Ch 23]` — the roofline-flip device this chapter ports; its
  A18 Pro points (~3.6 vs ~455 F/B at batch 128) are Ferrite-lineage and do
  not transfer to Muser measurements.
- [glossary](../glossary.md) — terms introduced this chapter: TTFT,
  disaggregated prefill, producer, consumer (receiver), tensor core, TTFT
  cliff, local-prefill fallback.
