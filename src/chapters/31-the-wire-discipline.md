# Chapter 31 — The wire discipline
> **status:** polished  ·  **path:** Muse Glimmer, pinned Muser tree

*Prerequisites: Chapters 22–26 (you know what the KV cache costs and how
kvpack moves it), Chapters 27–30 (you know why prefill is disaggregated,
what the GX10 producer is, and how Handoff V2 authenticates every byte).
This chapter is about the unglamorous layer under all of that: making
1.8 GB cross a 10GbE link in about two seconds — and knowing, with
receipts, why it took exactly that long.*

Chapter 30 ended with a sealed manifest: every segment HMAC-verified, the
replay ledger consulted, the ACK meaningful. That is the *correctness*
story of the wire. This chapter is the *performance and honesty* story.
Between 2026-08-16 and 2026-08-23, the disaggregated lane's transfer looked
sick four different times, and four times the culprit turned out to be
Muser's own infrastructure — a pacing pin, a filesystem, a power-saving
 Ethernet feature, and a stale reference number — never the network the
symptoms were blaming. The pacing ladder is the campaign's cleanest
example of a rule this book keeps returning to: **the link is never the
constraint until it suddenly is, and every stall is self-inflicted until
proven otherwise.**

The rule is not a slogan; it is written into the operating agreements of
the repo itself: "every stall in this campaign was self-inflicted
infrastructure until proven otherwise" is how the research ledger frames
the wire question `[measured-numbers §3, question 3]`, and the operator
cheat sheet at the Muser root states the standing lesson — operational
state (replay ledger, sockets, locks) belongs on the internal disk,
because "the evidence volume's directory-fsync tail produces bimodal ~1 s
stalls in commit paths" `[AGENTS.md, muser root]`.

---

## 31.1 The bill: what actually crosses the wire

Start with the invoice, derived the way Chapter 22 taught you. One
layer-token of KV is 2 KV heads × 128 head-dim × 2 bytes (f16) × 2 planes
(K and V) = **1,024 B** `[docs/memory-footprint.md §KV formula]`. The 52
layers split 39 SWA (window 2,048) + 13 NoPE (full history)
`[docs/muser-architecture.md]`, so a handoff carries two kinds of bytes:

- **NoPE tiles** — absolute, position-free planes: 13 layers × 1,024 B =
  **13,312 B per prompt token**, all of `[0, position)` minus the boundary
  token the receiver holds back for local first logits.
- **SWA groups** — the trailing window only: 39 layers × 2,048 rows ×
  1,024 B = 3 groups of 13 layers ≈ **81.8 MB total**, re-sent whole
  whenever the suffix outgrows the window
  (`muse_schedule_span`, `[crates/muser-cluster/src/schedule.rs:84-98]`).

At the shallow end, a 2,048-token prompt moves ≈104 MiB of target KV plus
≈42.5 MiB of DFlash context in 36 segments — ≈146 MiB combined
`[docs/disaggregated-prefill-sealing-plan-20260818.md §2]`. At the deep
end, the 130,815-token cell moves exactly **1,823,184,896 B** — the
per-class decomposition above reconciles to that byte, derivation and
receipts in [Ch 22 §22.7](22-the-price-of-context.md)
`[docs/kvpack-merge-handoff-20260820.md §3 D1; receipt
phase4-disagg-20260820/130815-g900091/out-p4/f-p4-text-g900091-client.json]`.
(An early doc misread that payload as "~7 GB" by confusing it with the
producer's `--kv-cache-memory-bytes` *allocation* — one of this book's
standing landmines, told in full at
[Ch 22 §22.7](22-the-price-of-context.md).)

Now the physics. A 10GbE link at its measured ~9.4 Gbps raw ceiling moves
1.823 GB in `1.823e9 × 8 / 9.4e9 ≈ 1.55 s`; at the 3.9 Gbps the lane
originally achieved it takes `≈ 3.73 s`; at the 6.995 Gbps deep-cell
floor it takes `≈ 2.09 s`. That last number is the chapter's title claim:
**about two seconds, honestly.** The sealing plan's own floor table says
the same thing in campaign units — at 3.9 Gbps the lossless transfer
floors are 2k ≈ 224 ms, 32k ≈ 1.06 s, 131k ≈ 3.75 s, and "at 131k the
floor exceeds the entire current 2k handoff — delta transfer (W3) is what
keeps long context viable, not a faster NIC"
`[docs/disaggregated-prefill-sealing-plan-20260818.md §4]`. Everything
else in this chapter is the gap between 1.55 s of physics and whatever
the lane actually delivered on a given night. Figure 31.1 draws the
whole path now, so you can place each villain as it appears.

```
 The wire path, with every discipline that touches it (2026-08-23 topology):

 ┌──────────────────────── GX10 producer (producer-1) ────────────────────────┐
 │  vLLM NVFP4 prefill (CUDA, layer-major)                                  │
 │                                                                          │
 │   SWA groups (~82 MB) stream during the last CUDA ubatches               │
 │   NoPE bulk (1.74 GB, 95.5% of payload) waits for layer 51               │
 │                    │                                                     │
 │   sender thread: SO_MAX_PACING_RATE = 8 Gbps pin (fail-closed readback)  │
 │   wire clock:     TCP_INFO.busy_time  (the only honest denominator)      │
 └────────────────────┼─────────────────────────────────────────────────────┘
                      ▼
   [ enp1s0f0np0 · 192.0.2.20    ]══════════[ MikroTik 10GbE fabric ]
                      │                        EEE off — enrolled link invariant
                      ▼                        (6.42 s retransmission ladders
   [ Mac en0 · 192.0.2.10    ]                  otherwise; §31.4)
 │  per-segment: drain → HMAC verify → install into a DETACHED Metal gen    │
 │  terminal seal → ReplayLedger.reserve (write+fsync+rename+dir-fsync      │
 │                  on the INTERNAL disk — never the evidence volume)       │
 │  atomic swap → ACK          Wi-Fi en1 exists and never carries a         │
 └──────────────────────────── measurement [scripts/gx10/README.md:7-11] ──┘
```
*Figure 31.1: The full wire path with its annotations. Every box on this
diagram is a chapter villain at least once: the pacing pin (§31.2), the
ledger volume (§31.3), EEE on the switch link (§31.4), and the reference
numbers after the topology change (§31.5).*

## 31.2 The pacing ladder: 3.9 of 9.4 Gbps was our own pin

The ladder is best told as the campaign told it, one dated rung at a time.

**Rung 0 — the symptom (2026-08-17).** The F-series engineering packet
measured installed payload at **3.910 Gbps median, CV 0.401%** on the
2,048-class cell — stable, but less than half of what 10GbE should do
`[docs/nvfp4-fast-lane-evidence-20260817.md §Measured product numbers]`.
The production sender had just acquired a **4.0 Gbps kernel pacing
ceiling** in the N5 transport fix, and five fresh 109 MB serving handoffs
passed at median 3.893 Gbps, CV 0.6074% `[ledger §N5]`. Note what the CV
is telling you: the *sender* was rock-steady. A stable wrong number is
still a wrong number.

**Rung 1 — the raw ceiling (2026-08-18, T0).** A 5-second TCP probe in
both directions answered the only question that matters first: is the
wire healthy? The point-to-point 10GbE path sustained **9.40 Gbps
single-stream, MTU 1500, zero tuning, zero retransmits over 30 s**
`[ledger §T-series T0]`. Conclusion, verbatim from the ledger: "The wire
was never the constraint."

**Rung 2 — the culprit (W0).** The 3.9 Gbps was the sender's own
`SO_MAX_PACING_RATE = 500 MB/s` pin — a Linux socket option that caps the
kernel's transmit pacing — set as "the N-series floor guard" to protect
the 3 Gbps product floor on a then-unhealthy link
`[docs/disaggregated-prefill-sealing-plan-20260818.md §5 W0]`. Both
producers shared it, because the vLLM connector imports the sender from
the llamacpp path.

**Rung 3 — the raise (W1).** The pin went 4 → 8 Gbps, env-configurable.
Installed payload moved **3.91 → 5.89 Gbps median** `[ledger §T-series
T1]`. The code that owns this today reads:

```python
# scripts/gx10/llamacpp/muser_v2_send.py:55
HANDOFF_PACING_BYTES_PER_SECOND = 1_000_000_000  # 8.0 Gbps; the direct link measures 9.4

# scripts/gx10/llamacpp/muser_v2_send.py:58
def handoff_pacing_bytes_per_second() -> int:
    """Configured payload pacing ceiling (default 8.0 Gbps).

    The N-series pin of 500 MB/s protected the 3 Gbps product floor on an
    unhealthy link. The direct 10GbE path measures 9.4 Gbps single-stream, so
    the default now sits ~15% under line rate. MUSER_GX10_PACING_BYTES_PER_SECOND
    overrides for experiments without a redeploy; a value the kernel refuses
    still fails closed in `configure_linux_pacing`.
    """
```

Two disciplines live in that one constant. First, **the pin stays ~15%
under line rate on purpose** — it exists so the kernel smooths bursts
rather than letting a 1.74 GB NoPE flood collide with TCP's own
congestion logic (§31.4 explains why bursts are this lane's resting
state). Second, **the pin is fail-closed in both directions**: the
resident producer's receipt validator refuses any handoff whose
`payload_pacing_bps` is below 4 Gbps — `handoff["payload_pacing_bps"] <
4_000_000_000` is a hard rejection in the daemon's receipt check
`[scripts/gx10/vllm/muser_native_prefilld.py:569-570]` — and a kernel
refusal of the socket option aborts the send rather than silently
unpacing.

**Rung 4 — the honest clock.** Raising the pin exposed a measurement
defect: at pins above the achievable rate, the producer's TCP busy-time
metric tracks real jitter (CV ~5%) instead of the pacer (CV 0.4%), and
the old ≤2% link-CV gate had been calibrated on the flattened version.
The campaign's response is worth quoting as a pattern: the wire clock
ruling is that Linux `TCP_INFO.busy_time` is *the only honest link
denominator* — userspace send-time and receiver first-read clocks were
both rejected as buffer- and compute-dependent `[ledger §P4; §N5]` — and
the link gate was re-specified from a rate-CV to a **per-counted-rep
floor: every counted repetition ≥ 3.0 Gbps installed payload**, with the
CV retained in receipts for audit only `[docs/disaggregated-prefill-
sealing-plan-20260818.md §7.4]`. An earlier five-rep row of 5.581 /
4.550 / 5.309 / 5.769 / 4.765 Gbps at CV 8.9985% survives in the record
only as failed-metric evidence `[measured-numbers §1e]`.

**Rung 5 — the landed numbers.** With the ladder climbed, the lane's
shallow headline became the final-image 2,048/256 packet: **TTFT median
1.493 s, counted CV 0.22%, ≥6.23 Gbps installed payload (receipt minimum
6.228), deterministic, exit 0** — `stable: true`
`[claims #6; ledger §T-series "Final packet on the final image"; receipt
nvfp4-pacing8g-20260818/p4-wrapper23/]`. The deep 130,815 cell under the
EEE-off invariant holds a **≥6.995 Gbps floor** with TTFT median 137.405
s at CV 0.576% `[claims #6; ledger §EEE A/B at 130815]`. And the one
place the lane exceeds the pin is the wizard's onboarding recipe, where
three enrollment handoffs measured **9.812 / 8.887 / 8.690 Gbps**
installed payload — onboarding-recipe scope, explicitly not a
serving-throughput claim `[claims #9]`. The count of bytes never changed
on this ladder; only the honesty of the clocks and the height of a pin
we set ourselves.

## 31.3 Our own fsync tail: the durability lesson of 2026-08-18

The pacing fix did not kill the 2,048-class TTFT variance. What remained
was a **bimodal ~1 s stall on random repetitions** — the packet whose
five-rep TTFT median was 2.699 s at CV 21.40% `[docs/nvfp4-fast-lane-
evidence-20260817.md §Measured product numbers]`. The hunt is a small
masterpiece of elimination: per-frame arrival stamps and absolute seal
clocks ruled out the raw link (0 retransmits under saturation), CPython
GC (sub-ms collections), producer compute (flat CPU), Mac-side engine
phases (constant), and session creation (constant ~650 ms) `[ledger
§T-series T2]`. What was left sat between `prepare_commit` and `commit`:
**the replay ledger's directory fsync on the evidence volume** — reserve
median 0.22 ms, observed tail **691 ms** `[ledger §T-series T2]`.

Here is why a directory fsync is on the critical path at all. Replay
admission (Chapter 30's defense against a replayed generation) must be
*durable before the ACK leaves and before any live engine pointer is
swapped* — the timeline in Figure 31.2 shows where the tail lands.
`ReplayLedger::reserve` therefore persists the new high-water mark with
the classic crash-safe sequence — write a temp file, `fsync` the file,
`rename` over the target, then `fsync` the *directory* so the rename
itself survives power loss. The function is `persist_replay_state`
(`security.rs:460-491`), walked step by step in
[Ch 30 §30.6](30-handoff-v2-transport.md); the expensive step is the last
one, and on the big external evidence disk — a volume tuned for throughput,
under load from everything else the campaign writes to it — it has a
bimodal ~0.7–1.0 s tail `[scripts/gx10/durable_fsync_probe.py:5-12]`.

```
 one handoff's commit path, time going right (log scale in spirit):

   segments drain ── verify+install ── seal ── RESERVE ─────────────► swap → ACK
   (paced by sender)  (~0.2 s total)          │
                                             │ write tmp
                                             │ fsync file        ~0.2 ms median
                                             │ rename
                                             │ fsync DIRECTORY ← 0.22 ms median,
                                             │                    691 ms tail on the
                                             │                    evidence volume
                                             ▼
                                   TTFT absorbs the tail whole
```
*Figure 31.2: Where the ~1 s went. The receiver phases the N-series
instrumentation isolated (verify+install+seal+commit ≈ 0.2 s, constant
`[ledger §N2]`) are not the stall; the stall is one `fsync` of one
directory on the wrong volume `[ledger §T-series T2]`.*

The fix was one line of configuration with a permanent lesson attached:
move `replay_ledger` to the internal disk. TTFT median went **2.699 s /
CV 21.40% → 1.596 s / CV 0.56%** `[ledger §T-series T2]`. The lesson,
now enshrined in the repo's working agreements: **operational state
(replay ledger, sockets, locks) belongs on the internal disk; the
evidence volume is append-only storage, not a commit path** `[AGENTS.md,
muser root]`. And the guard became code, not advice — the receiver now
*refuses to bind* on a slow ledger volume, before any transfer can be hurt
by it. The gate is the bind-time probe that
[Ch 30 §30.7](30-handoff-v2-transport.md) introduced (twenty iterations of
the exact reserve pattern against the ledger's parent directory, refusal
past a 100 ms worst sample); its error message teaches the fix:

```rust
// crates/muser-cluster/src/receiver.rs:139 (inside check_ledger_volume_with)
    if tail > LEDGER_RESERVE_PROBE_MAX_TAIL {
        return Err(format!(
            "replay ledger volume {} has a {tail:?} durable-reserve tail; \
             the handoff commit path would stall on it — point replay_ledger \
             at the internal disk (see scripts/gx10/durable_fsync_probe.py)",
            directory.display()
        ));
    }
```

`RemoteReceiver::bind` runs `check_ledger_volume` — twenty real
write+fsync+rename+dir-fsync cycles against the ledger's parent
directory — before the listener even opens, and refuses the config if the
worst sample exceeds 100 ms `[crates/muser-cluster/src/receiver.rs:189-206]`.
The probe `probe_ledger_reserve` is a faithful in-process twin of
`persist_replay_state`'s sequence `[crates/muser-cluster/src/receiver.rs:150-178]`.

The operator-side twin is `scripts/gx10/durable_fsync_probe.py`, whose
docstring is the pre-flight ritual in miniature: run it "against the
directory that will host `replay_ledger` in the receiver cluster config";
"median ~0.1-0.3 ms and max < ~5 ms: healthy; the ledger may live here";
"a max (or p99) of hundreds of ms: do NOT put operational durability
state on this volume" — and it **exits 1 if the worst reserve exceeds
`--max-tail-ms`**, so it can gate automation, not just inform humans
`[scripts/gx10/durable_fsync_probe.py:16-33]`. The docstring ends with
the scar: "This exact failure cost the fast lane its stability gate on
2026-08-18."

## 31.4 EEE: when the power saver eats the burst

The third stall was the strangest, because the shallow cells were now
healthy and the *deep* cells collapsed. Two investigations, two
payoffs.

**The controlled diagnosis (N2, 2026-08-16 era).** Six repetitions of the
identical strict 1×2048×256 cell, with receiver phases instrumented
(Figure 31.3):

| Condition | Rep | Producer wire s | installed Gbps | verify+install+seal+commit |
|---|---|---:|---:|---:|
| as-is | 1 | 13.25 | 0.093 | 0.20 s |
| as-is | 2 | 6.58 | 0.187 | 0.16 s |
| as-is | 3 | 19.75 | 0.062 | 0.20 s |
| serial quiesce | 1 | 19.54 | 0.063 | 0.20 s |
| serial quiesce | 2 | 0.22 | **5.526** | 0.20 s |
| serial quiesce | 3 | 6.57 | 0.187 | 0.19 s |
| EEE off (probe) | 1 | 0.20 | **6.215** | 0.20 s |
| EEE off (probe) | 2 | 0.24 | **5.208** | 0.19 s |

*Figure 31.3: The N2 table. Installed payload swings 0.062–5.526 Gbps —
roughly 90× — across identical conditions, while the receiver's entire
post-wire cost sits still at ~0.2 s `[ledger §N2]`.*

Read the columns the way the campaign did. Receiver backpressure:
refuted (constant ~0.2 s). Producer compute: refuted. The collapsed
component is the **wire span itself**. [EEE](../glossary.md#eee-energy-efficient-ethernet) —
Energy-Efficient Ethernet, the link's low-power idle mode — was
enabled-and-active on the GX10 side (Tx LPI 19 µs), and with EEE disabled
on the Spark side both probe reps ran the fast case `[ledger §N2]`. One
wrinkle that matters operationally: macOS exposes no EEE control on
`en0`; the toggle exists only on the Spark side `[ledger §N2]`.

**Why this lane, specifically.** EEE punishes bursty senders, and this
lane's wire schedule is bursty by architecture. The SWA groups (~82 MB)
stream early, during the last CUDA ubatches; but the NoPE bulk — 1.74 GB,
95.5% of the payload by the §31.1 arithmetic — cannot start until CUDA
finishes layer 51, because NoPE planes are absolute full-history planes
built in layer order `[docs/kvpack-merge-handoff-20260820.md §6 "Pacing
reality"]`. The deep cell therefore manufactures **41–47 s of forced link
idle and then dumps a 1.74 GB burst onto a link that has gone to sleep**.

**The blackouts, quantized.** On the 130,815-token payload the failure
mode is not vague jitter but *discrete retransmission ladders quantized
at 6.42 ± 0.03 s*, after the LPI idle, and counted reps split into a
0.68–1.73 Gbps regime versus a 7.20 Gbps regime — same fixture, same
producer, same night `[ledger §"EEE link ruling — operator decision
(2026-08-20)"; receipts phase4-disagg-20260820/130815-g900091/]`.

**The ruling (2026-08-20).** The operator decision, recorded in the
ledger: "EEE-off is authorized for deep-payload p4 packets and
diagnostics; **EEE-off is enrolled as the link invariant for the
disaggregated lane**; 'disable EEE on the point-to-point 10GbE link'
ships as production guidance" `[ledger §EEE link ruling]`.

**The confirmation by intervention (2026-08-21).** A ruling deserves an
experiment, so the ladder session ran a same-night A/B in which *only*
the EEE state differed — 1 warmup + 5 counted reps per arm
`[ledger §"EEE A/B at 130815"]`:

- **Arm A, EEE-active:** TTFT median 138.886 s, CV 2.0013%, per-rep
  payload [7.213, 7.445, 7.315, **1.728**, 7.275] Gbps — exactly one rep
  lost one ~6.4 s retransmission ladder; `stable:false`.
- **Arm B, EEE-off:** TTFT median **137.405 s, CV 0.576%**, payload
  [7.043, 7.063, 7.084, 6.995, 7.084] Gbps (minimum 6.995),
  deterministic.

Attribution "by intervention, not correlation" — flip the link state, the
blackout follows it. Note also what the two medians say: at this depth
EEE's damage is *variance, floor violations, and occasional +6.4 s reps —
not the median* `[ledger §EEE A/B]`. The payoff number the claims
register carries — **4.149×** median TTFT versus the 570.122 s local
131,008-token mean baseline — is explicitly the EEE-off arm's
`[claims #6]`.

## 31.5 After the rebuild: re-prove the ceiling, and never trust en1

On 2026-08-23 the lab moved from the retired direct `retired /30` link
to the wired MikroTik fabric: Mac Ethernet `en0` at `192.0.2.10` to
the producer's `enp1s0f0np0` at `192.0.2.20` `[AGENTS.md, muser
root]`. Every historical wire number in this chapter — including the 9.40
Gbps T0 ceiling — was measured on the old path, and the topology
amendment to the sealing plan says what to do about that: "qualification
must first re-anchor raw TCP on the new topology and must never use Mac
Wi-Fi `en1`" `[docs/disaggregated-prefill-sealing-plan-20260818.md,
topology amendment]`.

The re-anchor found something the direct link had hidden: **asymmetry**.
In the product payload direction (GX10 → Mac) single-stream TCP measured
**9.256 Gbps** (adjacent probes 9.218 and 9.291); the reverse direction
measured only **6.161 / 6.501 / 5.410 Gbps** — retained as a deviation
and deliberately *not* promoted to a pass `[ledger §"Final GX10 campaign
attempt 4"; receipts final-campaign-20260823/attempt-4/phase0/tcp-*.json]`.
The rule this installs is the chapter's last and most portable one: **a
reference number is a property of a topology, not of hardware** —
re-prove the raw ceiling before relying on any inherited figure, and
treat a deviation in the non-product direction as data, not as a
boundary to quietly round up. The product direction being the healthy
one is what let the campaign proceed; the reverse number stays in the
record as an open deviation `[measured-numbers §1e]`.

With the ceiling re-anchored and EEE off, the lane re-qualified on the
switched fabric: the 2,048/256 P4 packet measured **TTFT median
1.535889499 s at CV 0.322%**, installed payload 6.4592–7.2065 Gbps, all
six outputs sharing the same token and full-logit digests, `stable: true`
`[ledger §"Post-router GX10 lane requalification"; receipt
final-campaign-20260823/attempt-4/p4/P4_VERDICT.json]`. And the Wi-Fi
rule is written where operators will trip over it: "Mac Wi-Fi is `en1`;
a measurement is invalid if it routes there" `[scripts/gx10/README.md:7-11]`.

## 31.6 The tools: a diagnostic ladder, bottom-up

Every lesson in this chapter is productized under `scripts/gx10/`,
documented in `scripts/gx10/README.md` and tested by
`scripts/tests/test_gx10_diagnostics.py`. The flow is deliberately
bottom-up — fix the layer under suspicion before debugging the layer
above it `[scripts/gx10/README.md:21-41]`:

1. **`tcp_probe.py`** — "Answer exactly one question, fast: is the wire
   healthy?" Interpretation bands are in the docstring: ~9+ Gbps
   single-stream means look at the product path (pacing pin, ledger
   placement, producer health); ~3–6 Gbps means check MTU, socket
   buffers, and on the GB10 the ConnectX-7 driver (pre-580.142 drivers
   throttle the 200GbE ports to ~13 Gbps); and — the sentence that
   saves the next agent an afternoon — "**Bimodal stalls of ~1 s under
   load are NOT a raw-link symptom**; that pattern in the product lane
   points at the receiver's durable reserve" `[scripts/gx10/tcp_probe.py:2-31]`.
2. **`durable_fsync_probe.py`** — reproduces the exact
   write+fsync+rename+dir-fsync reserve pattern against a candidate
   ledger directory; exits 1 past `--max-tail-ms` (§31.3)
   `[scripts/gx10/durable_fsync_probe.py]`.
3. **`handoff_report.py`** — turns one qualification packet's retained
   receipts into the per-repetition phase table (producer `sched/first/
   d2h/hash/pack/seal/wire/gbps` beside receiver `drain/verify/install/
   commit/seal_off`) and maps patterns to causes: "wire stable + seal
   bimodal by ~1 s … the replay ledger's directory fsync on a slow
   volume"; "`d2h` or `first_layer` growing across reps: producer-side
   slowdown"; "wire far below what tcp_probe.py measures raw: the
   sender's pacing pin or a genuinely sick link, in that order"
   `[scripts/gx10/handoff_report.py:2-33]`.
4. **`restart_resident_producer.py` / `supervise_resident_producer.py`**
   — the fail-closed producer's restart ritual (stale `O_EXCL` startup
   receipt, RoPE cache, and socket must be moved aside; a bare `docker
   restart` is not enough) and its unattended supervisor that latches
   off after three consecutive failed starts `[AGENTS.md, muser root;
   scripts/gx10/README.md:42-48]`.

The deep-payload soak that closed this arc deserves its own line: eight
consecutive 130,815-token handoffs with zero producer deaths and
deterministic output, payload declining 6.87 → 3.47 Gbps across the soak
with every rep ≥ the 3.0 floor — a bounded soak, explicitly "not a 20-rep
W4 stability packet" `[ledger §"bounded eight-handoff deep-load soak";
receipts final-campaign-20260823/attempt-4/soak/run-attempt-3/SOAK_VERDICT.json]`.

## 31.7 Tradeoffs

**Pin vs no pin.** The obvious alternative — remove `SO_MAX_PACING_RATE`
and let TCP self-clock at line rate — was never taken, and the measured
record explains why the pin survives at 8 Gbps: the sender's own floor
guard origin ("the N-series pin of 500 MB/s protected the 3 Gbps product
floor on an unhealthy link" `[scripts/gx10/llamacpp/muser_v2_send.py:58-66]`),
the fail-closed readback discipline, and a burst schedule that is
architecturally coerced (§31.4). The measured cost of the wrong pin was
3.91 vs an available ~9.4 Gbps — a 2.4× self-cap; the measured benefit
of keeping *a* pin is a sender whose receipts can be validated against
`payload_pacing_bps ≥ 4 Gbps` at the daemon
`[scripts/gx10/vllm/muser_native_prefilld.py:569-570]`. Multi-stream
slicing, the other obvious lever, was "evaluated and rejected by the W0
measurement (a single stream already saturates)"
`[docs/disaggregated-prefill-sealing-plan-20260818.md §5 W1]`.

**Evidence volume vs internal disk.** The append-only evidence volume is
the campaign's integrity substrate — and the worst possible home for a
0.22-ms-median, 691-ms-tail directory fsync on the ACK path. The
measured consequence of getting this wrong was CV 21.40% on a five-rep
TTFT packet; the measured consequence of fixing it was CV 0.56% at
median 1.596 s `[ledger §T-series T2]`. The design resolution keeps both
virtues: evidence stays append-only on the big disk, operational state
moves inside, and the boundary is *enforced* by the bind-time probe
rather than remembered by operators `[crates/muser-cluster/src/receiver.rs:108-148]`.

**EEE on vs EEE off.** Energy-efficient idle is a fine default for a
mostly-quiet office link and a measurable liability for a link whose
workload is 41–47 s of silence followed by 1.74 GB. The campaign's
disposition is scoped, not global: EEE-off is the *enrolled link
invariant for the disaggregated lane*, ships "as production guidance,"
and shallow 2,048-class packets had passed repeatedly with EEE
enabled-active earlier in the campaign (the F-series and N5 packets)
`[ledger §EEE link ruling; docs/nvfp4-fast-lane-evidence-20260817.md]`.
What EEE-off buys: the 6.42 s ladders vanish and arm B's floor holds at
6.995 Gbps `[ledger §EEE A/B]`. What it costs: a standing operator
instruction, and one producer death during the EEE-off sequence's ninth
consecutive deep handoff that remains an open
sustained-deep-load follow-up `[claims #13]`.

**Clocks.** Every wire-rate number in this chapter is
`TCP_INFO.busy_time`-based or receiver-drain-based because the two
convenient alternatives — userspace send time and receiver first-read —
were *measured to be wrong* (buffer- and compute-dependent), and an
entire five-rep row at CV 8.9985% had to be retired as failed-metric
evidence before the gate was re-specified to a per-rep floor
`[ledger §P4, §N5; docs/disaggregated-prefill-sealing-plan-20260818.md §7.4]`.

## 31.8 Where the gap lives

This chapter's "gap" is the spread between the raw ceiling and installed
payload. Where does it live, post-ladder? At the shallow end, the
wrapper23-class cell sits at ≥6.23 Gbps against a 9.4-class raw path:
the residue is TLS+framing+per-segment verify overhead and the pacer's
deliberate ~15% margin under line rate, with a measured ~133 ms
pacer-drain tail on the 2,048 cell `[docs/disaggregated-prefill-sealing-
plan-20260818.md §5 W2]`. At the deep end, the floor is wire-dominated
by construction — 1.82 GB cannot arrive faster than physics — which is
precisely why the reuse machinery of Chapters 25–26 (warm hits, delta
handoffs) attacks the *bytes*, not the link. And the variance component
of the gap, the part that looked like mystery for a week, lived
entirely in our own footprint: a pin, an fsync, a power saver. None of
it was the network.

## 31.9 What comes next

Bytes now arrive intact, paced honestly, on a link whose invariants are
written down. But intact is not the same as correct. The producer on the
far side of Figure 31.1 computes your prefill KV with CUDA kernels,
NVFP4 tensor cores, and reduction orders no Metal kernel reproduces —
and the Mac then decodes from those bytes as if they were its own. What
does "the producer's KV is good enough" have to mean before that is not
faith but engineering? That is [Ch 32](32-precision-across-the-handoff.md)
— the trust chapter: exact-token policies, declared bounded-logit rules,
the integer-dot anchor, and the drift record that keeps every one of
those words honest.

---

## References

- `[AGENTS.md]` — Muser root working agreements: the 2026-08-18
  durability lesson (operational state internal, evidence append-only),
  the ~9.4 Gbps healthy reference, `en1` prohibition, the gx10 tool
  cheat sheet.
- `[ledger §T-series]` — `docs/goal-parity-ledger-2026-08.md`: T0 raw
  ceiling (9.40 Gbps), T1 pacing pin (3.91 → 5.89 Gbps), T2 seal-stall
  root cause (ledger dir-fsync tail 691 ms; TTFT 1.596 s / CV 0.56%
  after the move), T3 clean-image validation, the final wrapper23
  packet (1.493 s / CV 0.22% / ≥6.23 Gbps).
- `[ledger §N2]`, `[ledger §N5]`, `[ledger §P4]` — the EEE collapse
  table (0.062–5.526 Gbps, ~90×), the 4 Gbps pacing-ceiling origin, and
  the wire-clock ruling (`TCP_INFO.busy_time`).
- `[ledger §"EEE link ruling — operator decision (2026-08-20)"]`,
  `[ledger §"EEE A/B at 130815"]` — the 6.42 ± 0.03 s retransmission
  ladders, the enrolled EEE-off invariant, the intervention A/B
  (138.886 s / CV 2.0013% vs 137.405 s / CV 0.576%, floor 6.995 Gbps).
- `[ledger §"Final GX10 campaign attempt 4"]`,
  `[ledger §"Post-router GX10 lane requalification"]`,
  `[ledger §"bounded eight-handoff deep-load soak"]` — post-rebuild
  asymmetry (9.256 vs 6.161 Gbps), the re-anchored P4 packet, the soak.
- `[claims #6]`, `[claims #9]`, `[claims #13]` — `docs/launch-claims.md`:
  the 1.493 s / ≥6.23 Gbps and 137.405 s / 4.149× EEE-off scopes; wizard
  rates 9.812/8.887/8.690 Gbps; producer self-recovery boundary.
- `[docs/disaggregated-prefill-sealing-plan-20260818.md]` — §2 payload
  (≈146 MiB at 2k), §4 link floors (224 ms / 1.06 s / 3.75 s at 3.9
  Gbps), §5 W0/W1 (the 500 MB/s pin, the 4→8 Gbps raise), §7.4 the
  link-gate re-spec, the 2026-08-23 topology amendment.
- `[docs/kvpack-merge-handoff-20260820.md]` — §3 D1 (payload 1,823,184,896
  B, reconciled; the "~7 GB" correction), §6 "Pacing reality" (SWA ~82 MB
  early, NoPE bulk behind layer 51).
- `[docs/nvfp4-fast-lane-evidence-20260817.md]` — installed payload
  3.910 Gbps / CV 0.401% and the 2.699 s / CV 21.40% packet.
- `[crates/muser-cluster/src/security.rs:355-491]` — `ReplayLedger`
  admission/reserve semantics (reserve before publication, latched
  degradation) and the write+fsync+rename+dir-fsync
  `persist_replay_state`.
- `[crates/muser-cluster/src/receiver.rs:108-206]` —
  `check_ledger_volume` / `probe_ledger_reserve` and the bind-time
  refusal; the module doc's "cannot accidentally bypass" guarantee.
- `[crates/muser-cluster/src/schedule.rs:84-157]` — the span schedule:
  NoPE tiles over `[cut, position)`, SWA over the trailing window,
  layer-major order.
- `[scripts/gx10/llamacpp/muser_v2_send.py:54-76]` — the 8 Gbps
  `HANDOFF_PACING_BYTES_PER_SECOND`, its rationale docstring, the
  `MUSER_GX10_PACING_BYTES_PER_SECOND` override, fail-closed readback.
- `[scripts/gx10/vllm/muser_native_prefilld.py:514-577]` — producer
  receipt validation incl. `payload_pacing_bps >= 4_000_000_000` and the
  `linux-tcp-info-busy-time-v1` wire source.
- `[scripts/gx10/tcp_probe.py]`, `[scripts/gx10/durable_fsync_probe.py]`,
  `[scripts/gx10/handoff_report.py]`, `[scripts/gx10/README.md]` — the
  diagnostic ladder and its interpretation bands.
- `[measured-numbers §1e]` — the book's wire-rate table incl. the retired
  CV 8.9985% row and the asymmetry caveat.
- [Ch 22](22-the-price-of-context.md), [Ch 24](24-kvpack-the-format.md),
  [Ch 26](26-delta-handoff-and-migration.md) — the byte arithmetic and
  the reuse machinery that shrinks this chapter's invoice.
- [Ch 30](30-handoff-v2-transport.md) — the authentication machinery
  under everything here; [Ch 32](32-precision-across-the-handoff.md) —
  what "intact but correct" must mean next.
