# Glossary seeds — Part VI (second half: Ch 31–33)

Terms these chapters define or lean on heavily, seeded for the book
glossary merge. Format: `### term — one-line definition (Ch N)`.

### pacing pin — the producer-side `SO_MAX_PACING_RATE` socket cap that deliberately holds the sender under line rate (8 Gbps against a ~9.4 Gbps path) so the kernel smooths the handoff's bursts; fail-closed in both the readback and the receipt validator. (Ch 31)

### installed payload — the measured goodput of a handoff: payload bytes over the producer's `TCP_INFO.busy_time`, the campaign's only trusted wire clock. (Ch 31)

### wire floor — the minimum transfer time for a given payload at a given rate (2k ≈ 224 ms, 131k ≈ 3.75 s at 3.9 Gbps); the irreducible wire bill before any overhead. (Ch 31)

### replay ledger — the receiver's durably persisted, monotonic high-water mark of admitted generations per HMAC key/epoch, consumed before the ACK and before any live engine pointer is swapped. (Ch 31)

### durable reserve pattern — the crash-safe commit sequence write-temp + fsync + rename + directory-fsync used by the replay ledger; its directory-fsync tail is why operational state must live on the internal disk, never the evidence volume. (Ch 31)

### EEE (Energy-Efficient Ethernet) — the link's low-power idle mode (LPI); on this lane's burst schedule it produced retransmission blackouts quantized at 6.42 ± 0.03 s, so EEE-off is the enrolled link invariant. (Ch 31)

### link invariant — a link-level setting a measurement or claim is enrolled under (here: EEE off on the 10GbE path), shipped as production guidance rather than re-derived per run. (Ch 31)

### raw ceiling — the single-stream TCP rate the physical path sustains with zero tuning (9.40 Gbps pre-rebuild direct; 9.256 Gbps GX10→Mac on the switched fabric, asymmetric in reverse); a property of a topology that must be re-proven after any change. (Ch 31)

### trust ladder — the three strictness rungs of the disaggregated lanes: bit-exact full handoff → exact tokens + declared bounded logits → parity-within-noise; each rung is declared in the lane's identity, with its costs and what it rules out. (Ch 32)

### bounded-logit policy — a qualification contract mode (`bounded-drift`) in which greedy tokens must be bit-exact while full-logit drift must fit sealed bounds (native lane: max < 11.0, mean < 1.25), declared in the frozen identity and checked per sample and per summary. (Ch 32)

### drift envelope — the measured max/mean absolute full-logit (and KV) deltas between two engines' outputs on a fixed fixture (native vs exact: 7.270581/1.040619 at 32 tokens; 10.884401/1.233789 at 2,048/256); deterministic but nonzero, and published as part of the claim. (Ch 32)

### integer-dot producer — the `MUSER_NVFP4_EXACT=1` producer mode whose NVFP4 arithmetic is deterministic by integer construction; a verification anchor that is never served ("Verification only" in the lane matrix). (Ch 32)

### target-cache identity — the digest a receiver pins per producer recipe so KV entries from different recipes (exact vs native) can never alias; part of the cluster config, not derivable from the model file alone. (Ch 32)

### qualification recipe — the exhaustive per-lane qualification contract (`KquantTargetPlusDflash` or `NativeText`); adding a lane without choosing one is a compile error, and an unknown serialized lane is refused before enrollment can mint keys. (Ch 32)

### calibrated gate — a tolerance derived from a measured second quantization (Q6-vs-kquant disagreement) rather than chosen; used for the deep-ladder content controls so no threshold can be accused of convenience. (Ch 32)

### content-local sensitivity — a quality exceedance confined to one content class at one depth (docs@65,536: 15.134% vs 13.339% top-token gate) that did not replicate cross-document; published, not capped. (Ch 32)

### reference lock — an explicit lane users can route to when the native lane's published sensitivity matters (the kquant lane), chosen manually because an automatic disagreement proxy would itself require a second reference computation. (Ch 32)

### cross-vendor arithmetic ABI — the pinned set of reduction orders and materialization dtypes that lets CUDA-produced logits match Metal bit-for-bit; what the wizard's one-ULP chase (attempts 10–31) had to version. (Ch 32)

### maximal coupling — the speculative acceptance rule `accept if rng ≤ min(p/q, 1)` with a residual-corrected resample on rejection, which makes the output marginal exactly the target's regardless of draft quality. (Ch 33)

### Mirror-SD — the split-graph speculative verify overlap: execute the target through a capture layer synchronously, submit the remaining layers and LM head without waiting, and accept no result until the pending suffix completes (`begin/finish_dflash_verify_suffix`). (Ch 33)

### all-accept control — a diagnostic run under forced 100% acceptance (the distributed lane's 110.59 tok/s standard trace) that proves the pipeline's plumbing while proving nothing about throughput; never citable as serving performance. (Ch 33)

### verifier-only ceiling — `output_tokens ÷ sum(remote verifier wall)`: an upper bound granting zero cost to drafting, transport, installation, and scheduling; the decisive rejection bound when it sits under the bar. (Ch 33)

### weight-stream-bound — the structural cost of a remote verifier: one full model-weight read per verification round, which no protocol tuning removes (why the linear distributed lane's ceilings capped at 20.15–55.96 tok/s). (Ch 33)

### target-engine epoch — the named identity of *which* engine executed authoritative target transitions (Mac/Metal vs GX/vLLM); silently switching epochs mid-session is not exact, so fallbacks need a signed seam. (Ch 33)

### carried frontier — the target-selected token held un-evaluated between speculative rounds; its explicit witness geometry prevents publishing state whose KV rows do not exist yet. (Ch 33)

### falsification ledger — the recording device that lists each hypothesis with its evidence class and verdict (the frontier's 14-attempt disposition table; the dispatch-gap reconciliation), so rejected designs teach instead of haunting. (Ch 33)

### hardware-aware token tree — the one surviving distributed-speculation experiment: spend otherwise-idle GX batch arithmetic covering near-miss branches, admitted only through a preregistered emitted-tokens-per-node screen against the 107.9 bar. (Ch 33)

### PREPARED/staged-render/WAL/activation/ACK — the V2 research protocol's durable verification transaction (fsynced commit WAL before idempotent renderer activation); implemented and fault-tested locally, deliberately unwired from serving. (Ch 33)
