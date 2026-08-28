# Chapter 32 — Precision across the handoff
> **status:** polished  ·  **path:** Muse Glimmer, pinned Muser tree

*Prerequisites: Chapters 5–7 (the two weight lanes and their arithmetic),
Chapter 20 (the soft cap and why it changes logit gaps), Chapters 28–31
(the producer, the transport, and the wire it rides). This chapter
assumes you believe the bytes arrive intact — Chapter 31 bought you that
— and asks the harder question: are they **right**?*

Chapter 31 ended on a deliberately sharp distinction: intact is not the
same as correct. The Handoff V2 machinery of Chapter 30 proves the bytes
on the Mac are the bytes the producer signed; nothing in a MAC tells you
the producer's arithmetic was worth signing. This chapter is the one this
book was explicitly demanded to contain — the user's own words were that
what they wanted read out of this codebase was "the special care we had
for precision while bringing prefill from vLLM" — and it is where that
care lives: in declared policies, sealed bounds, mode-separated
identities, an integer-dot anchor, and a drift record that is published
even when one row of it is unflattering.

The demand is not sentimental. Ask first what breaks if this is wrong.
When the GX10 prefills, a different machine — CUDA kernels, NVFP4 tensor
cores, cuBLAS reduction orders, Blackwell software E2M1 rounding —
computes the KV cache that the Mac's decoder will treat as ground truth
for every subsequent token. Not for one token: for all of them. A
producer that is slightly wrong does not announce itself with a crash.
It hands over a plausible cache, and the Mac then generates fluent,
confident text on a foundation nobody checked.

We went in expecting the disagreements to be rounding noise. They were
not even the same *kind* of arithmetic. The two engines do not agree on
*how to sum*: the wizard campaign traced one mismatch to "CUDA's serial
128-dim attention reduction vs Metal's 32-lane tree, and F32 vs F16
residual materialization," and that only surfaced after a layer-0 ladder
chase (`attn_norm-0` element 4 → K RoPE 256 → `attn_out-0` element
4,096) isolated two arithmetic-ABI splits. We kept the trail:
`[measured-numbers §2 Arc 4; ledger §2b, 2026-08-24]`.

Nor is this a Muser embarrassment that trying harder would fix. Nobody
in the field achieves bitwise CUDA↔Metal equivalence — llama.cpp itself
uses tolerance-based backend diffs `[docs/disaggregated-prefill-sealing-
plan-20260818.md §4]`. So "good enough" has to be a *contract*, and the
contract has to be checkable. This chapter is about the three contracts
Muser actually runs.

---

## 32.1 The trust ladder

Before the details, the shape. The question to carry through the rest of
the chapter is not "is the handoff correct" — that is unanswerable as
posed — but *how much agreement did we buy, and what did each level of
agreement cost?* Muser's disaggregated lanes answer it as a three-rung
ladder, and every rung names its price and what it rules out
(Figure 32.1):

```
                THE TRUST LADDER (strictest at the top)

 ┌─────────────────────────────────────────────────────────────────────┐
 │ RUNG 3 — BIT-EXACT FULL HANDOFF  (combined kquant lane)             │
 │   exact target tokens  AND  exact FULL logits (max/mean delta = 0)  │
 │   AND exact DFlash tokens/trace                                     │
 │   cost:    a versioned cross-vendor arithmetic ABI — wizard         │
 │            attempts 10–31 chasing one f16 ULP in layer-1 V into     │
 │            51.7 M differing logits (4–7 accelerator-hours)          │
 │   rules out: any logit-level doubt at all; what remains is          │
 │            scheduling and speed                                     │
 ├─────────────────────────────────────────────────────────────────────┤
 │ RUNG 2 — EXACT TOKENS + DECLARED BOUNDED LOGITS  (native/text)      │
 │   greedy tokens bit-exact; full-logit drift must fit a rule that    │
 │   is WRITTEN IN THE FROZEN IDENTITY:  max |Δ| < 11,  mean |Δ| < 1.25│
 │   cost:    "zero drift" wording is prohibited; the envelope must    │
 │            be re-measured per identity; logits are never claimed    │
 │            equal                                                    │
 │   rules out: token divergence, nondeterminism, unbounded or         │
 │            undeclared drift                                         │
 ├─────────────────────────────────────────────────────────────────────┤
 │ RUNG 1 — PARITY-WITHIN-NOISE  (plain decode, both lanes)            │
 │   NVFP4 35.491 tok/s vs adjacent kquant 35.440 — a 0.14% split      │
 │   inside 0.03–0.13% CVs; "never call decode faster"                 │
 │   cost:    no throughput claim may lean on the split                │
 │   rules out: the idea that either lane is the other's inferior      │
 └─────────────────────────────────────────────────────────────────────┘
```
*Figure 32.1: The trust ladder. Rung citations: rung 3 `[claims #9;
measured-numbers §1l]`; rung 2 `[scripts/gx10/vllm/native_onboarding_
identity_v1.json:57-71; crates/muser-server/src/node/smoke.rs:451-469]`;
rung 1 `[ledger §P1.3; claims #11]`. A lane picks its rung by declaring
it at enrollment — it cannot drift between rungs unnoticed, because the
rung is part of the identity (§32.3).*

The rest of the chapter climbs the ladder from the middle out: rung 2 is
the product lane and the most instructive (§32.2–32.3), rung 3 is what
the wizard bought with its ABI chase (§32.4), rung 1 is where decode
lands (§32.5). Then the apparatus underneath all three: the soft cap's
role in the contract's units (§32.6), the integer-dot anchor (§32.7),
mode-separated cache identities (§32.8), the measured drift record and
deep content controls (§32.9–32.10), the reference lock (§32.11), and
the reason the Mac never *trusts* when it can re-check (§32.12).

## 32.2 Declaring the policy: the rule is written before the run

Start with the question this section answers: *when two engines cannot
agree exactly, who decides how much disagreement is acceptable — and
when do they decide it?* The second half is the part with teeth. A rule
invented once the outputs are in hand is not a rule; it is a description
of what happened, wearing a rule's clothes. So Muser's rule is written
down first, and written somewhere it cannot be quietly edited later.

The native/text lane's contract is therefore not a wiki page or a
convention; it is a field in the frozen onboarding identity that both
peers pin:

```json
// scripts/gx10/vllm/native_onboarding_identity_v1.json:57
"onboarding_qualification": {
  "schema": "muser.native-text-onboarding.v1",
  "prompt_tokens": 2048,
  "output_tokens": 256,
  "repetitions": 3,
  "prompt_sha256": "149ac0d9c37c957823e53c0637b52a38f2ac601089dbda9f98eec4bc5f369030",
  "require_exact_tokens": true,
  "require_deterministic_remote_output": true,
  "full_logits": {
    "mode": "bounded-drift",
    "maximum_absolute": 11.0,
    "maximum_mean_absolute": 1.25
  },
  "basis": "muser-receipt://nvfp4-f4-native-text-p4-accelerator-20260817/20260817T074427Z-44c761545afa47d78f8407734f8d2145.command.log"
}
```

Deconstruct it. **Tokens are exact** — `require_exact_tokens: true`, no
tolerance, no exceptions. **The remote output must be deterministic
across repetitions** — a producer whose own output wobbles cannot be
gated at all. And **full logits are bounded, not equal**: the mode is
literally named `bounded-drift`, with two numbers — a maximum absolute
delta (`11.0`) and a maximum mean absolute delta (`1.25`) — plus a
`basis` receipt tying the numbers to the measurement they came from.
This is what "the special care" means in practice: the tolerance is
declared *before* any qualification run, in a file whose SHA-256 is part
of the lane's identity, not chosen after seeing the outputs.

And the Mac side does not merely read that file — it re-derives it
against a sealed constant and refuses to proceed if they disagree:

```rust
// crates/muser-server/src/node/artifacts.rs:188
const NATIVE_SEALED: SealedNativeIdentity = SealedNativeIdentity {
    // … checkpoint, image, adapter, consumer, tokenizer, template,
    // context-policy, RoPE-cache digests …
    target_cache_identity_sha256:
        "a3bbd72fc16116322b5c9dc701f155d35349146b2af3e3b8465732b7df1eabd0",
    prompt_sha256: "149ac0d9c37c957823e53c0637b52a38f2ac601089dbda9f98eec4bc5f369030",
    // …
    maximum_logit_absolute: 11.0,
    maximum_logit_mean_absolute: 1.25,
};

// crates/muser-server/src/node/artifacts.rs:472
        if qualification.schema != "muser.native-text-onboarding.v1"
            || qualification.prompt_tokens != 2_048
            || qualification.output_tokens != 256
            || qualification.repetitions != 3
            || !qualification.require_exact_tokens
            || !qualification.require_deterministic_remote_output
            || qualification.full_logits.mode != "bounded-drift"
            // … finite, positive, and exactly the sealed values …
            || qualification.full_logits.maximum_absolute != NATIVE_SEALED.maximum_logit_absolute
            || qualification.full_logits.maximum_mean_absolute
                != NATIVE_SEALED.maximum_logit_mean_absolute
            || qualification.basis.is_empty()
        {
            return Err("native onboarding qualification rule is invalid".into());
        }
```

*(lines 188–210 and 472–488 elided in the middle: the digest pins; see
file)*. Notice what that closes off. An operator who "relaxes" the rule
to 12.0 in the JSON now fails onboarding against the compiled constant —
the tolerance can only change through a code change that says so, in a
commit, in front of a reviewer. Widening a bound stops being a
configuration decision and becomes an edit to the product.

Where did the two ceilings come from, then? Not from taste. We measured
the lane first and set the rule a hair above what we saw. The
fast-vs-exact envelope on the 2,048/256 five-repetition comparator was
max/mean **10.884401 / 1.233789** with 100% token agreement in all five
runs, and 7.270581 / 1.040619 on the 32-token standard fixture
`[docs/nvfp4-fast-lane-evidence-20260817.md
§Determinism]`. Sitting the ceiling that close to the observed envelope
is a trap we set for ourselves on purpose: a lane that drifts even
slightly worse than the lane we measured does not squeak through, it
fails onboarding. It is also why the claims register prohibits the
flattering framing — the fixture "may be described as deterministic and
token-identical only with its scope; zero drift … remain[s] prohibited"
`[claims #10]`. We are allowed to say the drift is bounded. We are not
allowed to say there is none.

## 32.3 Checking the policy: three ordered handoffs, per sample and per summary

A declared rule is only as good as its enforcement, so the next question
is: *who reads the rule at run time, and what happens to a run that
misses it?*

The wizard's smoke step runs the qualifier three times. What the
operator watches scroll past is the contract restated in English —
"three ordered 2,048/256 exact-token handoffs with bounded full-logit
drift" `[crates/muser-server/src/node/smoke.rs:43-52]` — driven by the
native recipe's own flags,
`--onboarding-native --drift-graded --reference-once`
`[crates/muser-server/src/node/smoke.rs:644-650]`. The middle flag is
the interesting one. Left alone, the qualifier *refuses* on any logit
digest mismatch at all `[crates/muser-bench/src/remote.rs:690-694]`,
which is the correct default for a lane claiming equality and useless
for a lane claiming a bound. `--drift-graded` trades refusal for
measurement: retain the comparison, compute the deltas, then judge them.
And the wizard judges twice — once per sample, once over the summary
arrays:

```rust
// crates/muser-server/src/node/smoke.rs:451
        if let Some(rule) = native {
            let maximum = value
                .get("remote_local_logit_max_abs")
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| "native sample omits maximum full-logit drift".to_string())?;
            let mean = value
                .get("remote_local_logit_mean_abs")
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| "native sample omits mean full-logit drift".to_string())?;
            if !maximum.is_finite()
                || !mean.is_finite()
                || maximum > rule.full_logits.maximum_absolute
                || mean > rule.full_logits.maximum_mean_absolute
            {
                return Err(format!(
                    "native full-logit drift exceeds its identity rule: max={maximum}, mean={mean}"
                ));
            }
        }
```

The summary-level check then re-verifies that all three repetitions
agree on token rate exactly 1.0, that every per-repetition maximum and
mean sits inside the rule, and that the fast generated-token digest is
present `[crates/muser-server/src/node/smoke.rs:538-569]`. A missing
field is a refusal, not a zero ("native sample omits maximum full-logit
drift"). Put the same idea the other way round, because it is the one
readers hand-wave past: a sample that forgot to report its drift is not
a sample with no drift. Silence is not a measurement. Absent evidence is
failed evidence — the same fail-closed posture as everywhere else in
this book.

All of which is machinery until it meets a real node. The measured
result of running exactly this gate live: native/text attempt 9 passed
all seven progress labels with three exact-token handoffs under the
bounded-logit rule (deltas inside 10.884401 / 1.233788776) at payload
rates 6.866 / 6.976 / 6.708 Gbps, and the node reached `healthy`. The
run that proved it is retained: `[claims #9; measured-numbers §1l; receipt
wizard-validation-20260823/attempt-9-native-live-20260824T051305Z/validation-summary.json]`.

## 32.4 Rung 3: what bit-exact costs

One rung up sits the combined kquant lane, and it asks for the whole top
box of Figure 32.1: exact target tokens, exact *full* logits — the
digests equal, deltas zero — and exact DFlash tokens and trace. This is
the rung we assumed would be a scheduling problem. It was not, and the
story of finding that out is the most useful thing in the chapter.

With the transport already trustworthy, we expected the remaining work
on the combined lane to be plumbing: line the sequence up, pin the
fixtures, watch the digests agree. The digests did not agree, and the
wizard spent **attempts 10 through 31** finding out why. The trigger was
as small as a divergence can be — a single F16 ULP in layer-1 V, the
last place anyone wants to go hunting — and by the time it had
propagated through the stack it had become 51.7 M differing logits
`[ledger §2b, 2026-08-24]`. That escalation is the point: at the top of
the ladder there is no such thing as a small disagreement, because
nothing downstream damps it.

What the chase found underneath were the two reduction-order splits
named in this chapter's opening, and no amount of retrying was going to
dissolve them. They had to be *specified*. So the fix was a versioned
cross-vendor arithmetic ABI, and we know what it cost: 4–7
accelerator-hours `[measured-numbers §1l]`. Attempt 31 then passed 7/7
with bit-exact logits and payload rates 9.812 / 8.887 / 8.690 Gbps
`[claims #9; receipt wizard-validation-20260823/attempt-31-combined-full-20260824T132639Z/validation-summary.json]`.

The lesson generalizes the integer-exact philosophy of Chapter 7 §7.4.2:
when you cannot reproduce another engine's reduction *topology*, you
either (a) pin an ABI that makes the controllable parts
associativity-free — the a16-q8 kernel's trick — or (b) chase and pin
each divergence site until the digests match, which is what the wizard
did. What you may not do is (c) declare a tolerance and quietly widen it
each time a cell fails. The 4–7 accelerator-hours are the price of (b),
and the receipt chain is what makes the spend auditable.

## 32.5 Rung 1: parity-within-noise, never "faster"

At the bottom of the ladder, the plain decode lanes — where the question
is not whether the two lanes agree, but whether either one is entitled
to brag. The paired P1.3 cells measured native NVFP4 decode at
**35.490711722 tok/s** (CV 0.130%) against an adjacent kquant control at
**35.439527527 tok/s** (CV 0.037%) — a +0.1444% split inside the noise
of cells whose CVs bracket it
`[ledger §P1.3]`. The claims register's standing instruction is three
words long: "Never call decode faster" `[claims #11]`. The trust-relevant
content of rung 1 is that neither lane is entitled to be called the
other's quality fallback by speed arguments — the native lane's
*differences* from kquant are quality-shaped, not speed-shaped, and
§32.10 is where they live.

## 32.6 The soft cap is part of the contract's units

A tolerance is a number *and* a unit, and the unit is the part everyone
forgets. So before trusting a bound, ask what it is measured in.
Chapter 20 §20.7 proved the soft cap is order-preserving (greedy tokens
cannot move) but gap-compressing — by hand, a 6.00 raw gap becomes 4.915
capped, a 20.0 gap becomes 4.049. Carry that into this chapter: every
number in the `bounded-drift` rule is a delta of *capped* logits, because
both engines apply `1/√26` then `tanh@20` as the last step before these
bytes are compared `[Ch 20 §20.4, §20.7]`. That has two consequences you
must hold simultaneously. First, the bounds 11.0/1.25 are *units of the
contract* — recalibrate either engine's scale-and-cap order and the
tolerance is meaningless, which is why the whole transform is pinned in
both engines. Second, the cap compresses exactly the large-logit regime
where the two lanes' numerics disagree most, so the measured drift
envelope is *smaller* than an uncapped comparison would produce — the
bound is doing quiet work, and "without the cap the same deltas would be
much larger and the tolerance would have to be re-derived" `[Ch 20
§20.7]`. A reader comparing Muser's 11.0 against some other engine's
uncapped logit tolerance is comparing different units — do not.

## 32.7 The integer-dot anchor: a producer that exists to be compared against

Every ladder needs a plumb line. "Bounded drift" is a claim of the form
*no farther than this from something* — and the something has to be an
engine we can re-run on demand, or the bound is unfalsifiable. Muser's
plumb line is the **exact producer mode**: the same GX10 node, the same
transport, but a producer whose NVFP4 arithmetic is integer-dot
deterministic — built to be *compared against*, not served. The shipped
lane matrix gives it its own row:

| Lane | Prefill | Decode | Speculative | Intended use |
|---|---|---|---|---|
| Native NVFP4 | Spark tensor-core FP4 | Mac NVFP4, 35.491 tok/s | Rejected (fail-closed) | Fast product lane |
| kquant/reference | Reference path | kquant, 35.440 tok/s | 107.9 tok/s | Speculative + reference lock |
| Exact NVFP4 flag | Integer-dot verification producer | Mac NVFP4 | Verification only | Deterministic anchor |

`[docs/muser-architecture.md, lane matrix]`

The mode is selected **producer-side only**, by the Python environment
flag `MUSER_NVFP4_EXACT=1`. It does not exist anywhere in the Rust tree
— a Mac-side reviewer will find nothing to set — and the native
benchmark *refuses to run* with it set, because mixing modes would
invalidate the claim being measured:

```python
# scripts/gx10/vllm/benchmark_native_prefill.py:98
    # The benchmark is intentionally stock: importing muser_vllm exact modules
    # or setting MUSER_NVFP4_EXACT would invalidate the native-path claim.
    if os.environ.get("MUSER_NVFP4_EXACT") == "1":
        parser.error("native benchmark refuses MUSER_NVFP4_EXACT=1")
    os.environ["MUSER_NVFP4_EXACT"] = "0"
```

The resident native daemon likewise pins `MUSER_NVFP4_EXACT=0` into the
container's environment `[scripts/gx10/vllm/muser_native_prefilld.py:446]`.
(Chapter 7 §7.7 introduced this split; the point to add here is what the
anchor is *for*.) The exact lane's value is that its outputs are stable
enough to be a reference: the G3 live recheck re-ran the retained strict
cell after routing changes and reproduced **every** retained digest
bit-for-bit — generated-token SHA, full-logit digests, payload SHA, all
52 seam digests, KV max deltas 9.625/18.4580078125, logit errors
7.270581/1.040619, 32/32 tokens `[docs/nvfp4-fast-lane-evidence-20260817.md
§G3]`. When you ask "how far has native drifted?", you are asking it
*against this lane's numbers*.

## 32.8 Mode-separated cache identities, and refusing the unknown

An anchor is only an anchor if you can tell it apart from the thing it
anchors. Bounded drift is safe only when native KV and exact KV can
never be mistaken for each other — a cache entry produced by tensor
cores must not be served to a session whose contract says "integer-dot
anchor," because such a session would then be measuring drift against
drift and reporting the answer as ground truth. The fast-lane evidence
note states the rule in one line: exact and native
producers "use mode-separated target-cache identities, so exact and
native KV entries cannot alias" `[docs/nvfp4-fast-lane-evidence-20260817.md
§Product route]`. Concretely, the receiver configuration carries a
`target_cache_identity_sha256` field that is validated as a digest at
load `[crates/muser-cluster/src/config.rs:41-42, 103-115]`, and the
sealed native identity pins the exact value (`a3bbd72f…`,
`[crates/muser-server/src/node/artifacts.rs:206-207]`). A cache identity
is a *property of the recipe*, not of the model file alone.

The same fail-closed logic covers recipes Muser has never heard of.
Enrollment maps a producer lane to exactly one qualification contract,
and an unknown lane dies before any key is minted:

```rust
// crates/muser-server/src/node/registry.rs:39
/// Exact qualification contract selected by the enrolled producer lane.
/// Keeping this exhaustive beside `ProducerKind` means adding a lane without
/// choosing a recipe is a compile error; an unknown serialized lane is
/// refused while loading the registry, before enrollment can mint keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualificationRecipe {
    KquantTargetPlusDflash,
    NativeText,
}
```

Two recipes exist — `KquantTargetPlusDflash` (the rung-3 combined lane)
and `NativeText` (the rung-2 bounded-drift lane)
`[crates/muser-server/src/node/registry.rs:49-76]` — and the mapping is
total: `Llamacpp → KquantTargetPlusDflash`, `Native → NativeText`.
Because the compiler forces the match to be exhaustive, a future lane
must *choose* a recipe to compile; because loading refuses unknown
serialized values, an old registry file cannot silently re-interpret
itself. The architecture document's summary is the sentence to quote
whole: "Exact and native producers derive different target-cache
identities; an unknown recipe is refused at enrollment"
`[docs/muser-architecture.md §Durable and remote KV]`. (The native lane
additionally cannot enroll a DFlash identity at all
`[crates/muser-cluster/src/config.rs:128-131]` — that refusal is Chapter
33's opening subject.)

## 32.9 The measured drift record: what bounded actually means

A bound with no published measurement behind it is empty theater: it
tells the reader what we promised, never what actually happened. So here
is the drift the rule bounds. The fast-lane evidence note's determinism
section is that record (Figure 32.2), for the standard 2,048-token
prompt with 32 greedy tokens `[docs/nvfp4-fast-lane-evidence-20260817.md
§Determinism]`:

| Quantity | Measured envelope (fast vs exact) |
|---|---:|
| Greedy token agreement | 32/32, 100% |
| First divergent token | none |
| Full-logit maximum absolute error | 7.270581 |
| Full-logit mean absolute error | 1.040619 |
| KV key maximum absolute delta across 52 layers | 9.625 |
| KV value maximum absolute delta across 52 layers | 18.458008 |
| Per-layer KV key mean-absolute range | 0.015648–0.431876 |
| Per-layer KV value mean-absolute range | 0.044231–0.387556 |

*Figure 32.2: The standard-fixture drift envelope. The KV deltas are the
raw material of everything downstream: 9.625 on keys and 18.458 on
values, per layer, feeding attention for every future token — and the
greedy tokens still agree 32/32.* The 2,048/256 five-repetition
comparator extended the same result — 100% token agreement in all five
runs, max/mean 10.884401/1.233789 — and immediately disclaims its own
scope: "This demonstrates deterministic, coherent output on the standard
fixture; it does not prove a general zero-divergence contract"
`[docs/nvfp4-fast-lane-evidence-20260817.md §Determinism]`. And the seam
itself is reproducible: repeated A requests, A/B/A queue interleaving,
and a full engine restart reproduce the same 52 layer hashes, the same
seam SHA-256, and the same payload SHA-256 `[docs/nvfp4-fast-lane-
evidence-20260817.md §Determinism]`. Drift, yes — but *deterministic*
drift, which is the difference between a risk you can bound and one you
can only fear.

## 32.10 Deep content controls: the sensitivity that was published, not hidden

A 2,048-token fixture is a shallow probe of a 131,072-position model. So
the honest next question was the uncomfortable one: what is happening
further out, where we had not yet looked? The deep ladder went looking,
and this is the stretch of the record that contains a row that fails. We
walked into it in three stages, and the order we walked them in matters
as much as the outcome:

- **E1 calibrated the yardstick first.** Before judging native against
  kquant, the campaign measured an *accepted alternate quant* (Q6) against
  kquant on the same rows and set each cell's disagreement gate from that
  two-sided 95% Wilson bound plus two points — calibrated gates of
  8.796–15.299% across the long-context cells. Native exceeded the band
  by 1.746–1.996 points at 8k/16k/32k `[docs/nvfp4-fast-lane-evidence-
  20260817.md §E1]`. A tolerance derived from a second real quantization
  cannot be accused of being tuned for convenience.
- **E2 asked whether length was the culprit, and the answer was no.**
  The natural hypothesis — written down before the run, which is what
  makes the answer worth anything — was that native drift grows with
  context: past some length the lane goes bad, and you route around it
  with a number. E2 held content fixed across three nested documents
  (rust, python/shell, docs) and varied length. Rust and python passed
  every disagreement row at every length 2k–32k; only the **docs**
  document exceeded its gate at 8k/16k/32k, and the 512-token position
  profiles kept the sensitive regions localized rather than showing a
  length transition — so the preregistered replicated-length criterion
  was false `[docs/nvfp4-fast-lane-evidence-20260817.md §E2/E3]`. The
  tidy story we expected was not on offer. What we had instead was a
  *content* effect, which is far harder to route around than a length
  threshold, because content is not a knob on the request.
- **The stage-3 yardstick published the exceedance.** At 65,536 tokens
  on the docs corpus, native-vs-kquant top-token disagreement was
  **15.134% against a calibrated gate of 13.339%** (relative PPL +4.227%)
  — recorded, published, and explicitly scoped: a *content-local
  sensitivity* that "did not replicate cross-document"; persistence at
  131,008 could not be tested because the corpus is too short `[claims
  #10; receipts kvpack-ladder-20260820/attempt-5-…-stage3-compact/stage3-e2-quality/]`.

So what do you do with a row like that? Before publishing it, we tried
to make it go away. The route-exhaustion matrix put eight native vLLM
runtime variants (chunked prefill, BF16, Triton, batch-invariant
CUTLASS, FlashInfer B12X/cuDNN, engine ceiling) against the docs 65,536
row, on the expectation that one of them was the real culprit and the
sensitivity was a configuration mistake wearing a numerics costume. All
of them failed identically or worse, and no runtime change was promoted
`[ledger §"native-route exhaustion"; measured-numbers §1i]`. There was
nothing left to out-configure.

That left two cheap ways out, and the record rejected both. Hiding the
row would violate the publish-the-sensitivity rule
`[measured-numbers §6 rule 9]`. Capping context — the "native up to N,
kquant beyond N" policy — sounds principled right up until you look at
which cells actually fail: a cap would have silently included the
failing 8k cell while failing to cover the passing deeper cells, the
non-monotonicity that killed exactly that policy back in the D1 routing
ladder `[docs/nvfp4-fast-lane-evidence-20260817.md §D1]`. So no context
cap was imposed through the measured 32k range, and the sensitivity was
published instead.

## 32.11 The reference lock: an explicit lane, not an automatic proxy

What is a user who *does* care about the docs-class sensitivity supposed
to do? The answer is deliberately manual: "Users who need a reference
lock select the existing kquant lane explicitly"
`[docs/nvfp4-fast-lane-evidence-20260817.md §E2/E3]`; the architecture
document repeats it as a first-class property — "the kquant lane is the
explicit reference lock for that class" of high-entropy
numeric/digest/tabular documentation content `[docs/muser-architecture.md,
lane-matrix paragraph]`. Why explicit? Because no *automatic*
disagreement proxy is claimable — building one "would require a second
reference computation" `[docs/nvfp4-fast-lane-evidence-20260817.md
§E2/E3]`, i.e., the proxy would itself need the very reference it is
pretending to replace. The same logic explains the lane matrix's
"kquant/reference" row: a reference lock is a lane you can *route to*,
whose own parity is maintained against the pinned llama.cpp comparator
(the parity ledger is [Ch 38](38-measuring-against-llama-cpp.md)'s
subject), not a background oracle.

## 32.12 Why the Mac re-checks rather than trusts

Everything above describes contracts the *producer* satisfies. The final
layer of care is that the Mac never takes even a satisfied contract on
faith at qualification time. The remote qualifier's module doc states
the design in four lines — it uses "the same `RemoteReceiver` as
serving" (so a benchmark cannot bypass the machinery), and "every sample
performs cold local recomputation and an authenticated remote install,
compares 256 greedy tokens plus every full target-logit row, and retains
producer/transport phase times needed to prove real overlap"
`[crates/muser-bench/src/remote.rs:3-8]`. The trust direction matters:
the remote-installed KV is decoded *and compared* against a local cold
recompute, token by token and logit row by logit row, before a lane is
called qualified. The producer is not asked to certify itself; it is
checked against a computation the Mac performed on its own.

Serving then inherits the disposition, not the measurement: once
enrolled, the lane's correctness lives in its identity and gates —
and the one machinery that must never trust a *draft* in flight,
speculative decoding, is under an absolute rule: "DFlash drafts are
always verified by target distributions" `[docs/muser-architecture.md
§Model and engine]`, with the remote qualifier enforcing an acceptance
floor of `DFLASH_ACCEPTANCE_MINIMUM = 0.95`
`[crates/muser-bench/src/remote.rs:33]`. The all-accept diagnostic that
proved the native target+DFlash path *correct* (240/240 drafted tokens,
canonical digest retained) is carefully recorded as establishing
"correctness of the diagnostic, not competitive speculative throughput"
`[docs/nvfp4-fast-lane-evidence-20260817.md §G1 tail]` — correctness and
speed claims are never allowed to borrow each other's evidence.

## 32.13 Tradeoffs

**Bounded drift vs bit-exactness.** Rung 2 gives up logit equality and in
exchange gets a lane the tensor cores can actually serve at speed; rung 3
gets equality and pays an arithmetic ABI plus per-divergence-site pinning
(4–7 accelerator-hours for the wizard's fix `[measured-numbers §1l]`).
The measured consequence of choosing rung 2 honestly: a drift envelope
of 10.884401/1.233789 published next to the tokens-exact claim, and the
prohibition on "zero drift" wording `[claims #10]`. The measured
consequence of rung 3: digests equal, deltas zero, at 9.812/8.887/8.690
Gbps `[claims #9]` — but only after the ULP chase.

**Publishing the sensitivity vs capping the lane.** The docs@65,536
exceedance (15.134% vs 13.339%) could have been "fixed" by a context
cap; the D1 ladder measured why that fails (non-monotonic, no honest N
exists) and E2's preregistered criterion formally rejected the
length-effect interpretation `[docs/nvfp4-fast-lane-evidence-20260817.md
§D1, §E2/E3]`. What publishing costs: one uncomfortable row in the
record, carried forever. What it rules out: a user unknowingly serving
the sensitive class on the native lane with no way to know
`[measured-numbers §6 rule 9]`.

**Explicit reference lock vs automatic proxy.** An automatic
disagreement-triggered fallback was rejected because the proxy needs a
second reference computation to be trustworthy
`[docs/nvfp4-fast-lane-evidence-20260817.md §E2/E3]`. The cost of the
explicit lock is that routing is a human decision; the benefit is that
no silent heuristic ever re-ranks the lanes on unaudited numbers.

**Integer-dot anchor vs serving it.** The exact producer is qualified
enough to reproduce every retained digest bit-for-bit
`[docs/nvfp4-fast-lane-evidence-20260817.md §G3]` and slow enough that
an attempted 33,024-token exact-lane scoring pass was stopped during
warmup when "the measured approximately 50x compute ratio" showed it
would spend the experiment box just warming the graph `[docs/nvfp4-
fast-lane-evidence-20260817.md §G1 intro]`. The anchor is a plumb line,
not a product — which is why its matrix row says "Verification only"
`[docs/muser-architecture.md]`.

## 32.14 What comes next

The trust ladder now stands complete: bytes intact (Chapter 31), tokens
exact, logits either equal or bounded-by-declared-rule, identities
mode-separated, sensitivities published, and a reference lane one switch
away. One precision machinery in this system remains unexamined — the
one that makes a *guess* participate in exact inference: speculative
decoding, where a five-layer draft proposes and the 52-layer target
disposes, and where the same evidence culture had to pass a verdict on
its own distributed variant. That is
[Ch 33](33-speculation-and-the-distributed-verdict.md): the local win,
the fail-closed refusal, and the measured rejection — with receipts.

---

## References

- `[scripts/gx10/vllm/native_onboarding_identity_v1.json:57-71]` — the
  declared `bounded-drift` rule (11.0 / 1.25), exact-token and
  determinism requirements, and its basis receipt.
- `[crates/muser-server/src/node/artifacts.rs:188-210, 472-488]` —
  `NATIVE_SEALED` (incl. `maximum_logit_absolute: 11.0`,
  `maximum_logit_mean_absolute: 1.25`, the `a3bbd72f…` cache identity)
  and the validation that refuses any other qualification rule.
- `[crates/muser-server/src/node/smoke.rs:43-52, 451-469, 538-569,
  644-650]` — the recipe's progress label, per-sample and summary
  bounded-drift checks, and the `--onboarding-native --drift-graded
  --reference-once` wiring.
- `[crates/muser-bench/src/remote.rs:3-8, 32-33, 690-694]` — the
  qualifier's cold-recompute + 256-token + full-logit-row comparison,
  `LINK_GBPS_MINIMUM`/`DFLASH_ACCEPTANCE_MINIMUM = 0.95`, and the
  non-graded refusal on logit digest mismatch.
- `[crates/muser-cluster/src/config.rs:41-49, 103-115, 128-131]` —
  `target_cache_identity_sha256` validation; native mode cannot enroll
  DFlash geometry.
- `[crates/muser-server/src/node/registry.rs:29-76]` —
  `ProducerKind`/`QualificationRecipe`, the exhaustive mapping, and the
  unknown-lane refusal before key minting.
- `[scripts/gx10/vllm/benchmark_native_prefill.py:98-102]`,
  `[scripts/gx10/vllm/muser_native_prefilld.py:446]` —
  `MUSER_NVFP4_EXACT` is producer-side Python only; the native benchmark
  refuses it set; the daemon pins `=0`.
- `[docs/nvfp4-fast-lane-evidence-20260817.md]` — §Product route
  (mode-separated identities, Fallback B), §Determinism (the drift
  envelope and its disclaimer), §G1/§D1/§E1/§E2-E3 (widened fixtures,
  yardstick calibration, content controls, routing disposition), §G3
  (bit-for-bit anchor recheck).
- `[docs/muser-architecture.md]` — the lane matrix (exact-flag row:
  "Verification only"), "unknown recipe is refused at enrollment,"
  "DFlash drafts are always verified by target distributions," the
  kquant reference-lock paragraph.
- `[claims #9]`, `[claims #10]`, `[claims #11]` — `docs/launch-claims.md`:
  wizard attempts 9/31 with their exactness policies and rates; the
  docs@65,536 sensitivity scope and prohibited "zero drift" wording;
  35.491/35.440 parity scope ("Never call decode faster").
- `[ledger §P1.3]`, `[ledger §2b 2026-08-24]`,
  `[ledger §"native-route exhaustion"]` — the paired decode cells; the
  one-ULP wizard chase; the eight-variant exhaustion matrix.
- `[measured-numbers §1i, §1l, §2 Arc 4, §6]` — the quality-gate table,
  wizard validation cells, the arithmetic-ABI arc, and the claim-
  discipline crib sheet (publish the sensitivity; evidence wins).
- [Ch 7](07-nvfp4-native-lane.md) §7.6–7.7 — Fallback B and the
  producer-mode preview this chapter builds on.
- [Ch 20](20-final-norm-lm-head-softcap.md) §20.7 — the soft cap's
  monotonicity and gap compression; why bounded-logit deltas are capped-
  logit units.
- [Ch 30](30-handoff-v2-transport.md), [Ch 31](31-the-wire-discipline.md)
  — the authentication and wire machinery underneath; [Ch 33](33-speculation-and-the-distributed-verdict.md)
  — the verification machinery that never trusts a draft.
