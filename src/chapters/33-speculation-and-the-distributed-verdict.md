# Chapter 33 — Speculation: the local win and the distributed verdict
> **status:** polished  ·  **path:** Muse Glimmer, pinned Muser tree

*Prerequisites: Chapter 8 (the DFlash draft — its context ABI, its window
bug, and the four guarantees that make exact verification possible) and
Chapter 7 (Fallback B's refusal). This chapter completes the loop the
draft opened: the accept/verify algorithm as code, the local lane's
measured scope, the rejections told with receipts, and the one
experiment still alive.*

Chapter 32 ended with a system that re-checks rather than trusts —
every lane qualified by cold recompute, every draft verified against
full target distributions. This chapter is about that last clause taken
to its conclusion. Speculative decoding is the book's most demanding
precision machinery, because it deliberately introduces an *approximate*
participant — a five-layer draft guessing what the 52-layer target will
say — and then has to prove the approximation changed nothing. And it is
the machinery Muser took furthest: to a local lane that wins on
synthetic fixtures, to a fail-closed refusal on the native lane, and to
a distributed variant that was built, measured, and rejected in a single
day, with the rejection recorded so carefully that the receipts are
still worth reading.

One framing before any number. In this chapter, **ratios are llama ÷
muser** (above 1.0 means muser wins), synthetic-versus-natural scope is
load-bearing, and one famous number — 107.9 tok/s — exists in this
record only as a *bar*, never as a result. Hold that; §33.1 explains
why.

---

## 33.1 The local win, stated with its scope language

The mechanism in one paragraph, since [Ch 8](08-the-dflash-draft.md)
§8.1 built it: decode is bandwidth-bound because each token reads the
whole 16.76 GB artifact; speculation has a cheap draft propose k tokens
so the target can *verify* them in one batched forward pass, k+1 rows of
decision per weight read. The draft is pure overhead that pays only when
its guesses are good; the target still makes every decision, which is
what keeps the output exact.

**The bar.** The campaign's Stage B verdict measured kquant DFlash
speculative decode at **107.9136 tok/s median, CV 0.200%, against
llama's 81.3047 — ratio 1.3273** (2,048+256 streamed, verify length 15,
five reps) `[ledger §L2 Stage B verdict]`. That number was measured
**before the 2026-08-21 draft-window fix** (§33.3), so it is a
landmine: it survives in the record only as the *bar* later lanes were
judged against, and the pre-fix ratios 1.3273/1.3012 are superseded
`[measured-numbers §1b, §7]`.

**The current synthetic scope.** After the fix, in retained fixed-window
synthetic packets, exact-token decode ratios are **1.23692× at 2,048,
1.20323× at 16,384, and 1.19616× at 32,768, with 5/5 exact reps per
depth** `[claims #15]`. The claims row's own instruction is part of the
claim: never generalize to natural text, native NVFP4, or untested
depths; the 8,192 and 65,536 cells of that family are single-rep
diagnostics (1.214†, 1.188†) `[measured-numbers §1b]`.

**Natural text is a different regime.** On real corpora the two engines'
outputs diverge cross-engine, so speed stands without an exactness gate,
and the picture splits: spec decode *wins* python-like content (16,384:
1.186; 8,192 suffix: 1.321) and **loses** high-acceptance shallow text
(rust at 2,048: 0.931, improving only to 0.945 at verify length 7) —
llama's lighter draft wins where acceptance is nearly free
`[docs/benchmarks.md §2]`. That asymmetry is why serving froze
verify-length 7 while the comparison harness pins 15
`[docs/benchmarks.md §2]`. And at the deepest tested scope, the funded
131,008/48 packet crossed end-to-end **wall** parity for the first time
at **1.02536×** `[claims #16]` — wall, because the tempting
**1.64960× decode figure is barred**: its first-round split is not an
accounting-neutral cross-engine per-round metric, per the 2026-08-23
ledger amendment `[claims #16; measured-numbers §7]`.

So the local lane's honest headline is: *about 1.20–1.24× on three fixed
synthetic fixtures with exact tokens, a wall-parity crossing at 131k,
wins and losses on natural text by content class* — and every clause of
that sentence has a receipt.

## 33.2 The loop as code: propose, verify, accept — exactly

Chapter 8 pinned what the draft guarantees; here is the algorithm that
consumes those guarantees. Acceptance happens **on the CPU, against full
target distributions** — the target's complete probability row per
position, not a top-k sketch:

```rust
// crates/muser-engine/src/sampling.rs:1033
pub fn verify_full_speculative_mt_ordered(
    draft_tokens: &[u32],
    draft_probabilities: &[Vec<f32>],
    target_probabilities: &[Vec<f32>],
    target_orders: &[Vec<u32>],
    rng: &mut Mt19937,
) -> Result<SpeculativeDecision, SamplingError> {
    // … geometry validation elided (row counts, widths, token bounds) …
    for (index, (&token, (draft, target))) in draft_tokens
        .iter()
        .zip(draft_probabilities.iter().zip(target_probabilities))
        .enumerate()
    {
        let token = token as usize;
        let q = draft[token];
        let p = target[token];
        let acceptance = if q <= 0.0 { 1.0 } else { (p / q).min(1.0) };
        if rng.uniform_f32() <= acceptance {
            continue;
        }
        let mut residual = target
            .iter()
            .zip(draft)
            .map(|(&p, &q)| (p - q).max(0.0))
            .collect::<Vec<_>>();
        let total = residual.iter().sum::<f32>();
        if total <= 0.0 {
            residual.clone_from(target);
        } else {
            for probability in &mut residual {
                *probability /= total;
            }
        }
        let order = target_orders[index]
            .iter()
            .copied()
            .filter(|token| residual[*token as usize] > 0.0)
            .collect::<Vec<_>>();
        return Ok(SpeculativeDecision {
            accepted: index,
            next_token: sample_distribution_mt_ordered(&residual, &order, rng)?,
        });
    }
    Ok(SpeculativeDecision {
        accepted: draft_tokens.len(),
        next_token: sample_distribution_mt_ordered(
            target_probabilities.last().ok_or(SamplingError::Geometry)?,
            target_orders.last().ok_or(SamplingError::Geometry)?,
            rng,
        )?,
    })
}
```

*(lines 1040–1059 elided: the geometry guard that rejects any shape or
token-bound mismatch before a single RNG draw — see file.)* Walk it
once, slowly:

- For each proposed token, look up the draft's probability `q` and the
  target's `p` at the *same* token. Accept with probability
  `min(p/q, 1)` — the maximal-coupling rule behind Leviathan et al. and
  the independent DeepMind formulation `[arxiv:2302.01318; frontier
  §"The target identity invariant"]`. One `uniform_f32()` per attempted
  draft, always consumed in the same order.
- On the first rejection, stop. Build the **residual** distribution
  `max(p − q, 0)`, renormalize, and draw the replacement token from it.
  This correction is what makes the *marginal* output distribution equal
  the target's exactly — rejection is not an error, it is part of the
  sampler.
- Every draw comes from a source-pinned MT19937 stream deliberately
  isolated from the generic RNG so "a `rand` algorithm or conversion
  change" cannot alter tokens or a persisted session frontier
  `[crates/muser-engine/src/sampling.rs:1001-1007]`, and the ordered
  variant (`_mt_ordered`) walks each row in a precomputed token order so
  the cumulative sum is deterministic.

The engine side overlaps draft work with the target's suffix
(Figure 33.1). The Metal mirror-SD route splits the target graph at a
capture layer:

```rust
// crates/muser-engine/src/decode.rs:3293
    /// Execute through `capture_end` synchronously, then submit the remaining
    /// target layers and LM head without waiting. The returned hidden rows are
    /// exact target activations and are stable before ANE sees them. This is
    /// the narrow public-Metal half of Mirror-SD; no target result is accepted
    /// until [`Self::finish_dflash_verify_suffix`] succeeds.
    pub(crate) fn begin_dflash_verify_suffix(
```

`begin_dflash_verify_suffix` (`decode.rs:3298`) encodes embedding through
`capture_end` in one command buffer, submits the remainder without
waiting, and returns a `PendingMetalDFlashVerify` handle;
`finish_dflash_verify_suffix` (`decode.rs:3635`) then waits, reads back
the full `token_count × vocab` logit block, advances `n_past`, and
returns the distributions that `verify_full_speculative_mt_ordered` will
judge `[crates/muser-engine/src/decode.rs:3635-3666]`. The acceptance
decision itself never touches the GPU — by design, the exactness-critical
step runs where every rounding decision is visible.

```
 one speculative round (verify length 3 shown; real lanes use 3/7/15):

 CPU/GPU time ──────────────────────────────────────────────────────►
 DFlash draft   │███ block forward + argmax → proposals t1 t2 t3
 Target (Metal) │        │──── begin: layers 0..capture_end ────┐
                │        │     (hidden rows captured here)      │ async submit
                │        │◄── exact activations ────────────────┤ layers ..51
                │        │                                      │ + LM head
 CPU accept     │        │                   wait ──►│██│  ──────┘
                │        │                         logits rows 0..3
                │        │                         verify_full_speculative_
                │        │                         mt_ordered (CPU, MT draws)
 output         │        │                            ├── accepted prefix ──►
                │        │                            └── residual sample ──► next round
```
*Figure 33.1: The round structure. The draft proposes; the target's graph
is split so its suffix overlaps; acceptance is CPU-side against full
distributions on the pinned MT19937 stream; a rejected proposal becomes
the next round's seed via the residual sample. The checkpoint machinery
that rolls back target KV on rejection is `MetalSpeculativeCheckpoint`
(`decode.rs:213-226`, [Ch 8](08-the-dflash-draft.md) §8.6).*

The qualification gate for any of this is numeric and cold: the remote
qualifier "compares 256 greedy tokens plus every full target-logit row"
per sample, with a speculative-acceptance floor
`DFLASH_ACCEPTANCE_MINIMUM = 0.95`
`[crates/muser-bench/src/remote.rs:3-8, :33]` — exactness as a gate, and
acceptance itself as a qualified quantity.

## 33.3 The bug that rewrote every number: a precision-culture lesson

[Ch 8](08-the-dflash-draft.md) §8.4 told the draft-side story: Muser
never read `dflash.attention.sliding_window` (2,048) from the sidecar and
hardcoded sink 64 + window 1,024, so the draft ran on half its trained
window for an entire campaign. This chapter owns the *consequences*,
because they are a lesson about precision instruments, not about drafts.

The ledger's geometry sweep on the collapsed cell is the cleanest
falsification table in the whole campaign — same cell, only the
conditioning changed (Figure 33.2) `[ledger §"ROOT CAUSE FOUND AND
FIXED"]`:

| sink | window | muser acceptance | decode |
|---|---|---:|---:|
| 64 | 1024 (shipped) | 2.2% | 0.535 |
| 1 | 1024 | 2.2% | — |
| 1 | 32768 | 2.2% | 0.534 |
| 1 | **2048** | **72.5%** | **1.315** |
| 64 | **2048** | **72.5%** | **1.315** |

*Figure 33.2: Half the trained window collapses acceptance; sixteen
times too much collapses it equally; exactly the declared window works;
the sink is immaterial `[ledger §"ROOT CAUSE FOUND AND FIXED"]`.* The fix
(commit `a7a4d11`) took python-suffix-8,192 acceptance from **1.1% to
72.7%** (decode 0.535 → 1.322) and made every natural-text cell
token-exact — while the synthetic fixture's acceptance moved only 100% →
99.6% and its ratio *fell* 1.3012 → 1.2368, "the draft now attends 2048
context rows instead of 1024, which is real work the under-sized window
was skipping" `[ledger §"ROOT CAUSE FOUND AND FIXED"]`.

Now the precision-culture part. The defect was **invisible to the
exactness gate for the entire campaign**, because the period-8 synthetic
stream "is predictable with no context at all, scored 100% acceptance
throughout the defect's lifetime, and therefore certified a broken draft
lane" `[ledger §"ROOT CAUSE FOUND AND FIXED", consequence 2]`. The
verifier was exact; the tokens were exact; the *draft was broken* — and
nothing that compared outputs could see it, because the outputs are the
target's outputs no matter what the draft does. That is the theorem of
§33.2 read as a warning: losslessness means a bad draft costs speed, not
correctness, so *no exactness gate can detect a bad draft*. Detecting it
requires fixtures whose acceptance depends on conditioning — natural
text — which is why natural-text cells became a standing part of the
matrix despite carrying no exactness gate `[ledger §"ROOT CAUSE FOUND
AND FIXED", consequence 2]`. And before the root cause was found, two
earlier hypotheses were falsified *with counters* and recorded as wrong
(the governor — "correctly protecting throughput"; the "window
eliminated" row — invalid because `reset()` rebuilt the cache at the
hardcoded geometry, so the override never took effect) `[ledger
§"ROOT CAUSE FOUND AND FIXED", consequence 3]`. Every retroactive
restatement in §33.1 exists because of this table.

## 33.4 The first rejection: native NVFP4 speculation, fail-closed

The draft is kquant-only, and the reason is the *target's* verify
arithmetic, not the draft. [Ch 7](07-nvfp4-native-lane.md) §7.6 gave
the full treatment; the receipts, once more, because they set the
pattern for §33.5: a native NVFP4 W4A4 batched-verification diagnostic
ran at **6.805 tok/s** against the 107.9 bar — one diagnostic,
explicitly unqualified — with verification consuming **35.915 s of a
37.619 s decode span** `[docs/nvfp4-fast-lane-evidence-20260817.md
§Measured product numbers; ledger §F-series remediation]`. The claims
register's wording discipline: the lane's verifier diagnostics "missed
the qualified bar," and native NVFP4 speculative decode "has no launch
claim and remains fail-closed" `[claims #4]`.

Fail-closed here means *code*, in two layers. A receiver configuration
declaring `producer_mode: "native"` cannot even enroll a DFlash identity
— `"native producer mode cannot enroll DFlash context geometry"`
`[crates/muser-cluster/src/config.rs:128-131]` — and the server refuses
the combination at startup with an operator-facing message
`[crates/muser-server/src/state.rs:1667-1678]`, quoted in
[Ch 7](07-nvfp4-native-lane.md) §7.6: *"native NVFP4 fast-lane
speculative decode is unqualified; omit --dflash and use plain NVFP4
decode, or route speculative serving to the kquant lane."* The design
choice worth meditating on: the alternative to refusing was *serving
6.805 tok/s silently* — more than five times slower than the lane's own
plain decode (35.491 tok/s) and nearly sixteen times below the 107.9
bar `[claims #4; ledger §F-series remediation]`. Fallback B is
fail-closed beats silently-slow, instantiated.

## 33.5 The distributed verdict: measured, with receipts

The largest question this machinery ever faced: if the Mac drafts, could
the **GX10 verify** — putting the authoritative target transition on the
node whose tensor cores are idle after prefill? The frontier doc of
2026-08-18 opens by overturning its own prior assumption: lossless
speculative decoding does **not** require drafter and target to share a
checkpoint; it "requires one endpoint to execute the authoritative target
transition" — the other may be any approximation — so the real question
is *target identity*, and a Dudeman verifier on the GX10 consuming the
same RedHat-produced prefix KV is admissible `[docs/nvfp4-distributed-
speculative-frontier-20260818.md §Decision]`.

**The screen that set the bar.** Thirty-one warm prefix-cached GX
Dudeman M16 runs, with the five f32 DFlash target layers pinned in host
memory, measured **107.152 ms median** target wall (p95 107.947 ms).
Charging the already-measured 26.9 ms Mac draft, 0.78 ms RTT, ~4.37 ms
to move 2,129,920 capture bytes, and ~0.01 ms for sparse q, full
acceptance projects to **114.93 tok/s** — which established the
preregistered requirement: **at least 99.151% IID per-edge acceptance**
(99.229% at p95) to beat 107.9 `[docs/nvfp4-distributed-speculative-
frontier-20260818.md §Decision]`. Read that number twice. The lane was
only interesting if the draft was right 99 times out of 100.

**The end-to-end verdict.** The organic runs then rejected the lane
(Figure 33.3) `[docs/nvfp4-distributed-speculative-frontier-20260818.md
§End-to-end linear-lane verdict]`:

| Trace | Output | Rounds | Accepted / drafted | Acceptance | Measured tok/s | Verifier-only ceiling |
|---|---:|---:|---:|---:|---:|---:|
| Standard | 512 | 35 | 477 / 477 | 100.00% | **110.59** | 125.61 |
| Documentation | 256 | 109 | 147 / 1,592 | 9.23% | 15.53 | 20.15 |
| Python | 256 | 55 | 201 / 764 | 26.31% | 11.17 | 40.04 |
| Rust | 256 | 39 | 217 / 570 | 38.07% | 15.41 | 55.96 |

*Figure 33.3: The linear-lane verdict table. The standard trace is the
all-accept control; the three organic strata are real content. Every
cell has retained Mac and GX service receipts (SHA-256 pairs in the
frontier doc's receipt table).*

The **verifier-only ceiling** is the decisive bound, and its definition
is the whole trick: `output_tokens / sum(GX verifier wall)` — it grants
**zero** time to DFlash drafting, feature decode, transport,
installation, and scheduling, i.e., "physically impossible zero-Mac-cost"
assumptions `[docs/nvfp4-distributed-speculative-frontier-20260818.md
§End-to-end linear-lane verdict; claims #14]`. All three organic strata
stay below the 107.9 bar *even under that impossible grant* (20.15 /
40.04 / 55.96 tok/s); documentation stays below even the 35.5 tok/s
plain-decode floor. The measured end-to-end numbers (15.532 / 11.172 /
15.412 tok/s) are point estimates only — the python and rust walls
overlapped unrelated local validation work — but that cannot weaken an
upper bound that was already under the bar `[frontier §End-to-end
linear-lane verdict; measured-numbers §1g]`. And the 110.59 tok/s
standard-trace cell is a **positive control under forced acceptance** —
it proved the machinery worked (477/477 proposals committed, 34/34
Mirror-SD speculative transactions retained), and it is *never* a
serving result `[claims #14; measured-numbers §7]`. For contrast,
llama's own draft-dflash accepted 65–81% on the same fixture families —
the collapse was muser-side draft conditioning at the time (this was
pre-window-fix, §33.3) — but the frontier's conclusion does not rest on
that: the organic ceilings sit below the bar regardless
`[measured-numbers §1g]`.

**The fallbacks were measured too.** Adaptive width is safe but no
escape: singleton verification — width reduced to the carried
`[frontier]` token alone — costs one Dudeman weight stream per emitted
token and runs "a roughly 9–10 tok/s safety mode" `[frontier §End-to-end
linear-lane verdict]`. Silently switching
to Mac plain decode is *not exact* — Mac/Metal and GX/vLLM are distinct
target-engine epochs, and a legitimate switch would need a signed
epoch seam plus replay of the accepted suffix; even granting that
unbuilt handoff zero cost, one failed gamma-14 probe followed by the
measured 35.491 tok/s Mac lane projects only ~35.16 tok/s over 512
tokens, below the documented >35.5 fallback gate `[frontier §End-to-end
linear-lane verdict]`. Mirror-SD itself missed on documentation and
python's first attempts, and rust — "the adversarial classifier case" —
passed its first 14 proposals and failed the second Mirror attempt, so
"a one-round admission probe is therefore unsafe" `[frontier §End-to-end
linear-lane verdict]`.

**The verdict, in the register's own words.** "We measured remote
speculation across the wire and rejected it for general serving — the
verifier cost eats the gain. The shipped disaggregated lane is fast
remote prefill plus plain parity decode" `[claims #14]`. The economics
are structural, not tuning: a remote verifier is weight-stream-bound —
it must read a model's weights per round trip — while the local kquant
verifier was engineered until its 16-row batch forward cost 128.4 ms
against the draft's 26.9 ms `[ledger §Stage B L1]`.

## 33.6 The falsification ledger: a device worth porting

The frontier doc's most durable artifact may be its *form*: every
hypothesis carries its evidence class and its verdict, so the next
experimenter inherits a map of the dead ends. A selection, verbatim in
structure `[docs/nvfp4-distributed-speculative-frontier-20260818.md
§Architecture attempts and disposition]`:

| # | Hypothesis | Receipt | Verdict |
|---|---|---|---|
| 1 | Optimize the Mac Dudeman verifier | Measured: best 227.864 ms GPU / 239.564 ms wall — 13.9% over the 200 ms gate, 1.77× the 128.400 ms kquant reference; hard ceiling 70.2 verified rows/s `[ledger §"Fallback A follow-up"]` | Rejected as primary (Fallback A) |
| 2 | Second resident Dudeman verifier on GX10, linear chain | Authenticated composite import, end-to-end traces (§33.5) | **Rejected for general serving** |
| 3 | Existing RedHat session as target, Mac DFlash proposes | Measured mismatch screen: 190,449 teacher-forced rows, 8.116% RedHat/Dudeman top-1 disagreement; docs 15.428% → projected 42.82 tok/s | Algorithmically valid, fails the speed gate on docs |
| 4 | Shared-Gumbel coupling | Literature + reference implementation | Retained experimental; changes sampler semantics |
| 6 | Stateless GX verifier, authenticated Mac log, prefix-cache soft state | V2 transcript/journal implemented; service unimplemented | Preferred *service* architecture, if a lane ever returns |
| 10 | Uncertainty-aware token tree on GX | Measured batch curve; rank/coverage unmeasured | **Only remaining performance experiment** |
| 14 | Cross-checkpoint KV translator / CRDT branch union / wrong target certifying | Invariant rejection + published approximate systems | **Rejected as unsafe** |

*Figure 33.4: The falsification ledger (selected rows). The frontier's
own table runs to fourteen attempts, each with evidence class and
disposition `[frontier §Architecture attempts and disposition]`.*

This device is not new to this book — Figure 33.4's rows 1 and 2 are
the Fallback A no-go and the linear-lane rejection this chapter has
already met with their receipts. It is the same discipline the
decode-dispatch-gap note used when it reconciled the +196 closure gap
into named families and then *rejected the numerically inexact fusion*
(logprob error 3.197e-4 over the 1e-4 contract) rather than shipping it
`[docs/decode-dispatch-gap-20260815.md]`, and the same discipline the
offline GX trace analysis used when it refused to invent a fix from
insufficient evidence: the retained hashes "cannot identify the first
proposal token," so "inventing a rollback or serialization fix from
these data would be guesswork" `[docs/gx-speculative-trace-offline-
20260815.md §What the evidence does and does not prove]`. Reconcile
exactly, reject what breaks the contract, name what you do not know.
The distributed-speculative campaign simply ran the same culture at
cluster scale, and its rejection row in the claims register is the
output format: what was measured, what it cost, what ships instead
`[claims #14]`.

## 33.7 What survives: the V2 machinery, unwired

A rejected lane leaves usable parts. The research record retains a
**V2 typed protocol** whose pieces are individually tested even though
"nothing is wired into the serving route" `[frontier §What was
implemented]`:

- **The carried-frontier state machine.** A target-selected token that
  is not yet evaluated or emitted; a round evaluates
  `[frontier_in, draft_0, …, draft_(D-1)]` over D+2 target witnesses,
  and "this explicit geometry prevents a common off-by-one: publishing a
  replacement whose target KV row and five DFlash features do not yet
  exist" `[frontier §Carried-frontier state machine]`.
- **A replayable request.** The V2 request carries the complete
  evaluated-token transcript and the *actual* MT19937 state; the GX side
  reconstructs the f64 normalizer, replays one `uniform_f64` draw per
  row, checks every proposed token, and rejects a mismatched post-draft
  snapshot — consistency with the authenticated Mac's q bytes, with the
  honest caveat that it "does not prove that those q bytes came from the
  declared neural model" `[frontier §Coupling policy decision]`.
- **A durable transaction.**
  `PREPARED reservation → immutable staged render + fragment closure →
  durable result → fenced head CAS + render activation → emit → ACK`
  `[frontier §Distributed state and reconciliation]` — an fsynced commit
  WAL before idempotent activation, content-addressed fragments that
  absorb duplicates and reordering, Ed25519 target-only result
  signatures so a proposer holding the request key cannot forge verifier
  output `[frontier §Fragment closure and security limits]`.
- **A promotion-gate list** that any future revival must pass:
  composite-genesis exactness against a non-speculative oracle, a fused
  verifier service ("do not infer service latency from the Python
  screen"), real acceptance including the 24-task agentic set,
  end-to-end economics against a paired 107.9-lane control with a
  preregistered lower confidence bound, semantic requalification, and
  authority hardening `[frontier §Product promotion gates]`.

The architecture document's one-paragraph summary carries the boundary
that matters for readers of the shipped tree: the linear policy is
rejected, and "the production authority, renderer, executor, and stream
boundary remain unwired, so this research does not change the fail-closed
lane" `[docs/muser-architecture.md, distributed-verifier paragraph]`.
The only remaining performance experiment is the **hardware-aware token
tree** — spend the GX's otherwise-idle batch arithmetic on covering
near-miss branches rather than paying one weight stream per linear
round — and even it enters through a preregistered admission screen: at
24/32/48/64 nodes the measured capture curves require mean emitted
tokens per call of 15.77/16.18/17.01/17.84 (48 nodes is the first
sensible target; 64 only if its extra nodes add ≥0.83 token/call), and
"each organic stratum must exceed 107.9 tok/s with a preregistered lower
confidence bound; otherwise reject trees too" `[frontier §Frontier
attempt: uncertainty-aware token trees]`.

## 33.8 Tradeoffs

**Local speculation vs distributed verification, measured.** The local
lane's verify is a 16-row batch on shared weights (128.4 ms forward
against a 26.9 ms draft, `[ledger §Stage B L1]`); the distributed lane
pays a weight-stream per round plus transport, and its organic
acceptance (9.23–38.07%) put even its zero-cost ceilings at 20.15–55.96
tok/s `[frontier; claims #14]`. The crossover the tree experiment chases
is real but unproven; nothing in the shipped product depends on it.

**Draft cheapness vs draft quality.** Chapter 8 §8.8 measured it: the
draft is the *small* side of its own loop, so effort went to the verify
kernels and — after 2026-08-21 — to conditioning. The window bug's
arithmetic is the sharpest statement of the tradeoff in this book:
fixing the draft cost ~5% of synthetic speed and bought 1.1% → 72.7%
natural-text acceptance `[ledger §"ROOT CAUSE FOUND AND FIXED"]`.

**Fail-closed vs silently-slow.** Fallback B's refusal (§33.4) and the
unwired V2 (§33.7) are the same decision twice: a route that measured
6.805 tok/s (native spec) or 15.53 tok/s (distributed docs stratum) is
not served while awaiting a better design — it is refused, with the
evidence retained. The measured alternative was always available and
always rejected.

**The all-accept control as instrument, not result.** 110.59 tok/s
proved the pipeline's plumbing end-to-end; the same session's organic
strata killed the lane. Controls that cannot fail teach nothing;
controls that can fail — and this one could only pass by construction —
calibrate the instrument. Citing it as performance inverts its purpose
`[claims #14; measured-numbers §6 rule 6]`.

## 33.9 Where the gap lives

For the speculative lane, the dispatch-gap question gets a twist: the
gap is mostly *the point*. A speculative round deliberately runs more
kernels than plain decode (draft block, split target, wider verify) —
work that only pays if acceptance converts it into emitted tokens. The
measured acceptance collapse of §33.5 is what "the gap ate the win"
looks like when the extra work is remote: every rejected proposal paid
a full GX weight stream for zero emitted tokens. Locally, the same
arithmetic is why the engine distrusts its own draft (the windowed
disable gate of [Ch 8](08-the-dflash-draft.md) §8.7) and why the
natural-text losses on high-acceptance shallow text are real: when
llama's lighter draft verifies nearly everything, extra conditioning
work is pure overhead `[docs/benchmarks.md §2]`.

## 33.10 What comes next

Everything in this book so far has been one accelerator's story — one
Mac's kernels, one producer's prefill, one pair's wire — plus the
evidence culture that keeps their numbers honest. Part VII assembles
the machine that owns it all: one scheduler that owns one accelerator,
slots the requests, favors decode over prefill, and decides — per
token, under load — which of the lanes this Part built actually runs.
That is [Ch 34](34-scheduler-and-slots.md): the scheduler and the
slots.

---

## References

- `[crates/muser-engine/src/sampling.rs:1001-1097]` — the source-pinned
  MT19937 stream, `verify_full_speculative_mt` / `_mt_ordered`, the
  `min(p/q, 1)` acceptance rule and residual-corrected resample.
- `[crates/muser-engine/src/decode.rs:3293-3369, 3635-3666]` —
  `begin_dflash_verify_suffix` / `finish_dflash_verify_suffix` (the
  Mirror-SD split-graph overlap); `:213-226` the speculative checkpoint.
- `[crates/muser-bench/src/remote.rs:3-8, :33]` — the 256-token +
  full-logit-row comparison and `DFLASH_ACCEPTANCE_MINIMUM = 0.95`.
- `[crates/muser-cluster/src/config.rs:128-131]`,
  `[crates/muser-server/src/state.rs:1667-1678]` — Fallback B's two
  refusal layers.
- `[docs/nvfp4-distributed-speculative-frontier-20260818.md]` — the
  decision and target-identity invariant, the 107.152 ms screen and the
  99.151% preregistered bar, the end-to-end verdict table and receipt
  SHA pairs, the carried-frontier and coupling-policy sections, the
  architecture-attempts disposition table, the tree admission screen,
  and the product promotion gates.
- `[docs/nvfp4-fast-lane-evidence-20260817.md]` — the 6.805 tok/s
  native spec no-go and its qualification boundary; the 240/240
  correctness diagnostic.
- `[ledger §L2 Stage B verdict]`, `[ledger §Stage B L1]`,
  `[ledger §"ROOT CAUSE FOUND AND FIXED"]`,
  `[ledger §F-series remediation]`,
  `[ledger §"Fallback A follow-up — weight-only verifier final no-go"]`
  — `docs/goal-parity-ledger-2026-08.md`: the pre-fix 107.9136/1.3273
  bar, the L1 cycle decomposition, the window-bug sweep and
  consequences, the native no-go, the Mac verifier no-go.
- `[claims #3]`, `[claims #4]`, `[claims #14]`, `[claims #15]`,
  `[claims #16]` — `docs/launch-claims.md`: exact-token depths; native
  spec fail-closed; the distributed rejection and its wording
  discipline; the fixed-window synthetic ratios; the 131,008 wall
  parity and the barred 1.64960× figure.
- `[docs/benchmarks.md §2]` — natural-text wins/losses and the
  verify-length conventions.
- `[docs/decode-dispatch-gap-20260815.md]`,
  `[docs/gx-speculative-trace-offline-20260815.md]` — the mirror
  disciplines: exact reconciliation with inexact fusions rejected;
  refusal to invent fixes from insufficient evidence.
- `[docs/muser-architecture.md]` — the lane matrix; "DFlash drafts are
  always verified by target distributions"; the distributed-verifier
  paragraph (V2 unwired).
- `[measured-numbers §1b, §1g, §2 Arc 5, Arc 7, §6, §7]` — the spec
  scopes and landmines; the distributed-speculative table; the window
  arc; the claim crib; the mis-cited-numbers list.
- `[arxiv:2302.01318]` — Chen et al., the independent lossless
  speculative decoding formulation; the frontier doc's research basis
  additionally links Leviathan, Kalman, and Matias (PMLR 2023) for the
  same core result.
- [Ch 7](07-nvfp4-native-lane.md) §7.6, [Ch 8](08-the-dflash-draft.md)
  — Fallback B's full treatment; the draft's contracts and gates this
  chapter consumes.
- [Ch 34](34-scheduler-and-slots.md) — the scheduler that owns
  everything this Part built.
