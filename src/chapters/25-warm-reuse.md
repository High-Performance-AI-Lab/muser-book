# Chapter 25 — Warm reuse: the cache as an asset
> **status:** polished  ·  **path:** Muse Glimmer, pinned Muser tree

*Prerequisites: [Ch 24](24-kvpack-the-format.md) (the format, the identity
digest, the seal), [Ch 23](23-the-swa-ring-and-the-growing-cache.md) (the
interchange cut: 39 SWA tails + 13 NoPE fulls), [Ch 22](22-the-price-of-context.md)
(what a cache weighs). This chapter's numbers are the most tempting to
over-quote in the book; every one carries its scope, and §25.5 is about a
time the scope discipline failed in public.*

---

## 25.1 The question this chapter answers

[Ch 24](24-kvpack-the-format.md) built the vault: identities, seals,
content-addressed packs, a fail-closed receiver. A vault is only worth what
its contents return on withdrawal. This chapter is the withdrawal record —
what happens when a request arrives whose prefix the engine has already
computed, under what identity, with what controls, and what the hit is
worth against recomputing.

kvpack's overview doc frames the whole ladder in three rungs, each working
on more machines and costing less `[docs/kvpack.md §The reuse ladder]`:

| Rung | What happens | Result |
|---|---|---|
| Warm prefix, one Mac | Local resident/durable cache answers from installed state | ~65 ms to resume a shallow warm prefix |
| Warm prefix at depth, remote producer | Producer holds the exact prefix; no compute, no transfer | 0.613 s at 65,536 tokens (vs 68.6 s cold), 1.057 s at 130,815 (vs 147.8 s), bit-identical output |
| Delta handoff | Only the missing suffix crosses the wire | 54.2851 % of full payload bytes; output SHA-256 exactly equal |

*Table 25.1: the reuse ladder `[docs/kvpack.md]`. This chapter cashes the
first two rungs; [Ch 26](26-delta-handoff-and-migration.md) the third.*
And the doc's own warning frames the method: "the miss controls matter as
much as the hits" — a fast path is only genuine reuse if a *different*
prompt through the same path is not fast.

## 25.2 The ladder as code, and what counts as a hit

The runtime ladder is one ordered walk, `muser-kvpack/src/reuse.rs:1`:
"Ordered exact-prefix reuse: current session, resident, durable, then
remote." Four tiers, each stricter than the last:

1. **Current session** — the live session's own token history is a raw
   prefix of the request: no restore at all, just continue
   (`reuse.rs:330-335`).
2. **Resident** — an in-process, identity-scoped token radix holding
   content-interned Muse plane chunks (`resident.rs`); fast, but it has no
   cryptographic authentication of its own. Its internals are shaped
   entirely by [Ch 22](22-the-price-of-context.md)'s two regimes: cuts are
   kept at 256-token alignment "for ancestor reuse (non-aligned kept
   exact-hit-only)," plane chunks are SHA-256 content-interned in a
   `ChunkPool` with weak-reference dedup, and a global byte-budget LRU
   (default 8 GiB) bounds the tier `[docs/kvpack-merge-handoff §4]`.
3. **Durable** — the kvpack `LocalStore`: packs on disk, manifests,
   MACs — the authentication authority. Publication is cadence-driven, not
   request-driven: packs publish "at 2,048-token multiples + prompt/turn
   boundaries," each carrying the exact-final-logits plane at the synthetic
   layer 52 (`MUSE_EXACT_LOGITS_LAYER`,
   `[docs/kvpack-merge-handoff §4]`; the writer side is
   `session.rs:183-219`, the logits plane written last at `:189-200`).
4. **Remote** — an authenticated remote import, quarantined on failure,
   then re-resolved locally ("a transport response must not become an
   alternate authority", `[docs/kvpack-merge-handoff §4]`).

Publication itself is two calls on the ladder object, and their shapes say
the design: `publish_resident` exports the live session's interchange
snapshot plus its cached logits into the radix, and `publish_durable`
hashes the token history under a domain tag
(`b"muser-durable-prompt-v1\0"`) into an idempotency key so the same
session saved twice is one pack, not two
(`reuse.rs:136-162`).

The ordering embeds one non-obvious rule worth quoting in full, because it
is the anti-self-dealing core of the design:

> The durable tier is the sole authentication authority: when it is
> configured, its deepest authenticated cut caps what the unauthenticated
> resident tier may serve, so a resident entry can never serve a deeper cut
> than the durable chain has authenticated for the same identity.
> `crates/muser-kvpack/src/reuse.rs:322-328`

Nothing resident — not even an entry with witnessed final logits — stands
beyond the deepest durable cut; the tests pin exactly this
(`witnessed_exact_hit_does_not_stand_beyond_the_authenticated_cut`,
`reuse.rs:536-558`).

And before any number is quoted, the accounting definition — because
"cache hit" is the easiest metric in a serving system to inflate:
a hit is recorded only after authenticated state has been restored *and*
committed into an engine slot; a lookup that fails verification or
installation "contributes no saved value." Continuing the same live session
and installing a fresh GX10 prefill are deliberately *not* hits — they land
in `session_continuation_hits` and `disagg_prefills` and "never inflate the
cache hit rate" `[docs/kvpack-economics.md §What counts as a hit]`. A
remote prefill is real compute, not a cache read; same-session continuation
never evicted anything.

## 25.3 The shallow warm hit, at its original scope

The first measured rung: a 2,048-token prompt whose exact prefix is already
resident. The P4 cell of the campaign ledger is a five-sample measurement
with one unmeasured warmup request priming the serve path:

| P4 cell | Five-rep samples | Median | CV |
|---|---|---:|---:|
| Warm resident prefix hit | 65.263, 64.502, 64.631, 64.621, 64.921 ms | **64.631 ms** | 0.4239 % |

*Table 25.2: the shallow warm hit `[ledger P4 table]`.* CV here is the
coefficient of variation — standard deviation over mean, the campaign's
stability metric ([Ch 1](01-why-inference-is-a-memory-problem.md)'s
convention). Against the same
packet's official five-repetition cold disagg median of 46.787 s, the ledger
computes "a 723.90x cold-to-warm TTFT ratio" `[ledger P4]` — TTFT being
time to first token, the quantity [Ch 27](27-why-disaggregate.md) makes the
whole disaggregation argument turn on. The cell's own
evidence trail is the accounting of §25.2 made visible: "the cache cell
itself recorded exactly one Spark prefill, one declared unmeasured
cache-path warmup, and five measured resident hits: cache hits advanced
0 → 6, disaggregated prefills remained 1, and 654,311,424 bytes were served
from cache" `[ledger P4]` — the counters moved exactly as a real hit should
move them and nothing else.

Scope, carried from the claims register because it is load-bearing: this is
the **shallow** (2,048-token, resident, one Mac) figure, it "remains valid
at its original scope," and the register's proposed wording for it stays
conditional on the final identity `[claims #11]`. It is never the number
to quote at depth — that would be the next section's, and the two must
never be conflated (the campaign's landmine list exists because "~64 ms
warm hits at depth" is exactly the sentence someone will eventually write).

## 25.4 Warm reuse at depth: stage 5 of the kvpack ladder

The deep question — does reuse deliver at 65,536 and 130,815 tokens, with
controls, on live hardware? — was answered by the kvpack ladder's stage 5,
run as ordered fail-closed stages `[ledger Arc 3]`. The apparatus: for each
depth, an isolated leased server runs three legs — a **cold** leg (remote
prefill from the producer, full handoff), a **warm** leg (same prompt
again, after the cold leg installed the state), and a **miss control**
(an unrelated 8,192-token prompt through the same path). The verdict
`[ledger "Kvpack ladder stage-5 isolated-depth verdict"]`:

| depth | leg | generation | first-token s | total s | producer exit |
|---:|---|---:|---:|---:|---:|
| 65,536 | cold | 960209 | 68.6166 | 74.5217 | 0 |
| 65,536 | warm | n/a (no producer) | **0.6132** | 6.5360 | n/a |
| 65,536 | miss control (8,192) | 960210 | 10.5567 | 12.8994 | 0 |
| 130,815 | cold | 960211 | 147.8321 | 157.7974 | 0 |
| 130,815 | warm | n/a (no producer) | **1.0566** | 11.0625 | n/a |
| 130,815 | miss control (8,192) | 960212 | 10.5088 | 12.8513 | 0 |

*Table 25.3: stage-5 isolated-depth results `[ledger stage-5 verdict;
receipts kvpack-ladder-20260820/attempt-9-20260822T074100Z-stage5-warmhit/
stage5-warm-hit/{65536,130815}/warmhit-*.json]`. The receipts record
`legs_valid: true`, `outputs_match: true`, and `producer_driven: false`
on both warm legs — I re-read the JSONs; the numbers above are the
receipts' own.* Three findings, in the register's own scope language:

1. **Cold and warm text was bit-identical at both depths** `[claims #11]`.
   Not similar — the same bytes, which is only claimable because the
   restored planes install at the exact ring rotation
   ([Ch 23 §23.4](23-the-swa-ring-and-the-growing-cache.md)) under the
   exact identity ([Ch 24 §24.5](24-kvpack-the-format.md)).
2. **No producer drive on either warm hit** `[claims #11]`. The warm leg's
   "n/a (no producer)" row is the receipt's `producer_driven: false`: zero
   remote compute, zero wire. The Mac answers from its own installed state.
3. **The miss controls stayed slow** — 10.5–12.9 s for an unrelated
   8,192-token prompt through the same path, producer-driven as expected.
   "The fast path is genuine reuse, not a cache that answers everything"
   `[docs/kvpack.md]`.

Now the discipline half of the claim, verbatim in substance, because this
row is under operator review and the book does not outrun it: these are
**two depth-specific samples, not a distribution** — one warm leg per
depth, not five repetitions — and the register's rule is "never call
decode faster or claim these two latency samples as a distribution";
proposed wording ("At 65k and 130,815-token prompts, resident kvpack reuse
skipped the producer, preserved bit-identical output, and returned the
first token in about 0.6 s and 1.1 s") is **OPERATOR REVIEW REQUIRED**
`[claims #11]`. The release lock is authoritative; this chapter reports the
measurement and its scope, not product copy.

**What the warm leg mechanically does** — because "no producer drive" is
an absence, and the reader deserves the corresponding presence. The state
restores "into an owned detached shadow, two-phase KV+logits swap with
checkpoint rollback" `[docs/kvpack-merge-handoff §4]`: the tier's bytes
land in a `MetalMuseModel` generation nobody is decoding from, the shape
gate of [Ch 23 §23.4](23-the-swa-ring-and-the-growing-cache.md) validates
the cut, the install reproduces ring rotation, and only then does an
infallible swap publish KV and logits into the serving session — the same
shadow-then-commit contract as the handoff sink
([Ch 24](24-kvpack-the-format.md)) and the context-shift staging
([Ch 23 §23.7](23-the-swa-ring-and-the-growing-cache.md)). A failed
restore aborts and resets engine state rather than leaving a partial cache
live `[third_party/kvpack/README.md §How it works]`. The 0.6132 s /
1.0566 s first tokens are what restoring-and-verifying roughly 0.87 GB /
1.74 GB of sealed state plus one locally decoded boundary token costs on
this Mac — NoPE arithmetic: 65,535 × 13,312 ≈ 0.87 GB and
130,814 × 13,312 ≈ 1.74 GB `[Ch 22 §22.7]` — with zero wire and zero
producer compute. No phase decomposition of the warm leg is retained in
the receipt beyond the wall clocks **[unverified]**.

One more scope note the receipts make concrete: the deep warm hit is a
*second-class* sibling of the shallow one — 0.6132 s and 1.0566 s, not
64.631 ms. Restoring and re-verifying roughly a gigabyte of sealed state
([Ch 22](22-the-price-of-context.md): a 130,815-token NoPE span is
~1.74 GB) costs real time even with zero wire; the warm totals (6.54 s,
11.06 s) include the subsequent streamed decode. Different rung, different
price, different number.

## 25.5 The retracted "failure" — and the three apparatus failures before it

The stage-5 verdict above was the *fifth* attempt, and the road there is
the best evidence-culture exhibit in this Part. The first isolated run at
65,536 reported `outputs_match: false` — the cache appearing to return
different text. It was retracted as an **apparatus failure**, and the
ledger's postmortem is worth reading in its own terms: the probe's
producer-timeout default was 240 s, a 65,536-token prefill exceeds it, so
the producer was killed mid-handoff; the "cold" leg returned an empty
string ("the Mac returned an empty completion in 3.7 s because no KV ever
arrived") and the warm leg timed out entirely — so "`outputs_match: false`
was comparing an empty string against a leg that had errored … No
cache-correctness conclusion can be drawn from this cell in either
direction" `[ledger e426ec0 postmortem]`. Two fixes landed, not just a
retraction: a depth-scaled producer timeout, and `legs_valid`/`leg_errors`
gating so "a timeout can no longer be published as a correctness failure"
`[ledger]`. The measured-numbers ledger carries the summary landmine: the
retracted cell was an infrastructure timeout, not a cache-correctness
failure; the valid cell is bit-identical `[measured-numbers §7]`.

Then three more apparatus defects failed loudly before the pass, each
caught by the ladder's stop rule: attempt 6's generation formula wrapped
every 16 min 40 s and the receiver correctly refused "replayed or stale
generation"; attempt 7 reused one long-lived server for both depths, so the
second "cold" leg inherited the first depth's radix state — an invalid
cold-control apparatus, replaced by fresh independently leased servers per
depth; attempt 8 hit an O_EXCL collision on a node-side operational file
after a *successful* handoff, refused correctly, namespaced by attempt
identity and generation `[ledger stage-5 verdict preamble]`. The pattern
to internalize: the fail-closed machinery refused every bad attempt, the
verdict waited for a valid one, and the retracted headline never became a
"cache returns wrong text" legend. Evidence wins over wording
`[measured-numbers §6 rule 10]`.

## 25.6 Admission identity discipline — the exact-hit contract

What makes a warm hit *admissible* is the same discipline that makes it
bit-identical. As code, three rules from the ladder:

- **Exact held identity, structurally unreachable otherwise.** Resident
  entries are keyed under the `MuseIdentity` digest
  ([Ch 24 §24.5](24-kvpack-the-format.md)); "a wrong identity is
  structurally unreachable from a lookup, never a best-effort hit"
  (`layout.rs:33-36`), and swapping the durable tier's identity re-scopes
  every lookup so stale entries miss (`reuse.rs:560-590`).
- **Exact hits need witnessed final logits.** A cut that would *end* a
  generation is only exact if the final target distribution was captured
  with the KV: "Exact-final state is useful only when its target
  distribution was captured with the KV cut. Older/generic entries remain
  eligible as aligned ancestors but cannot masquerade as exact"
  (`reuse.rs:352-360`). A full-depth restore without logits is an error,
  not a silent downgrade (`reuse.rs:203-217`).
- **Durable caps resident** (§25.2) — depth served is authenticated depth.

The identity chain binds everything [Ch 24](24-kvpack-the-format.md)
enumerated — model, tokenizer, template, context policy, adapter, layout,
scalar math — and the receipt-side proof that the binding has teeth is in
the refusal records: the well-formed config with one flipped adapter digest
was rejected with an explicit identity-mismatch error
`[docs/kvpack.md §Proven live]`. Admission is fail-closed on identity the
way `append` is fail-closed on continuity
([Ch 23 §23.2](23-the-swa-ring-and-the-growing-cache.md)): the engine would
rather recompute than trust state it cannot prove.

## 25.7 The economics of a hit, honestly

What is a hit worth? The dashboard's economics module answers with
formulas and refuses to answer without inputs
(`crates/muser-kvpack/src/economics.rs`, specified by
`[docs/kvpack-economics.md]`):

```text
restore_speedup = local_prefill_seconds / restore_seconds
seconds_saved   = max(0, local_prefill_seconds - restore_seconds)
```

"`restore_speedup` becomes measured only after a caller records positive,
paired wall-clock durations for a restore and the identical local-prefill
cut. Before that it is zero and … `mock`" `[docs/kvpack-economics.md §Timing]`.
`gflops_avoided` reports only the conservative linear weight-matmul floor
(`2 * 30e9 * restored_tokens`) and omits attention FLOPs — "the field
undercounts, not overclaims" (`economics.rs:53-57`); `joules_saved` has no
calibrated power source and is permanently tagged `mock`. The honesty
rules extend to bytes: durable and remote restores must bill the
authenticated manifest's byte count, never a `tokens × per_token_bytes`
estimate — because, as the doc notes, "Muse's 39 SWA layers don't grow
linearly past 2048 tokens, so a naive per-token estimate overstates value
on long contexts" `[docs/kvpack-economics.md §Byte accounting]`. (The
resident tier, an in-process copy with no manifest, may use the estimate —
labeled as such.)

Against that discipline, two measured anchors bound the answer. Against
*local Mac re-prefill*, deep reuse wins by orders of magnitude: warm
first-token 1.0566 s at 130,815 tokens versus a local 131,008-class
prefill mean of 570.122 s `[ledger "EEE A/B at 130815"]` — and the ladder
doc's summary, "at deep prompts, reuse in any form beats recompute by
orders of magnitude" `[docs/kvpack.md §Economics]`. Against an *overnight
GX10 producer at 5–20k tok/s*, the research frontier analysis puts the
honest fleet crossover at ~20–40× reuse counts, with per-request TTFT on
composed contexts "dominated by the SWA warm-up (~5–9 s at W = 2048 …),
not by the restore (~0.6 s for a full 131k NoPE image)" — labeled analysis,
its evidence tags [EXACT arithmetic]/[HYP], not a Muser measurement
`[docs/kv-reuse-frontier §1]`. The economic subject of a cache is time and
joules, not bytes: the cache is ~3,000× larger than the text it memoizes,
so "the appliance sells time (and joules), not bytes"
`[docs/kv-reuse-frontier §1]`.

## 25.8 Tradeoffs

**Exact-prefix reuse vs semantic or fuzzy reuse.** The ladder serves exact
token prefixes under exact identities, full stop. The alternative —
similarity-matched or spliced caches — is a research lane the frontier doc
maps in full, and its headline accounting is why the product stays exact:
"exact composition of B onto A costs a full prefill of B in A's context …
Every non-prefix composition of representations is approximate by
construction" `[docs/kv-reuse-frontier §2]`. Muser keeps the approximate
lane out of the admission path (composed caches are "RECONCILED, never
EXACT," a provenance distinction the research inherits from the same
culture `[docs/kv-reuse-frontier §4]`). The measured consequence of
exactness is Table 25.3's `outputs_match: true` at two depths; the cost is
that near-misses serve nothing.

**Two rungs, two numbers, never blended.** 64.631 ms (shallow, five-rep
median, CV 0.4239 %) and 0.6132 s / 1.0566 s (deep, one sample each) are
different scopes on different apparatus — the campaign's landmine list
exists because conflating them manufactures a claim nobody measured
`[measured-numbers §7]`. The honest comparison for the deep rung is its
own cold leg (111.9× at 65,536; 139.9× at 130,815 by first-token
arithmetic on Table 25.3) — single-sample ratios, labeled as such.

**Fast resident tier vs authenticated durable tier.** The resident radix
is the only tier fast enough for the ~65 ms rung, but it authenticates
nothing; the durable tier is authoritative but reads disk and verifies
MACs. The design neither trusts the fast tier nor slows every lookup to
the strict one — it caps the fast tier by the strict one's authenticated
depth (§25.2). The unmeasured cost is a durable catalog read on every plan
(a `find_deepest` walk, `session.rs:222-244`) even when the resident hit
would serve; no retained measurement isolates that lookup's wall cost
**[unverified]**.

**Where the gap lives.** Warm reuse is the *anti*-gap: the whole point is
that no Metal dispatch runs for the restored prefix. The costs it does
carry are the receiver/restore phases (verify, install, commit — the
~0.2 s-class constants of `[ledger N2]`) and the pack reads of §25.7,
which live in the economics panel, not the dispatch-gap table.

## 25.9 What comes next

A warm hit is the best case: the cache holds *exactly* what you need, and
nothing moves. One rung down is the case that dominates real traffic — the
cache holds a **prefix** of what you need: the system prompt and the first
32k of a document, say, with a fresh suffix appended. Exactness does not
have to be sacrificed to save the wire: the handoff can be *armed* as a
delta, admission can verify the held prefix to the token, and only the
missing suffix crosses — with a measured 54.2851 % cell to show for it.
How the cut is aligned, what gets re-sent and why, and how whole sessions
move between decode nodes without ever losing either end — that is
[Ch 26](26-delta-handoff-and-migration.md), the last chapter of this Part.

## References

- `[docs/kvpack.md]` — the reuse ladder (Table 25.1), miss-control
  framing, refusal receipts, economics summary.
- `[ledger P4]` — the shallow warm-hit five-sample cell, the 723.90×
  ratio, and the counter trail (0 → 6 hits, 654,311,424 B served).
- `[ledger "Kvpack ladder stage-5 isolated-depth verdict"]` — Table 25.3's
  source, the attempt-6/7/8 apparatus failures, the PASS verdict.
- `[ledger e426ec0 postmortem]` — the retracted 65,536 `outputs_match:
  false` cell: 240 s producer timeout, empty-vs-errored comparison, the
  two fixes.
- `[receipts kvpack-ladder-20260820/attempt-9-20260822T074100Z-stage5-warmhit/
  stage5-warm-hit/{65536,130815}/warmhit-{65536,130815}.json]` — per-leg
  TTFT/totals, `legs_valid`, `outputs_match`, `producer_driven` (re-read
  for this chapter).
- `[claims #11]` — the scope language carried in §25.3–25.4 (original
  scopes, no-producer-drive wording, OPERATOR REVIEW REQUIRED).
- `[measured-numbers §1d, §6, §7]` — the warm/delta rows, claim-discipline
  rules, and the "different scopes" landmine.
- `crates/muser-kvpack/src/reuse.rs:1-7, 136-162, 203-217, 322-368,
  536-590` — ladder order, resident/durable publication, exact-hit logits
  rule, durable-caps-resident, identity re-scoping tests.
- `crates/muser-kvpack/src/session.rs:144-244` — durable save and
  `find_deepest`.
- `crates/muser-kvpack/src/economics.rs` + `[docs/kvpack-economics.md]` —
  hit definition, byte accounting, mock-tagged derived values.
- `[docs/kv-reuse-frontier-20260820 §1-2, §4]` — the crossover and
  SWA-warm-up analysis (labeled research), the exact-vs-reconciled
  provenance regime.
- `[ledger "EEE A/B at 130815"]` — the 570.122 s local deep-prefill mean
  (§25.7's anchor).
- [Ch 22](22-the-price-of-context.md), [Ch 23](23-the-swa-ring-and-the-growing-cache.md),
  [Ch 24](24-kvpack-the-format.md) — weight, interchange, and identity
  machinery the hits stand on.
- [Ch 26](26-delta-handoff-and-migration.md) — the delta rung this chapter
  exits to.
