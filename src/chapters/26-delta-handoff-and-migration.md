# Chapter 26 — Delta handoff and session migration
> **status:** polished  ·  **path:** Muse Glimmer, pinned Muser tree

*Prerequisites: [Ch 25](25-warm-reuse.md) (the reuse ladder, the exact-hit
contract), [Ch 24](24-kvpack-the-format.md) (the seal, the receiver, the
`prefix_cut` lifted at the frame boundary), [Ch 22](22-the-price-of-context.md)
(the per-class byte arithmetic this chapter re-derives on the wire),
[Ch 23](23-the-swa-ring-and-the-growing-cache.md) (why SWA rows are
position-bound and NoPE rows are not).*

---

## 26.1 The case between "everything" and "nothing"

[Ch 25](25-warm-reuse.md) cashed the two clean rungs: a full hit, where the
cache holds exactly the prompt and nothing moves at all. Real traffic is
rarely that clean. The common shape is *half-cached*: the system prompt and
the first chunk of a document are already resident — from an earlier
request, an earlier session, an earlier handoff — and only the suffix is
new. The two obvious responses are both wasteful: recompute everything
(throw away proven state) or transfer everything (ship bytes you already
hold, bit-identically, and pay the wire again).

The delta rung is the third response: **leave the held prefix installed,
admit the difference against it, and move only what is new.** And once a
cache can be moved *partially* on the wire, it can also be moved *wholly*
between machines — a session migrating from one decode node to another, or
into enrolled storage — which is this chapter's second half. Both halves
obey one rule stated once and enforced everywhere: the receiver must hold
*exactly* what it claims to hold, provably, before anything is skipped.

## 26.2 The shallow cell: half-cached, half the bytes

The first measured delta cell was shallow and almost arithmetic-pure: hold
a 1,024-token prefix, request 2,048. The wire moved **49.98 %** of the
full-handoff bytes and the decode was bit-exact `[ledger T-series
"Delta-only prefill (W3)"; receipt nvfp4-pacing8g-20260818/delta-wrapper7/]`,
carried in the claims register as "the original half-cached 2,048-token
cell moved 49.98% of full bytes and decoded bit-exactly" `[claims #12]`.

Why 49.98 and not the naive 50.0 %? Derive it with [Ch 22](22-the-price-of-context.md)'s
formula, remembering one convention from [Ch 23](23-the-swa-ring-and-the-growing-cache.md):
the receiver holds back the boundary token and decodes it locally, so KV
ships for one token less than the prompt. At this geometry the suffix
(1,024 tokens) fits inside the 2,048 window, so the SWA span for the delta
is exactly the suffix's own rows — nothing is double-paid (§26.4 is where
that changes) — and the payload is the suffix's NoPE rows plus the suffix's
SWA rows, one token short: 49.98 %, the suffix share at that geometry
`[measured-numbers §1d]`. The cell's value is not the number, which
arithmetic predicts; it is the *bit-exact decode* — proof that skipping
1,024 tokens of prefill on the wire changes nothing downstream.

## 26.3 The deep cell: 32,768 of 65,536, measured to the byte

Stage 6 of the kvpack ladder ran the deep witness: hold a 32,768-token
prefix, request 65,536, and compare a delta handoff against a full-handoff
reference on the same prompt `[ledger "Kvpack ladder stage-6 delta-witness
verdict"]`. Three arms, one node, one night:

| arm | generation | prompt tokens | prefix cut | payload bytes | producer total s |
|---|---:|---:|---:|---:|---:|
| prefix identity witness | 960213 | 32,769 | 0 | 517,996,544 | 30.7269 |
| delta handoff | 960214 | 65,536 | 32,768 | **517,983,232** | 63.4784 |
| full reference | 960215 | 65,536 | 0 | 954,190,848 | 64.5361 |

*Table 26.1: the stage-6 arms `[receipt kvpack-ladder-20260820/
attempt-10-20260822T074826Z-stage6-delta/stage6-delta-65536/stage6-verdict.json]`
— `delta_share_of_full: 0.5428507652…` = **54.2851 %**, output SHA-256
exactly equal to the full-handoff reference (`2526a55d…19778`,
`exact_against_full_handoff: true`), `seal_eligible: false`.*

Every payload in that table reconciles to the byte against the per-class
arithmetic — this is [Ch 22 §22.7](22-the-price-of-context.md)'s method
applied three times, boundary token held back throughout:

```text
full reference:
    NoPE [0, 65,535)   = 65,535 × 13 × 1,024                =  872,401,920 B
    SWA window         =  2,048 × 39 × 1,024                =   81,788,928 B
                                                          total 954,190,848 B  ✓

prefix witness (32,769-token prompt, boundary held back → 32,768 rows):
    NoPE [0, 32,768)   = 32,768 × 13 × 1,024               =  436,207,616 B
    SWA window                                                 =   81,788,928 B
                                                          total 517,996,544 B  ✓

delta:
    NoPE [32,768, 65,535) = 32,767 × 13 × 1,024            =  436,194,304 B
    SWA window (re-sent)                                      =   81,788,928 B
                                                          total 517,983,232 B  ✓
```

Now the detail that makes the deep cell interesting rather than trivial:
the delta (517,983,232 B) is *larger* than `full − prefix` (436,194,304 B)
— by exactly one SWA window, 81,788,928 B. The held prefix's rings contain
the window at the *prefix's* tail, positions [30,720, 32,768); the finished
context needs the window at [63,488, 65,536) — different tokens, and on
RoPE layers different bytes ([Ch 23 §23.5](23-the-swa-ring-and-the-growing-cache.md):
an SWA key row is rotated by its absolute position at store time). The
schedule states the rule plainly: "delta span re-sends the whole window
when the suffix exceeds it" `[docs/kvpack-merge-handoff §6]`. So at this
geometry the delta ships a *full new window* plus the NoPE suffix, which
happens to land within 13,312 B (one NoPE token) of the prefix arm's size —
a coincidence of the 50 %-cut geometry, not a law.

And the register's caveat, carried verbatim in substance because it is the
difference between an engineering fact and a product claim: **"Do not
claim producer-side compute savings (suffix-only wire, not proven
suffix-only compute)"** `[claims #12]`. The wire provably carried
54.2851 % of the bytes; nobody has proven what the producer recomputed
behind the seal. The cell is also explicitly `seal_eligible: false` —
unsealed engineering evidence, like everything in this book
`[measured-numbers §6 rule 8]`.

## 26.4 Arming a delta: the ladder decides, admission enforces

The runtime path from "a prompt arrives on the remote lane" to "delta" has
two halves: the reuse ladder classifies the hit, then the handoff admission
verifies the cut. The ladder's classifier is small enough to read whole —
note the boundary-token convention and the alignment rule:

```rust
// crates/muser-kvpack/src/reuse.rs:25
/// What the reuse ladder can do for a prompt a remote producer would
/// otherwise prefill.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteReuseAction {
    /// The ladder holds the prompt minus at most the boundary token the
    /// receiver decodes locally: skip the remote transfer entirely.
    ServeLocal,
    /// The ladder holds a cut-aligned strict prefix: leaving it installed in
    /// the session arms the handoff as a delta (the producer's `prefix_cut`
    /// is validated against the session's held tokens at admission, and a
    /// full producer answer atomically replaces them, so arming never
    /// grafts unverified state).
    ArmDelta,
    /// Nothing the remote handoff could build on: run a full transfer.
    FullTransfer,
}

// crates/muser-kvpack/src/reuse.rs:55
fn remote_action(matched: usize, prompt: usize, cut_align: usize) -> RemoteReuseAction {
    if prompt < 2 {
        return RemoteReuseAction::FullTransfer;
    }
    if matched >= prompt - 1 {
        RemoteReuseAction::ServeLocal
    } else if matched > 0 && cut_align > 0 && matched.is_multiple_of(cut_align) {
        RemoteReuseAction::ArmDelta
    } else {
        RemoteReuseAction::FullTransfer
    }
}
```

A hit reaching `prompt − 1` is already full (the held-back token covers the
rest); a shorter prefix arms a delta *only* on the cut alignment; anything
else — including an unaligned partial, which the radix deliberately keeps
for exact-hit-only lookup — runs a full transfer, "left uninstalled, so
the full-transfer path can reset exactly as before" (`reuse.rs:226-237`).
The server adds one economic filter before arming: a live-session
continuation prefills its suffix locally for less than a handoff costs, so
only fetched tiers (resident, durable, remote) arm
(`arm_remote_delta`, `openai.rs:2907-2918`, with `matched + 1 < prompt`
guarding a nonempty transferable suffix).

The alignment constant is the format's, not the ladder's:
`PREFIX_CUT_ALIGN: u64 = 256` (`crates/muser-cluster/src/schedule.rs:26`) —
"Delta handoffs may begin only on a radix-friendly 256-token boundary,"
matching kvpack's 256-token prefix-key blocks ([Ch 24 §24.3](24-kvpack-the-format.md)).
Admission then verifies the cut against everything it must, fail-closed:

```rust
// crates/muser-cluster/src/identity.rs:140
/// transfer over `[cut, position)`. Fail closed unless the cut is
/// 256-aligned, leaves a nonempty suffix, names a prefix the receiving
/// session holds exactly, and — for declared schedules — the declared
/// target segments equal the span schedule for the cut.
fn validate_prefix_cut(&self, manifest: &BeginManifestV2) -> kvpack_handoff::Result<()> {
    // … ("delta prefix cut is not a 256-aligned cut inside the prompt";
    //     "delta prefix cut names a prefix the receiving session does not
    //     hold" when held_token_ids[..cut] != prompt_token_ids[..cut]) …
}
```

"Admission remains fail-closed on exact held identity, aligned nonempty
suffix, and span schedule" `[claims #12]` — the held tokens are compared
element-by-element against the manifest's prompt; a one-token disagreement
refuses. The span schedule itself is derived, not negotiated:
`muse_schedule_span_for` (`schedule.rs:91-130`) computes the NoPE tiles
over `[prefix_cut, position)` in 512-token steps and places the SWA span at
`position − 2,048` clamped to the cut — the exact decomposition §26.3
reconciled to the byte. One more honest wart from the format audit: the
typed `BeginManifestV2` still drops `prefix_cut`, so it travels as raw JSON
lifted at the frame boundary (`transport.rs:35-46`) and "delta handoff
(`ArmDelta`, 256-aligned cuts) has a Python-only producer"
`[docs/kvpack-merge-handoff §3 F1]` — cross-verification paying its
maintenance tax.

## 26.5 Session migration: moving the whole asset, two-phase

Delta handoff moves the *new* part of a cache. Migration moves the *whole*
session — target KV, DFlash state, sampler/RNG, replay messages, vision
rows, revision — between machines, and it is specified in the architecture
doc in five sentences that are effectively the protocol:

> Migration is two phase. Decode-node copy/move uses authenticated HTTPS
> between identically qualified Muser decoders; storage-tier copy/move uses
> enrolled kvpack storage. The destination durably commits before a move
> can delete the source, and transfer status is idempotently queryable
> after ambiguous failures. GX10 is not a decode destination.
> `[docs/muser-architecture.md §Context and sessions]`

The public surface is one route — `POST /v1/sessions/{id}/migrate`
(`axum_httpd.rs:527`, handler at `:3742`) taking `mode` (`copy`|`move`),
`tier` (`decode`|`storage`), `destination`, and an optional `transfer_id`
whose reuse is bound to the same session/destination/mode — replaying a
different migration under a known id is a 409 (`:3812-3830`).

**Decode tier.** The destination must be "an absolute HTTPS origin without
path, query, or fragment" (`validate_decode_destination`,
`axum_httpd.rs:4265-4277`) — a Muser decoder, authenticated by the source
server's API key, optionally pinned to a private CA
(`MUSER_DECODE_MIGRATION_CA`, `:4345-4359`). `run_decode_transfer`
(`:4394-4504`) is the two-phase spine:

```rust
// crates/muser-server/src/axum_httpd.rs:4428  (abridged to the spine)
let prepare = InternalTransferPrepare {
    transfer_id: transfer_id.into(),
    // bytes, sha256, transport_key,
    model_sha256: export.model_sha256,
    tokenizer_sha256: encode_hex32(&export.tokenizer_sha256),
    template_sha256: encode_hex32(&export.template_sha256),
    layout_abi: export.layout_abi,
    dflash_identity_sha256: export.dflash_identity_sha256,
    // (vision projector/preprocessing digests)
};
// 1. prepare at the destination  →  2. PUT the payload  →  3. commit
//    … then:
if !committed {
    return Err("destination did not durably commit the transfer".into());
}
server.logical_sessions.update_transfer(transfer_id, "destination_committed", None, false)?;
```

"Identically qualified" is not a hope — the prepare record carries the
model, tokenizer, template, layout-ABI, DFlash, and vision digests
(`InternalTransferPrepare`, `:3979-3997`; `session_identity` at `:4318-4343`),
the same identity family [Ch 24](24-kvpack-the-format.md) sealed into every
pack. A destination that cannot match them refuses. And the ambiguity rule
is explicit in code: if the upload or commit errors, the source *reconciles
by querying the destination's transfer status* and accepts a
`"committed"` verdict from there (`:4480-4491`, via
`GET {destination}/v1/session-transfers/{id}`, `:4380-4392`); failures
record status `"ambiguous"` unless the record already shows
`destination_committed` or `source_restored` (`record_transfer_failure`,
`:3942-3958`). Transfer status is queryable any number of times, on either
end, idempotently (`session_transfer_get`, `:3960-3977`) — after an
ambiguous failure, the answer to "did it land?" is a lookup, not a guess.

**Storage tier.** The destination is an *enrolled* node, and enrollment is
verified before anything moves: the registry entry must be healthy under
enrollment v2 with a live HMAC epoch (`enrolled_storage_node`,
`:4540-4559`). The remote side runs pinned shell fragments — prepare
(mkdir, chmod 700, sync), commit (verify byte count and SHA-256, then
`mv` temporary → final, `sync` file and directory), delete — quoted in
source at `:4506-4538`. The commit script is idempotent by construction:
if the final file already exists it re-verifies size and digest and exits
clean, so a retried transfer after an ambiguous failure cannot corrupt
either end. `run_storage_transfer` (`:4568-4640`) marks
`destination_committed` only after the remote commit script succeeds, and
the local payload is removed only when the record says the source side is
deleted (`:4630-4638`). A storage *restore* that is a `move` deletes the
remote bundle only after the local adoption succeeded — and if that delete
fails, the record degrades to `source_restored_remote_retained` rather
than pretending completion (`:4657-4685`).

**The move invariant.** Read across both tiers: no failure ordering deletes
the source before the destination's durable commit is on record. A `move`
is a `copy` plus a deletion that only ever runs after `"destination_
committed"` — which is why the status vocabulary (`starting`,
`transferring`, `destination_committed`, `source_restored`,
`source_restored_remote_retained`, `completed`, `ambiguous`) has a shape:
every prefix of a crash is recoverable, and every recovery is a status
query away. And the GX10 line is a topology fact, not a slight: the GX10
is a prefill producer with no Muser decode runtime; there is nothing on it
to receive a decode session `[docs/muser-architecture.md]`.

## 26.6 Tradeoffs

**Delta vs full handoff.** Measured: 54.2851 % of full bytes at the deep
cell with exactly-equal output SHA, 49.98 % at the shallow cell
`[claims #12; ledger stage 6]`. Unmeasured and unbought: resumability —
the format has "no offset/resume/retry vocabulary … the only partial-work
mechanism is `prefix_cut` at BEGIN," so a dropped 1.82 GB transfer restarts
from zero `[docs/kvpack-merge-handoff §3 F3]`. The register's caveat stands
guard on the interpretation: wire savings are proven, producer-compute
savings are not `[claims #12]`.

**The SWA window tax.** The delta pays 81,788,928 B to re-send a window it
"already has" a version of, because the version it has is position-bound
([Ch 23](23-the-swa-ring-and-the-growing-cache.md)). The alternative —
re-anchoring cached rotated keys by one rotation — is exact mathematics
and a research lane this program has deliberately not shipped
(`[docs/kv-reuse-frontier §2]`: the rotation group action is solved;
contextualization is not). At this geometry the tax is 15.8 % of the delta;
at deeper held prefixes it amortizes toward the NoPE-share floor of
[Ch 22 §22.7](22-the-price-of-context.md).

**Arming from the live session vs fetched tiers only.** `arm_remote_delta`
refuses current-session prefixes (`openai.rs:2907-2918`) — locally
prefilling a suffix you are already positioned on costs less than any
handoff. The cost of the rule is that the one case it blocks (a live
session exactly on an aligned cut with a huge suffix) pays a local prefill;
the benefit is that arming always corresponds to state a tier vouched for.

**Copy-then-delete vs in-place move.** The two-phase design spends an extra
copy's storage and one round of status traffic to buy the invariant that
no crash window loses the session. The alternative — in-place move with
compensation — would need exactly the distributed transaction machinery
the replay ledger/seal architecture already refuses to improvise
([Ch 24 §24.6](24-kvpack-the-format.md)). The retained receipts for this
lane are the wizard and ladder sessions; a dedicated migration failure-mode
matrix is not among them **[unverified]** — the design's guarantees are
code-and-doc anchored here, with the same status-reconciliation paths
exercised only incidentally.

**Where the gap lives.** Nothing here touches the decode graph; the costs
are wire and storage, and they are booked where [Ch 31](31-the-wire-discipline.md)
books wire costs. The one decode-graph interaction is the good kind: a
delta's installed prefix means fewer prefill chunks through the Metal
graph, which is the entire point.

## 26.7 What comes next — and the end of Part V

Part V has followed one asset through its whole life: what the cache costs
per token and per layer class ([Ch 22](22-the-price-of-context.md)), how
the ring and the growing plane implement it
([Ch 23](23-the-swa-ring-and-the-growing-cache.md)), how kvpack seals it
into a portable, provenance-pinned format
([Ch 24](24-kvpack-the-format.md)), what a warm hit is worth with controls
([Ch 25](25-warm-reuse.md)), and now how to move only the new part — or
the whole session — without ever trusting an unverified byte. The
discipline underneath every chapter has been the same: **move KV, don't
recompute it, and prove what moved.**

The next question is forced by the numbers already on the table. The cold
deep legs of [Ch 25](25-warm-reuse.md) measured 68.6 s and 147.8 s of
first-token latency on this lane — and the local alternative at 131k-class
depth was 570 s `[ledger "EEE A/B at 130815"]`. Someone computed that KV
fast, over a wire, under a seal, and it was not the Mac. If KV is an asset
that can move, then the machine that *computes* prefill and the machine
that *uses* it need not be the same machine — and the economics of splitting
them is exactly why Muser puts a Mac and a GB10 on the same fabric. That
argument — the TTFT cliff at depth, the roofline split, and what it costs
to trust someone else's prefill — opens Part VI:
[Ch 27](27-why-disaggregate.md).

## References

- `[claims #12]` — both delta cells, the admission rule, and the
  "suffix-only wire, not proven suffix-only compute" caveat (§26.2–26.4).
- `[ledger T-series "Delta-only prefill (W3)"]` +
  `[receipt nvfp4-pacing8g-20260818/delta-wrapper7/]` — the shallow cell.
- `[ledger "Kvpack ladder stage-6 delta-witness verdict"]` +
  `[receipt kvpack-ladder-20260820/attempt-10-20260822T074826Z-stage6-delta/
  stage6-delta-65536/stage6-verdict.json]` — Table 26.1's arms, the
  equal output SHA, `seal_eligible: false` (re-read for this chapter).
- `crates/muser-cluster/src/schedule.rs:20-26, 91-130` —
  `PREFIX_CUT_ALIGN`, `muse_schedule_span_for` (tiles, window clamp,
  layer-major streaming order).
- `crates/muser-cluster/src/identity.rs:143-160` — `validate_prefix_cut`
  (quoted abridged): alignment, nonempty suffix, exact held prefix, span
  schedule.
- `crates/muser-kvpack/src/reuse.rs:27-66, 226-237` —
  `RemoteReuseAction` and `remote_action` (quoted); unaligned partials
  left uninstalled.
- `crates/muser-server/src/openai.rs:2907-2918` — `arm_remote_delta`
  (fetched-tiers-only rule).
- `crates/muser-cluster/src/transport.rs:35-46` — the raw-JSON
  `prefix_cut` lift; `[docs/kvpack-merge-handoff §3 F1, F3]` — the
  Python-only producer and the missing resumability.
- `[docs/muser-architecture.md §Context and sessions]` — the five-sentence
  migration protocol (quoted in §26.5); §Durable and remote KV (the GX10's
  producer role).
- `crates/muser-server/src/axum_httpd.rs:527, 3742-3977` — the migrate
  route and handler; `:3942-3958` ambiguous-failure recording;
  `:3960-3977` idempotent status.
- `crates/muser-server/src/axum_httpd.rs:4265-4277, 4318-4343, 4394-4504` —
  HTTPS-origin validation, the identity set, `run_decode_transfer`
  (prepare/upload/commit spine quoted), destination reconciliation.
- `crates/muser-server/src/axum_httpd.rs:4506-4538, 4540-4659` — storage
  prepare/commit/delete scripts, enrollment-v2 gate,
  `run_storage_transfer`'s commit-before-delete; `:4642-4710`
  `run_storage_restore`'s move semantics and
  `source_restored_remote_retained`.
- `[ledger "EEE A/B at 130815"]` — the 570.122 s local deep-prefill mean
  (§26.7's cliff).
- [Ch 22](22-the-price-of-context.md) — the per-class arithmetic
  reconciled in §26.3; [Ch 23](23-the-swa-ring-and-the-growing-cache.md) —
  position-bound SWA rows behind the window tax; [Ch 24](24-kvpack-the-format.md)
  — the seal and receiver; [Ch 25](25-warm-reuse.md) — the ladder.
- [Ch 27](27-why-disaggregate.md) — Part VI's opening argument.
