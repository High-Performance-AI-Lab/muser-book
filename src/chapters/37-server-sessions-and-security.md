# Chapter 37 — The server surface: sessions, migration, and the security boundary
> **status:** polished  ·  **path:** Muse Glimmer, pinned Muser tree

*Prerequisites: [Ch 24](24-kvpack-the-format.md) (identity binding and the
sealed manifest), [Ch 26](26-delta-handoff-and-migration.md) (migration's
KV-side story), [Ch 34](34-scheduler-and-slots.md) (slots, admission, the
unhealthy latch), [Ch 21](21-sampling-argmax-and-grammar.md) (the sampler
state a session carries). No web-security background is assumed; CSRF,
cookies, and Origin are defined on first use.*

---

[Ch 36](36-prefill-vs-decode-paths.md) ended at the point where the two
graphs produce tokens — for whom? For a client, over a wire, under an
identity, with state that must survive a connection closing and a process
restarting. This chapter assembles `muser-server`: the frozen
llama-compatible HTTP surface, the Ollama aliases, the logical-session and
migration machinery, and — the part most worth studying as a *design* — the
deliberately asymmetric security model that makes a loopback laptop
frictionless and a LAN exposure impossible to configure by accident. Two
threads from earlier chapters converge here: the identity discipline of
[Ch 24](24-kvpack-the-format.md) (a bundle that binds everything, refusing
to restore across any mismatch) and the fail-closed latch of
[Ch 34](34-scheduler-and-slots.md) (uncertain state → 503, visible to the
client as ordinary HTTP).

---

## 37.1 The public serving surface

The active router is one Axum `Router` construction
[crates/muser-server/src/axum_httpd.rs:487-559], and the architecture
document describes it as "the frozen llama-compatible completion, chat,
tokenizer/template, embedding, slot, model/property, and health routes; the
Ollama-compatible `/api/generate` and `/generate` aliases; logical session
and migration routes; `/snapshot` JSON; `/metrics` Prometheus text;
authenticated `/stream` WebSocket telemetry; and temporary `/telemetry` SSE
keyframes" `[docs/muser-architecture.md §Public serving surface]`. Enumerated
with one-line purposes, grouped by role, paths exactly as routed
(Table 37.1):

| Routes | Purpose |
|---|---|
| `/`, `/dashboard` | the same-origin dashboard (live process state only) |
| `/snapshot` | JSON snapshot of live engine state (management auth) |
| `/metrics`, `/telemetry` | Prometheus text; temporary SSE keyframes |
| `/health`, `/v1/health`, `/healthz` | health — 503 when latched unhealthy (§37.8) |
| `/models`, `/v1/models`, `/props` | model identity/properties surface |
| `/slots`, `/slots/{id}` | resident-slot inspection and actions (erase) |
| `/tokenize`, `/detokenize`, `/apply-template` | tokenizer/template endpoints |
| `/embedding`, `/embeddings`, `/v1/embeddings` | embedding endpoints |
| `/completion`, `/completions`, `/v1/completions` | llama-compatible completions |
| `/api/generate`, `/generate` | Ollama-compatible aliases |
| `/v1/chat/completions` (+`/chat/completions`, `/control`) | the chat route of §37.3; mid-stream reasoning control |
| `/v1/stream` (GET/DELETE), `/v1/streams/lookup` | resumable SSE streams and lookup |
| `/v1/dashboard/login` | API-key → cookie exchange (§37.6) |
| `/v1/ws-tickets` | mint a 30 s single-use WebSocket ticket (§37.6) |
| `/v1/sessions` (+`/{id}`, `/save`, `/restore`, `/migrate`) | logical-session CRUD, durability, migration |
| `/__muser/v1/session-transfers/*` | migration two-phase control (§37.5) |
| `/stream` | authenticated WebSocket telemetry |
| `/v1/nodes` (+`/{name}/progress`) | GX10 node enrollment/progress (management auth) |
| `/__muser/benchmark/shutdown` | benchmark-lane control |

*Table 37.1: the serving surface
[crates/muser-server/src/axum_httpd.rs:488-543]. Compatibility claims are
bounded by the frozen contract: "security policy, identifiers, clocks,
paths, timings, build fingerprints, and documented Muser metrics may differ"
`[docs/muser-architecture.md §Product boundary]`.*

Two routers, not one, and the split is a bound: the main application layer
carries `DefaultBodyLimit::max(MAX_BODY)` (64 MiB), a 30 s request-body
timeout, and `ConcurrencyLimitLayer::new(256)`; a *separate* transfer-payload
router alone gets `DefaultBodyLimit::disable()`, a one-hour body timeout, and
`ConcurrencyLimitLayer::new(4)` [crates/muser-server/src/axum_httpd.rs:544-554].
Unbounded bodies exist only on the migration lane, four at a time.

## 37.2 Strict framing: unknown fields are errors

Every request DTO in the server carries `#[serde(deny_unknown_fields)]` —
`SlotSnapshot` [crates/muser-server/src/state.rs:504-510],
`CreateSessionRequest` [crates/muser-server/src/axum_httpd.rs:3606-3610],
`SlotActionRequest` [axum_httpd.rs:1074-1080], `OllamaOptions`
[axum_httpd.rs:2324-2326], and the rest. A misspelled field is a 400, not a
silently ignored value. Content types are likewise exact — the check is
literal string equality:

```rust
// crates/muser-server/src/axum_httpd.rs:4838-4843
fn exact_json_content_type(headers: &HeaderMap) -> bool {
    headers
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        == Some("application/json")
}
```

The architecture document summarizes the posture: "Request DTOs reject
unknown fields. Intentional rejections are listed in the compatibility
contract" `[docs/muser-architecture.md §Public serving surface]`. This is
fail-closed parsing — the same instinct as the engine refusing an unknown
producer recipe at enrollment — applied to JSON.

## 37.3 Stateful generation: the 409 protocol

Chat completions can be *stateless* (the full conversation rides every
request) or *stateful* (the server keeps the frontier). Stateful generation
is opt-in by supplying three things together — and the "together" is
enforced:

```rust
// crates/muser-server/src/openai.rs:1413-1428
let stateful = match (
    request.session_id.as_deref(),
    request.expected_revision,
    request.idempotency_key.as_deref(),
    request.idempotency_request_sha256,
) {
    (Some(id), Some(revision), Some(key), Some(request_sha256)) => {
        Some((id, revision, key, request_sha256))
    }
    (None, None, None, _) => None,
    _ => {
        return Err(ChatError::BadRequest(
            "session_id, expected_revision, Idempotency-Key, and a canonical request identity are all required for stateful generation"
                .into(),
        ))
    }
};
```

A **session ID**, an **expected revision** (a monotone counter), and an
**Idempotency-Key** (plus a canonical SHA-256 of the request body, computed
by the handler [crates/muser-server/src/axum_httpd.rs:3179-3186]). Any one
without the others is a bad request. Then the store's `begin` runs the
admission logic, and every failure mode in it is a **409 Conflict**
(`ChatError::Conflict` maps to 409 [crates/muser-server/src/openai.rs:656,
668]):

```rust
// crates/muser-server/src/session_store.rs:338-358
let record = records.get_mut(id).ok_or("session does not exist")?;
if let Some(cached) = record.idempotency.get(idempotency_key) {
    if cached.expected_revision != expected_revision
        || cached.request_sha256 != request_sha256
    {
        return Err(
            "Idempotency-Key is already bound to a different session mutation".into(),
        );
    }
    return Ok(BeginMutation::Replay(cached.result.clone()));
}
if record.busy {
    return Err("session is busy".into());
}
if record.revision != expected_revision {
    return Err(format!(
        "session revision conflict: expected {expected_revision}, current {}",
        record.revision
    ));
}
record.busy = true;
```

Read it as three guards in order. **Replay**: the same key bound to the same
revision *and* the same request digest returns the cached result — a network
retry after a lost response is answered with the original completion, not a
second generation [crates/muser-server/src/openai.rs:1437-1465]. The same
key with a *different* request is refused; a key is a commitment.
**Busy**: one mutation per session at a time. **Revision**: an
optimistic-concurrency compare — the client must state which revision it
believes it is extending, and a mismatch is a conflict, never a merge.
`commit` then atomically advances `revision = expected_revision + 1` under
the registry lock and files the idempotency record
[crates/muser-server/src/session_store.rs:385-409] (the per-session
idempotency map is bounded at 64 entries, cleared wholesale on overflow —
bounded everything, again). At most **64 logical sessions** are tracked
(`MAX_LOGICAL_SESSIONS` [crates/muser-server/src/session_store.rs:14];
`[docs/muser-architecture.md §Context and sessions]`).

## 37.4 The session bundle: identities welded to state

What does a session *contain*? The `SessionBundle` — and its field list is
the chapter's thesis in one type:

```rust
// crates/muser-server/src/session_store.rs:18-31
pub(crate) struct SessionBundle {
    pub schema: String,
    pub session_id: String,
    pub revision: u64,
    pub context_epoch: u64,
    pub model_sha256: String,
    pub tokenizer_sha256: [u8; 32],
    pub template_sha256: [u8; 32],
    pub layout_abi: String,
    pub dflash_identity_sha256: Option<String>,
    pub vision_projector_sha256: Option<String>,
    pub vision_preprocessing_sha256: Option<String>,
    pub target: muser_engine::cache::SessionCacheSnapshot,
    pub target_logits: Vec<f32>,
    pub dflash: Option<muser_engine::dflash::DFlashContextSnapshot>,
```

— plus the RNG seed, the full sampler-state snapshot (the four `Mt19937`
streams of [Ch 34 §34.3](34-scheduler-and-slots.md)), sampler history, the
detokenizer's and stop matcher's pending fragments, grammar state with its
own digest, the canonical replay plan, and vision rows
[crates/muser-server/src/session_store.rs:32-47]. The architecture document's
sentence: bundles "bind exact model, tokenizer, template, layout, and state
identities together with target/DFlash state, sampler state, replay
messages, vision rows, context epoch, and revision"
`[docs/muser-architecture.md §Context and sessions]`.

This is [Ch 24](24-kvpack-the-format.md)'s manifest discipline generalized to
a whole session: *state is only restorable against the identities that
produced it.* The chat path enforces it on every continuation — model,
tokenizer, template, the layout ABI string (`"muse-kv-layout-v1"`), the
DFlash draft identity, and the vision projector identities must all match
this server, or the request is a 409
[crates/muser-server/src/openai.rs:1516-1530]. Even the sampler stream is
welded in: "A restored logical session continues the exact sampler stream
that was committed with its KV/logit frontier. A fresh client seed only
applies when creating a new frontier, never midway through one"
[crates/muser-server/src/openai.rs:1489-1493]. A `grammar_sha256` mismatch is
likewise a conflict [crates/muser-server/src/openai.rs:1494-1498]. There is
no "restore anyway."

On disk, the bundle is encrypted and authenticated: serialized with
Postcard, sealed with **XChaCha20Poly1305** under a key held beside the
store, written with a magic `MUSER-SESSION-V3` envelope, a 0700-mode
directory, and an atomic private write
[crates/muser-server/src/session_store.rs:9-10, 15, 428-434]. Restore refuses
anything that is not a private regular file or fails authentication
[crates/muser-server/src/session_store.rs:445-457]. The architecture
document calls these "authenticated encrypted bundles"
`[docs/muser-architecture.md §Context and sessions]` — the tamper-evidence
role the HMAC seal plays on the wire in [Ch 30](30-handoff-v2-transport.md),
played locally for state at rest.

## 37.5 Migration: two-phase, destination first

Sessions move — between decode nodes (authenticated HTTPS between
"identically qualified Muser decoders") or to storage tiers (enrolled kvpack
storage) `[docs/muser-architecture.md §Context and sessions]`. The wire-side
story (deltas, cut points, bit-exactness) is
[Ch 26](26-delta-handoff-and-migration.md); here is the protocol shape:

1. `POST /v1/sessions/{id}/migrate` and
   `POST /__muser/v1/session-transfers/prepare` start an export — mode
   `copy` or `move`, tier validated, a durable transfer journal created, and
   a transfer ID bound irreversibly to `(session, destination, mode, tier)`
   [crates/muser-server/src/session_store.rs:479-512;
   axum_httpd.rs:528-539].
2. The payload moves over the *separate* router of §37.1 — `PUT
   /__muser/v1/session-transfers/{transfer_id}/payload`, the only unlimited
   body lane, capped instead by `MAX_TRANSFER_BYTES = 32 GiB`
   [crates/muser-server/src/session_store.rs:16].
3. `POST /__muser/v1/session-transfers/{transfer_id}/commit` completes the
   two-phase transaction, and status is idempotently queryable
   (`GET /v1/session-transfers/{transfer_id}`) "after ambiguous failures"
   `[docs/muser-architecture.md §Context and sessions]`.

The ordering law is the one [Ch 26](26-delta-handoff-and-migration.md) stated
for caches, holding here for whole sessions: *the destination durably
commits before a move may delete the source.* The journal's own status
vocabulary records the near-miss — a failed post-commit delete leaves the
transfer in `destination_committed_source_retained`, never in a state that
implies the source is gone [crates/muser-server/src/session_store.rs:303-316].
And one boundary is absolute: "GX10 is not a decode destination"
`[docs/muser-architecture.md §Context and sessions]` — the producer prefills;
it never holds sessions.

## 37.6 The security boundary, taught as a design

Here is the chapter's design study. Muser's authorization is *asymmetric on
purpose*, and the architecture document states the whole ladder at once:

> Authorization is deliberately asymmetric:
> - loopback inference is keyless;
> - loopback management needs bearer auth or a same-origin dashboard session;
> - dashboard login exchanges the API key for a Secure, HttpOnly,
>   SameSite=Strict cookie, and cookie-authenticated mutations additionally
>   need an exact Origin match and CSRF token;
> - a nonloopback bind is refused before listening unless a certificate,
>   mode-0600 private key, and mode-0600 API-key file are supplied;
> - every LAN inference, telemetry, WebSocket, session, cancellation, and
>   node management request needs authentication.
> `[docs/muser-architecture.md §HTTP and security boundary]`

Each rung is visible in code. **Loopback inference keyless:** the chat
handler authenticates only when the bind is non-loopback — `if state.lan &&
!valid_bearer(&state, &headers)` [crates/muser-server/src/axum_httpd.rs:3159]
— so `curl localhost` just works. **Management needs a key:** every
snapshot/slots/sessions/nodes handler opens with
`valid_management_auth(...)` (bearer or dashboard cookie; e.g.
axum_httpd.rs:601-604, 1082-1085). **The cookie is bound hard:**
`valid_dashboard_cookie` re-derives the server's own origin from `Host`,
compares it to the origin the session was minted for, requires an *exact*
`Origin` header match for browser WebSockets and all mutations, and compares
the CSRF token in constant time [crates/muser-server/src/axum_httpd.rs:4879-4916].
(Login itself demands TLS and an exact Origin match before it will mint
anything [axum_httpd.rs:3532-3563].) A quick definition for the reader new
to this: **[CSRF](../glossary.md#csrf)** is the attack where some other
website's page makes your browser send a request *with your cookie* to a
service you are logged into — an exact-Origin check plus a token the other
site cannot read is the standard counter, and both are present.

**The nonloopback wall is config-time, not request-time.** Before the server
listens at all, bind validation runs:

```rust
// crates/muser-server/src/axum_httpd.rs:256-265
let lan = addresses.iter().any(|address| !address.ip().is_loopback());
let tls_pair = security.tls_cert.is_some() && security.tls_key.is_some();
if security.tls_cert.is_some() != security.tls_key.is_some() {
    return Err("--tls-cert and --tls-key must be supplied together".into());
}
if lan && (!tls_pair || security.api_key_file.is_none()) {
    return Err(
        "nonloopback serving requires --tls-cert, --tls-key, and --api-key-file".into(),
    );
}
```

And the key files must be *private regular files* — mode `0600` or stricter,
refusing symlinks [crates/muser-server/src/axum_httpd.rs:281-295]. You
cannot expose this server to the network by forgetting a flag: exposure
requires a certificate, a locked-down key, and a locked-down API-key file,
or the process refuses to listen.

**WebSockets never carry long-lived keys in URLs.** The telemetry socket
accepts either a dashboard cookie (with Origin enforced) or a ticket, and
the comment states the threat: "Browser WebSockets are not protected by
fetch's same-origin response rules. A dashboard cookie therefore requires an
explicit Origin match; long-lived bearer auth must first be exchanged for a
single-use ticket" [crates/muser-server/src/axum_httpd.rs:4734-4737]. The
ticket mint is thirty seconds and single-use by construction:

```rust
// crates/muser-server/src/axum_httpd.rs:3594-3602
let ticket = random_secret();
let expires = Instant::now() + Duration::from_secs(30);
// …(insert into the ticket registry; elided)…
Json(serde_json::json!({"ticket": ticket, "expires_in": 30, "single_use": true}))
```

and `consume_ticket` removes it from the registry on first use
[crates/muser-server/src/axum_httpd.rs:4753-4771]. Default CORS is none, the
dashboard is same-origin with no override, and the local-CA operator
workflow (`muser tls init` / `muser tls issue`) is separate
`[docs/muser-architecture.md §HTTP and security boundary]`.

Why call this a *design* rather than a checklist? Because the asymmetry
encodes a model of who the attacker is. On loopback, the parties are the
user's own processes — so friction is the only cost of auth, and inference
is keyless while *mutations* still require intent. Off loopback, the model
inverts: nothing is trusted, so everything authenticates, exposure demands
explicit cryptographic readiness, and even then the strongest secrets never
appear in URLs where logs and history collect them.

## 37.7 Bounded everything

Collected in one place, because the bounds are the server's whole
resource story (each was met earlier in its natural habitat; Table 37.2
is the inventory):

| Bound | Value | Where |
|---|---|---|
| Request body | 64 MiB (`MAX_BODY`) | axum_httpd.rs:53, 544 |
| Request-body timeout | 30 s | axum_httpd.rs:545 |
| Concurrent HTTP requests | 256 | axum_httpd.rs:546 |
| Migration payload lane | unlimited body / 60 min / 4 concurrent | axum_httpd.rs:547-554 |
| Single transfer payload | 32 GiB | session_store.rs:16 |
| Waiting admissions | 64 (`MAX_QUEUED_REQUESTS`) | state.rs:253 |
| Streaming channel depth | 64 (`STREAM_CHANNEL_DEPTH`) | axum_httpd.rs:54 |
| Backpressured writer | 5 s grace, then cancel (499) | axum_httpd.rs:55, 2271-2289 |
| Logical sessions | 64 | session_store.rs:14 |
| Idempotency records per session | 64 (then cleared) | session_store.rs:398-400 |
| Resident generations | 4 + staging (never a 5th server) | state.rs:240-243 |

*Table 37.2: the bound inventory. Every queue in the process has a number;
nothing waits forever, and each overflow is a defined HTTP status, not a
hang.*

## 37.8 The unhealthy latch: fail-closed as an HTTP status

[Ch 34 §34.4](34-scheduler-and-slots.md) showed the `SlotPool` latching
itself unhealthy when a lease is poisoned. Here is the client-visible half.
`/health` reports the latch directly:

```rust
// crates/muser-server/src/axum_httpd.rs:738-754
let healthy = state
    .server
    .inference
    .as_ref()
    .is_some_and(|runtime| runtime.slots.is_healthy());
if healthy {
    Json(serde_json::json!({"status": "ok"})).into_response()
} else {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        // …(503 body; elided)…
```

The engine side of the same rule, from the architecture document: "If
rollback or accelerator state becomes uncertain, the engine latches
unhealthy and serving returns 503 until restart"
`[docs/muser-architecture.md §Model and engine]`. Note what this is *not*:
not a retry-after hint, not a self-healing reset — a wall, held until an
operator restarts the process. The rationale is the slot pool's own comment:
recovery-in-place "avoids serving from uncertain GPU state" — nobody can vow
for the bits anymore, so nobody is allowed to serve them
[crates/muser-server/src/state.rs:474-478]. Status routes speak the same
language ("accelerator state is unhealthy", axum_httpd.rs:1066-1070), and
`/healthz` exposes the latch for orchestrators alongside a `degraded` flag
[axum_httpd.rs:758-775]. The philosophy is [Ch 39](39-the-evidence-culture.md)'s
subject; here it matters that fail-closed composes cleanly with HTTP — the
client sees an ordinary, cache-unfriendly 503, and load balancers do the
right thing without knowing why.

## 37.9 Tradeoffs

**Framing strictness vs compatibility reach.** `deny_unknown_fields` plus
exact content types means a client sending an unknown field or
`application/json; charset=utf-8` gets an error where llama.cpp might
tolerate it. The trade is recorded as deliberate — intentional rejections
are *listed* in the compatibility contract
`[docs/muser-architecture.md §Public serving surface]` — and it buys a
surface that cannot silently accept a request it half-understands, the same
discipline the engine applies to GGUF metadata
([Ch 9](09-muse-glimmer-architecture.md)).

**Optimistic revision CAS vs server-side merge.** The 409 protocol refuses
concurrent writers instead of reconciling them. Clients must retry with the
new revision — extra round trips for a genuinely racing client, in exchange
for a frontier that is never a blend of two histories. The idempotency
replay path is what makes retry safe [crates/muser-server/src/session_store.rs:339-348];
without it, optimistic concurrency would punish exactly the flaky networks
that cause retries.

**Identity-welded bundles vs portable state.** A bundle refuses to restore
on any identity drift (model, tokenizer, template, layout, draft, vision —
[crates/muser-server/src/openai.rs:1516-1530]). Portability loses: you
cannot carry a session across a model update. Exactness wins everything:
a restored session continues the *same bits* the committed frontier
implies, which is the property [Ch 25](25-warm-reuse.md) and
[Ch 26](26-delta-handoff-and-migration.md) had to prove on the KV side with
receipts.

**Refuse-to-listen vs warn-and-serve.** The nonloopback gate could have been
a log line; it is a hard error before bind (axum_httpd.rs:261-265) with
file-mode checks on the secrets (281-295). The cost is setup friction for
every legitimate LAN deployment — TLS material must exist first. The benefit
is that the insecure configuration is unreachable, not merely discouraged;
combined with keyless-loopback, the security posture is chosen by *where*
you bind, not by flags you forgot.

**503-until-restart vs in-place recovery.** The latch turns a poisoned
mutex or an uncertain rollback into operator-visible downtime. The
alternative — reset and continue — would manufacture confidence in state
that may already be wrong; the campaign's own hybrid-fusion postmortem
([Ch 35](35-ordering-hazards-and-the-dispatch-gap.md)) is the precedent for
why Muser does not ship "probably fine" states. No serving-availability
measurement of the latch exists [unverified]; it is a correctness decision
priced in downtime.

## 37.10 What comes next

That is the whole engine, end to end: memory model, quantization, the model,
the decode kernels, the KV cache and its portable format, the disaggregated
lane, and now the orchestration and serving surface that expose it to
humans. Parts I–VII have made hundreds of claims, and every one of them
ended in a tag — a receipt path, a ledger row, a claims-register entry.
Part VIII is about those tags: how the parity matrices were actually run,
what a ratio may and may not say, why the J0 anchor flip mattered, and the
evidence culture that keeps copy from outrunning receipts.
[Ch 38](38-measuring-against-llama-cpp.md) begins with the comparator.

## References

- `crates/muser-server/src/axum_httpd.rs:487-559` — the router (Table 37.1)
  and the two-layer body/timeout/concurrency split.
- `crates/muser-server/src/axum_httpd.rs:53-55` — `MAX_BODY`,
  `STREAM_CHANNEL_DEPTH`, `SLOW_CLIENT_GRACE`.
- `crates/muser-server/src/axum_httpd.rs:244-295` — bind-security validation
  (quoted) and the 0600 regular-file checks.
- `crates/muser-server/src/axum_httpd.rs:733-775` — `/health` and
  `/healthz` (quoted).
- `crates/muser-server/src/axum_httpd.rs:3152-3186` — the chat handler:
  loopback-keyless inference, idempotency-key capture, request digest.
- `crates/muser-server/src/axum_httpd.rs:3532-3563, 3590-3604, 4727-4771` —
  dashboard login; the 30 s single-use ticket (quoted); WebSocket auth and
  `consume_ticket`.
- `crates/muser-server/src/axum_httpd.rs:4845-4916` — `valid_bearer`,
  `valid_management_auth`, origin-bound cookie + constant-time CSRF
  (§37.6).
- `crates/muser-server/src/openai.rs:656, 668, 1413-1428, 1431-1465,
  1489-1530` — Conflict→409; the stateful tuple (quoted); replay; the
  sampler-stream and identity-binding rules.
- `crates/muser-server/src/session_store.rs:9-47, 14-16` — `SessionBundle`
  (quoted), sampler snapshot, session/transfer caps.
- `crates/muser-server/src/session_store.rs:303-316, 325-410, 412-477,
  479-512` — migration disposition vocabulary; `begin`/`commit` (quoted);
  encrypted save/restore; export journal.
- `crates/muser-server/src/state.rs:240-243, 253-254, 474-483, 504-510` —
  staging isolation; admission constants; the unhealthy latch;
  `deny_unknown_fields` DTOs.
- `[docs/muser-architecture.md §Product boundary, §Context and sessions,
  §HTTP and security boundary, §Public serving surface, §Model and engine]`
  — the frozen surface, the 409/bundle contract, the asymmetric auth ladder
  (quoted), the latch sentence.
- [Ch 24](24-kvpack-the-format.md) — the manifest/identity discipline this
  chapter's bundles inherit; [Ch 26](26-delta-handoff-and-migration.md) —
  migration's KV-side exactness; [Ch 34](34-scheduler-and-slots.md) — the
  latch's scheduler side; [Ch 38](38-measuring-against-llama-cpp.md) —
  where the numbers come from; [Ch 39](39-the-evidence-culture.md) —
  fail-closed as philosophy.
