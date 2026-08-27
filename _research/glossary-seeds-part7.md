# Glossary seeds — Part VII (Chapters 34–37)

One line per term introduced or load-bearing in chapters 34–37, for merge
into `glossary.md`. Format: `### term — one-line definition (Ch N)`.
Terms already seeded by earlier chapters (slot (serving) Ch 22,
AcceleratorScheduler Ch 2, memoryBarrierWithScope Ch 2, hazard tracking
Ch 3, dispatch gap (+196) / PhaseProfiler closures Ch 10, SWA staging Ch 16,
staging shadow/session Ch 23, arithmetic intensity Ch 1, roofline flip /
TTFT Ch 27, 104 norm-boundary groups Ch 19, two-phase migration Ch 26,
sampler state Ch 21, ExactIdentityV1 Ch 24) are not repeated here.

### decode-over-prefill priority — the scheduler rule that any waiting decode outranks all prefill: prefill acquires only when no decode is queued (Ch 34)
### decode-aware chunk shrinking — prefill boundaries collapse from 512 rows to 64 the moment any decoder queues, capping decode's worst-case wait at one small interval (Ch 34)
### cyclic slot rotation — fairness by ascending sequence-ID order resuming after the last-served ID, implemented identically at both scheduler levels (Ch 34)
### AcceleratorPermit — the RAII guard returned by scheduler acquire; dropping it releases the accelerator and wakes the next waiter (Ch 34)
### packed decode group — `forward_decode_group`: 1..=4 ready decode rows from distinct slots packed into one concurrent encoder, one commit, one wait — one weight pass (Ch 34)
### DecodeBatcher (250 µs rendezvous) — the server-side decode-step coalescer: request threads keep slot ownership while one elected runner waits ≤250 µs to pack up to four rows (disabled at parallel=1) (Ch 34)
### SlotPool — bounded server admission for the 1..=4 resident slots: at most 64 waiters, immediate overload rejection past that, and the permanent unhealthy latch on poisoned state (Ch 34)
### staging generation — the full-capacity rebuild context (target + DFlash) kept deliberately outside the slot pool so atomic context shift can never become a fifth serving slot (Ch 34)
### slow-client cancellation — a streaming writer backpressured past its 5 s grace (channel depth 64) is cancelled with 499; a slow reader can waste its own request, never the accelerator (Ch 34)
### RAW hazard — read-after-write: a consumer kernel must see a producer's bytes; the dominant decode dependency, ordered by the store→barrier→attend pattern (Ch 35)
### WAW hazard — write-after-write: two writers of one buffer must serialize or the final bytes are order-dependent (Ch 35)
### WAR hazard — write-after-read: an overwriter must wait for outstanding readers — the ring-lap risk explicit origin bookkeeping prevents (Ch 35)
### RAR non-hazard — read-after-read: concurrent readers of immutable data need no ordering; the packed decode group's shared weight reads (Ch 35)
### tracked-by-default (b9678d4) — every Muser buffer allocates with Metal hazard tracking on; the one global-untracked experiment changed DFlash conditioning with identical greedy IDs and was reverted (Ch 35)
### targeted resource barrier — `memory_barrier_with_resources` naming exactly the buffers a dependency flows through (K/V planes, partials, staged shadow) instead of a whole-scope stall (Ch 35)
### serial prefill dispatch — `MUSER_SERIAL_PREFILL_DISPATCH`: restores the serial prefill encoder for exact A/B against the default concurrent grouping (Ch 35)
### hybrid postmortem — the removed retained-activation fusion whose one-f16-ULP layer-1 divergence became 201,970/202,048 differing logits and 3.197e-4 logprob error against the 1e-4 contract (Ch 35)
### bit-exactness-over-throughput — the disposition that keeps the 104 separated norm boundaries because their available fusion changes logprobs beyond contract (Ch 35)
### flash_contiguous — the prefill route predicate (origin 0, no wrap, fits capacity) that admits the direct store-then-attend routes before any staging (Ch 36)
### pinned vec per-query prefill — short chunks (<20 queries) run llama's own vec flash kernel once per query row with exact visible prefixes, reusing the upstream PSO and reduction order (Ch 36)
### mask/blk block classifier — llama's `flash_attn_ext_blk` skip/partial/dense tile bytes, prepared once per chunk and shared by every full-attention layer, making the pinned causal kernel cheap on the masked triangle (Ch 36)
### staging-shadow route — wrapped-SWA prefill: stage old ring rows plus the chunk into a detached F16 shadow (llama's padded absolute indices for the one-row variant), attend from the shadow, commit ring metadata only after (Ch 36)
### encode_batch_hidden_range — the batch prefill driver: the full 52-layer graph over T rows with staging shadows, mask/blk, capture layers, and entry/output switches — prefill's twin of encode_token (Ch 36)
### TPOT (time per output token) — the decode-dominated per-token latency a streaming user feels; DFlash's target, as TTFT is disaggregation's (Ch 36)
### M16 NVFP4 batch route — the W4A4 quantized-activation GEMM taken by 16-row NVFP4 chunks with 64-aligned inputs (opt-out MUSER_NO_M16_N32) (Ch 36)
### stateful generation — chat requests carrying session_id + expected_revision + Idempotency-Key (all or none); the server holds the frontier and refuses concurrent writers with 409 (Ch 37)
### Idempotency-Key replay — a retry of the same key+revision+request-digest returns the cached completion instead of generating again; the same key on a different request is a conflict (Ch 37)
### revision CAS — optimistic concurrency for sessions: commit advances revision only if the record still holds the client's expected value (Ch 37)
### session bundle — the durable unit binding model/tokenizer/template/layout-ABI/DFlash/vision identities to target KV, logits, RNG streams, sampler, grammar, detokenizer, and replay state (Ch 37)
### encrypted session envelope — the on-disk bundle format: Postcard bytes sealed with XChaCha20Poly1305 under a MUSER-SESSION-V3 magic, 0700 directory, atomic private write (Ch 37)
### CSRF — cross-site request forgery: a hostile page makes the victim's browser send a cookie-authenticated request; countered by exact-Origin matching plus a constant-time token on mutations (Ch 37)
### WebSocket ticket — the 30-second single-use secret minted by /v1/ws-tickets so long-lived bearer keys never ride URLs; consumed and removed on first use (Ch 37)
### nonloopback bind gate — the server refuses to listen on a non-loopback address unless a TLS certificate, a mode-0600 key, and a mode-0600 API-key file are all supplied (Ch 37)
### asymmetric auth model — keyless loopback inference, bearer-or-dashboard management, cookie+CSRF+Origin mutations, authenticated-everything on LAN — security posture chosen by where you bind (Ch 37)
### unhealthy latch — poisoned accelerator/session state flips a permanent flag: every request 503s until an operator restarts; fail-closed as user-visible HTTP (Ch 37)
### bound inventory — every queue in the server has a number (64 MiB bodies, 30 s timeouts, 256 connections, 64 admissions, 64-deep streams, 64 sessions); overflow is a status code, never a hang (Ch 37)
