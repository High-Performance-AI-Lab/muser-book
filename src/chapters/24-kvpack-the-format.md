# Chapter 24 — kvpack: the format
> **status:** polished  ·  **path:** Muse Glimmer, pinned Muser tree

*Prerequisites: [Ch 23](23-the-swa-ring-and-the-growing-cache.md) (the
interchange snapshot and its fail-closed shape gate), [Ch 22](22-the-price-of-context.md)
(what a cache weighs), [Ch 14](14-qk-norm-and-rope.md) (why NoPE rows are
relocatable bytes). Some exposure to [HMAC]/[SHA-256] as "keyed and unkeyed
digests" is assumed; both get their one-sentence definitions at the top of
[Ch 30 §30.1](30-handoff-v2-transport.md).*

---

## 24.1 What kvpack is

[Ch 23](23-the-swa-ring-and-the-growing-cache.md) ended inside one process:
the interchange snapshot was a byte contract, but nothing outside the engine
held it. kvpack is the thing that holds it. It is the format and protocol
family that moves a prefilled KV cache between producer and consumer —
between processes, between disks, and in [Part VI](27-why-disaggregate.md)
between a CUDA producer on a GX10 and the Metal decoder on a Mac. The
engineering stance is stated in one line of its overview doc: **"exactness
is the product; speed is the consequence"** `[docs/kvpack.md]`. Restored
state is proven byte-identical to what was saved and proven to carry the
exact computation identity it was produced under; anything short of both
proofs is a loud refusal, never a best-effort restore.

Three questions define the format's job, and kvpack answers each
mechanically rather than by convention `[docs/kvpack.md §Why it exists]`:

1. **Is this cache the cache I asked for?** — keyed, fail-closed identity.
2. **Have I seen this request before?** — replay protection.
3. **Did anyone touch it in flight?** — layered integrity and authenticated
   sealing.

The scale motivating all three is already derived: a 65k prompt's cache is
hundreds of megabytes; a 131k prompt's is nearly a gigabyte
([Ch 22](22-the-price-of-context.md), Table 22.1); the measured deep wire
payload is 1,823,184,896 B `[receipt phase4-disagg-20260820/130815-g900091/]`.
State that size cannot be re-checked by eyeball; it needs proof objects.

One definition before the bytes. **Replay**, in kvpack's vocabulary, means
*restoring inference state* — not replaying a request log, not semantic
caching:

> Here, **replay means restoring inference state**. kvpack is not an
> inference engine, a semantic prompt cache, or a request/response log. It
> stores and verifies the exact bytes an engine adapter gives it; the engine
> remains responsible for what those bytes mean.
> `[third_party/kvpack/README.md]`

That sentence is the format's integration boundary, and it is worth
pausing on, because every refusal in the rest of this chapter follows from
it. The line the README draws is a line between *provable claims*. kvpack
owns everything it can prove from bytes alone: the pack format, the durable
write protocol, validation, indexing, restore I/O, exact-token and
compatibility-identity matching, failure ordering, byte-for-byte
preservation. The engine adapter owns everything only a running engine can
prove — synchronizing device work before export, serializing its state
objects, installing and committing restored state, deciding where prefix
checkpoints are valid, bumping `engine_abi` when semantics change, and
"proving that restore produces the same model behavior as uninterrupted
execution" `[third_party/kvpack/README.md §The integration boundary]`. Said
the other way round: kvpack can prove the bytes came back unchanged and
were selected under the expected identity; only the engine can prove those
bytes are correct KV for its runtime.

## 24.2 The vendored tree and its provenance discipline

A format is only as trustworthy as the copy of it you are actually
running. So before any bytes: where does kvpack live in Muser's tree, who
is allowed to change it, and how would we find out if someone had?

In Muser's workspace, kvpack is not a crates.io dependency. It is vendored
— a hash-pinned snapshot living at `third_party/kvpack`, excluded from the
workspace's own crates, with every file's SHA-256 recorded. The adapter
crate's module doc explains why this one dependency gets special treatment:

```rust
// crates/muser-kvpack/src/lib.rs:3
//! kvpack is the **one** shared external dependency in `muser`
//! (docs/muser-architecture.md §1), pinned to a path-pinned release source —
//! the `release/muser-alpha2` branch's `kvpack-core`/`kvpack`/`kvpack-handoff`
//! crates at `0.1.0-alpha.2` (`docs/release-provenance.md`), not a live git
//! dependency. It defines the sealed V2 wire format (HMAC over a canonical
//! manifest) that the CUDA producer on GX10 and the Metal consumer on Mac
//! must agree on: the producer reimplements that format itself
//! (`scripts/gx10/llamacpp/spark_kv_export.cpp` + `muser_v2_send.py`) rather
//! than linking this crate, so agreement isn't "one format authority linked
//! on both sides" but cross-verification — the authenticated Mac-side
//! receiver rejects anything the producer's reimplementation gets wrong,
//! which is why HMAC verification on restore is load-bearing, not a
//! formality.
```

Read that twice, because it is unusual: the producer does not link the
crate. The CUDA side reimplements the format in C++/Python, and the two
implementations agree *because* the receiver verifies an HMAC over
canonical bytes and rejects anything that diverges — agreement by
cross-verification, not by shared code `[docs/kvpack-merge-handoff §6]`.
The seal is the format authority.

The vendored tree is three crates (`provenance.json workspace_members`):

- **kvpack-core** — the pure in-memory pack codec, no file I/O:
  `canonical.rs` (canonical JSON), `chunk.rs` (content-addressed chunks),
  `manifest.rs`, `pack.rs`, `identity.rs`, `keys.rs`
  (longest-committed-prefix lookup), `quant.rs`, `rotation.rs`,
  `validator.rs` `[code-map §8.2]`.
- **kvpack** — the engine-facing store: `store/`, `restore/`, `writer/`,
  `export/`, `gguf_layout/`, the adapter contract, the pack bridge.
- **kvpack-handoff** — the sealed V2 wire format this chapter quotes
  (`handoff_v2.rs`, `manifest.rs`, `receiver/`, `mac.rs`).

`provenance.json` pins the snapshot: schema `muser.vendored-source.v1`,
upstream `https://github.com/High-Performance-AI-Lab/kvpack` at commit
`70c34c7d790dbfc9c1271727dd34ea0e863404d2`, tag
`kvpack-v0.1.0-alpha.2-rc1`, upstream tree `7d56417c…`, and a per-file
SHA-256 map covering every source file in the tree — from
`crates/kvpack-core/src/canonical.rs` to the conformance tests
(`third_party/kvpack/provenance.json`). `python3
scripts/audit_vendored_kvpack.py` re-verifies the hashes against the tree,
so a silent local edit to the vendored format is a detectable event, not a
drift.

Two carried patches are recorded in the same file, each with its reason.
The first is the whole argument of this chapter compressed into a bug we
did not see coming, so it is worth walking through the way we met it.

Here is the fork. Both ends of a transfer serialize a descriptor to JSON
and hash the result; both were written against the same canonical-encoding
rule; so we expected the two digests to agree, and for a long while they
did. Then the workspace pinned serde_json's `preserve_order` feature — a
change made for unrelated reasons, by someone thinking about map iteration,
not about wire formats. What we expected from that change was nothing at
all. What we got was live transfers dying:

```json
// third_party/kvpack/provenance.json (patches[0], abridged)
"id": "canonical-json-sorted-keys-feature-independent",
"reason": "canonical_json passes through serde_json::Value, whose map
order follows the preserve_order Cargo feature; with preserve_order
pinned workspace-wide the receiver emitted insertion-ordered descriptor
bytes while the Spark producer emits sorted-key bytes, so Handoff V2
terminal seals failed (descriptor_sha256 mismatch) and live transfers
were dropped."
```

Read that reason field as a chain, because every link of it is ordinary.
`canonical_json` handed its map through `serde_json::Value`; with
`preserve_order` on, a `Value` remembers insertion order instead of sorting
its keys; the receiver therefore emitted descriptor bytes in one order
while the Spark producer emitted them sorted; the two `descriptor_sha256`
values diverged; and the terminal seal refused to match. Nothing was
corrupted. Nothing was silently accepted. The transfer simply died at the
seal, loudly, on a live link.

That is the lesson, and it cuts both ways. Two "identical" implementations
disagreed over something no code review would have flagged, because the
disagreement did not live in either implementation — it lived in a feature
flag. And the seal caught it anyway, because a seal does not care what
either side *meant* to encode. This is exactly the failure the
cross-verification design exists to catch, caught in the wild; the decision
it produced was to make the canonical encoding (recursive sorted keys)
feature-independent, so that no dependency flag can quietly redefine what
"canonical" means.

The second patch needs no story: it adds a domain-separated protocol
HMAC on the MAC key (`mac.rs`, domain tag `b"kvpack-domain-mac-v1\0"`),
consumed by muser-cluster's verifier lane `[third_party/kvpack/provenance.json
patches[1]]`. Both patches live as recorded, receipted facts rather than
local edits — and the merge ruling
of 2026-08-20 makes muser's vendored copy the canonical kvpack, with merge
direction vendor → upstream `[docs/kvpack-merge-handoff §1]`.

## 24.3 The durable container: pack v1

Start with the easy case — producer and consumer on the same machine,
talking through a file — and ask what has to be true of that file before a
restore from it can be called proven. Two things, and they are the two
things filesystems are worst at. It must be impossible to read a
half-written file as if it were whole. And it must be impossible to change
a written file without the change announcing itself.

The on-disk unit is a **pack**: an append-only immutable file whose shape
is header ‖ body ‖ footer with fixed-size bookends:

```text
  ┌──────────────┬─────────────────────────────┬──────────────┐
  │ 4 KiB header │  canonical manifest         │ 4 KiB footer │
  │              │  (optional ChaCha20Poly1305 │              │
  │              │   envelope, AAD = header)   │              │
  └──────────────┴─────────────────────────────┴──────────────┘
```

*Figure 24.1: pack v1 layout, per the architecture map — "4 KiB header ‖
canonical manifest (optional ChaCha20Poly1305, AAD = full header) ‖ 4 KiB
footer; footer HMAC binds body length and file size (truncation fails
closed)" `[docs/kvpack-merge-handoff §5]`.*

The integrity model is layered, and the doc lists the layers in a single
breath: "record headers and payloads are hashed, object IDs are
content-derived, the terminal commit carries an ordered inventory, a
canonical Merkle root binds it, and the footer seals the header digest plus
every byte of the file" so that "truncation, single-bit flips, reordering,
substitution, and length games are rejected" `[docs/kvpack.md]`.

That is dense, so unpack it as a ladder in which each rung catches a
different lie. Hashing records catches a flipped bit. Deriving object IDs
from content catches an object substituted under an honest-looking name.
The ordered inventory catches a reordering. The Merkle root over that
inventory catches an addition or a deletion. And the footer, by binding
body length and file size, catches the truncation that would otherwise look
like a perfectly valid shorter file. No rung on that ladder is aspirational;
every one of them is verified by exhaustive truncation and bit-flip
conformance corpora across Rust, Python, and C99 reference implementations
that produce byte-identical packs `[third_party/kvpack/README.md
§Conformance]`.

Three properties matter for everything downstream in this Part:

- **Crash-safe publication.** Packs are append-only; the commit is written
  last; pack sets publish through an exclusive atomic rename. "A torn
  write or SIGKILL mid-write (injected in tests) can never replace the
  last known-good generation" `[docs/kvpack.md]`.
- **Content-addressed chunks.** Payloads split into chunks of at most 4 MiB
  plaintext on token-aligned boundaries, each chunk addressed by its
  content `[docs/kvpack-merge-handoff §5]`. Chunk IDs bind token offset —
  no cross-position dedup — which is what makes prefix lookup exact rather
  than fuzzy.
- **Keyed prefix identity.** The cache key is a keyed HMAC prefix chain,
  one node per 256-token block, "context-bound to tenant ‖ semantic-model
  ‖ family ‖ aux root; no token witness retained; trailing partial block
  `reusable: false`" `[docs/kvpack-merge-handoff §5]`. Raw prompt text
  never appears in paths or telemetry — privacy by construction, since the
  restorer must prove the prefix bytes to even look them up
  `[docs/kvpack.md §Privacy by construction]`.

An honest threat-model line completes the picture: "pack hashes prove
integrity, not authenticity — a whole-file rewrite attacker needs the
`kvenc` envelope or the transport-layer authentication above"
`[docs/kvpack.md]`. Unencrypted packs detect accidents; the ChaCha20-Poly1305
envelope (`kvenc`) or the mTLS channel of §24.6 handles adversaries.

## 24.4 The Handoff V2 wire objects

The durable pack moves a cache between *processes on trustable storage*.
The disaggregated lane needs something harder: a wire protocol for state
that crosses a network between different engines. The difference is not
cosmetic. A file can be re-read from the top as many times as you like, so
a pack can afford to put its proof in a footer; a stream arrives once, in
order, and whatever the receiver intends to check it must be able to check
as the bytes go past — and it must be able to refuse before it has
installed anything into a live engine. That protocol is Handoff V2, defined
in `kvpack-handoff/src/handoff_v2.rs`.

Its model is **components × segments**: a transfer declares components
(target KV required, DFlash context optional, vision refused at Muser
admission), and each component arrives as an ordered stream of segments —
one segment per plane tile, each with its own descriptor and payload
digest. The descriptor is the atom of the format:

```rust
// third_party/kvpack/crates/kvpack-handoff/src/handoff_v2.rs:59
pub struct SegmentDescriptorV2 {
    pub sequence: u32,
    pub component_id: String,
    pub role: SegmentRoleV2,
    pub layer: Option<u32>,
    pub logical_start: u64,
    pub logical_count: u64,
    pub element_type: String,
    pub elements_per_token: u32,
    pub byte_len: u64,
    pub sha256: String,
}
```

`SegmentRoleV2` is a closed nine-variant set — `NopeKey/NopeValue`,
`SwaKey/SwaValue`, `NopeTile/SwaTile` (packed multi-plane tiles),
`DflashKey/DflashValue`, `Auxiliary` (`handoff_v2.rs:41-53`) — [Ch 22](22-the-price-of-context.md)'s
two-regime byte economics, made wire vocabulary: the roles exist because
NoPE and SWA bytes travel differently (tiles stream during prefill; window
groups ride the schedule `[docs/kvpack-merge-handoff §6]`).

A transfer opens with a begin manifest that binds everything the receiver
must agree to *before any payload byte* arrives:

```rust
// third_party/kvpack/crates/kvpack-handoff/src/handoff_v2.rs:73
pub struct BeginManifestV2 {
    pub protocol: String,
    pub transfer_id: String,
    pub generation: u64,
    pub created_unix_ms: u64,
    pub expires_unix_ms: u64,
    pub identity: ExactIdentityV1,
    pub prompt_token_ids: Vec<u32>,
    pub multimodal: Option<MultimodalIdentityV2>,
    pub hmac: HmacIdentityV2,
    pub components: Vec<ComponentV2>,
    /// Streaming producers cannot know hashes for future KV tiles at begin
    /// time. In deferred mode each ordered segment frame carries its complete
    /// descriptor; the terminal seal still binds the canonical descriptor
    /// stream and all payload bytes before commit.
    #[serde(default, skip_serializing_if = "is_false")]
    pub deferred_segments: bool,
    pub segments: Vec<SegmentDescriptorV2>,
}
```

Validation of the begin (`ValidatedBeginV2::validate`,
`handoff_v2.rs:100-160`) is fail-closed on each field, and the list is best
read as the questions the receiver settles before it has spent a byte of
memory on this transfer. Is this conversation the one we agreed to have —
exact protocol string, bounded transfer id, non-expired lifetime? Is the
sender who it claims to be, and not an echo — `hmac.key_id` equal to
the enrolled key, `hmac.epoch` at or above the minimum (the replay
floor), hex-shaped identity digests? And is the payload shaped like
something installable — exactly one of declared/deferred segment mode,
nonempty prompt tokens, unique component ids with a required `TargetKv`?
Any one of them failing ends the transfer there, at the cheapest
possible moment. A transfer closes with the seal:

```rust
// third_party/kvpack/crates/kvpack-handoff/src/handoff_v2.rs:233
pub struct SealCoreV2 {
    pub transfer_id: String,
    pub generation: u64,
    pub begin_sha256: String,
    pub descriptor_sha256: String,
    pub payload_sha256: String,
    pub segment_count: u32,
    pub total_bytes: u64,
}
// … :242
pub struct SealManifestV2 {
    pub core: SealCoreV2,
    pub hmac_sha256: String,
}
```

`SealManifestV2::sign` hashes the canonical JSON of every descriptor in
sequence order and every payload byte, refuses to sign any descriptor whose
`byte_len` or `sha256` disagrees with its payload ("cannot sign mismatched
segment material", `:265`), and then tags the whole core with the shared
key: `hmac_sha256 = key.tag_hex(&canonical_json(&core)?)` (`:285`). That
tag is the terminal seal — no unkeyed mode exists — and the receiver
re-verifies it before anything installs (`verify` at `:447-449` per the
merge map). This is the exact digest that the canonical-JSON patch of
§24.2 made feature-independent; a single reordered key anywhere in the
descriptor stream would have broken it.

Two supporting objects from the v1/v2 manifest complete the identity and
layout story. **`ExactIdentityV1`** is the compatibility namespace —

```rust
// third_party/kvpack/crates/kvpack-handoff/src/manifest.rs:58
pub struct ExactIdentityV1 {
    pub adapter_sha256: String,
    pub chat_template_sha256: String,
    pub context_policy_sha256: String,
    pub model_revision: String,
    pub model_sha256: String,
    pub tokenizer_revision: String,
    pub tokenizer_sha256: String,
}
```

— model, tokenizer, chat template, context policy, adapter, each pinned
either by content digest or by revision string. The doc counts eight
runtime inputs in all, once quantization and engine ABI join the set at the
descriptor level `[docs/kvpack.md]`. The point of enumerating them is that
a cache is only meaningful relative to the machine that produced it: change
any one of the eight and the stored bytes stop being an answer to the same
question.

**`LayoutClassV2`** describes
a layer class compactly — `from..until` stepped by `step`, minus `except`,
with `kv_heads`, `head_dim`, `dtype`, `roles`, and `window_tokens`:

```rust
// third_party/kvpack/crates/kvpack-handoff/src/manifest.rs:92
pub struct LayoutClassV2 {
    pub class: String,
    pub dtype: String,
    pub except: Vec<u32>,
    pub from: u32,
    pub head_dim: u32,
    pub kv_heads: u32,
    pub roles: Vec<TensorRoleV1>,
    pub step: u32,
    pub until: u32,
    pub window_tokens: u32,
}
```

"A class with `window_tokens > 0` ships only the trailing in-window tokens
of each plane" (`manifest.rs:88-89`) — the SWA declaration, in format form.
Muse needs exactly two classes: the 39 SWA layers are every layer with
`layer % 4 != 3`, and NoPE is expressed the other way — `from 3 until 52
step 4` — with the rest SWA (the partition rule of
[Ch 15 §15.1](15-kv-store-and-the-ring.md));
§24.5 shows the adapter deriving and cross-checking that table.

## 24.5 The Muse adapter: `muser-kvpack`

Everything so far is model-agnostic: kvpack does not know what Muse is, and
that is on purpose. Somebody has to tell it — how many layers there are,
which of them are windowed, what scalar constants the math ran under — and
that somebody is the adapter. So the questions for this section are where
Muse's shape enters the format, and what happens when the shape the adapter
believes disagrees with the model the engine actually loaded.

The Muse-specific half lives in
`crates/muser-kvpack`, which "re-exports the pinned-release API and adds
three things that are muser's own product surface" — the Muse K1/K3 layout
glue (`layout`), session save/restore plus relocation-as-memcpy
(`session`), and the dashboard's cache-economics accounting (`economics`)
(`crates/muser-kvpack/src/lib.rs:28-32`). The keys landed upstream are
recorded in the same doc, and each one guards a different way a cache
could lie. Muse layout table **K1** (NoPE theta = 0, fail-closed) pins the
fact that the NoPE layers never rotate — the memcpy free lunch the later
chapters lean on exists because K1 refuses any layout that would touch
those bytes. **K3** (two-class 39-SWA/13-NoPE) makes the split between the
two cache layouts a checked identity instead of a convention. Scalar-math
identity **K4** (qk_scale, output_mult, softcap, eps as f64 bits — "caught
2 real GGUF-vs-config regressions") binds the arithmetic itself. And
session artifact **K5** (13 NoPE planes as-is + 39 SWA windowed planes,
fail-closed resume) is the shape a restorable session must present — or
the resume is a miss (`lib.rs:19-26`).

The parenthetical inside K4 is the one not to skim past. Binding the scalar
constants of the math — the attention scale, the output multiplier, the
softcap, the norm epsilon — as raw f64 bit patterns reads as paranoia,
since surely a model file and its config agree about its own constants.
They did not, twice, and the identity caught both before anyone would have
noticed by staring at output quality.

The identity object scopes everything the resident tier serves:

```rust
// crates/muser-kvpack/src/layout.rs:21
pub struct MuseIdentity {
    pub model_sha256: [u8; 32],
    pub adapter_sha256: [u8; 32],
    pub tokenizer_sha256: [u8; 32],
    pub chat_template_sha256: [u8; 32],
    pub context_policy_sha256: [u8; 32],
    pub model_revision: String,
    pub tokenizer_revision: String,
    pub weight_precision: String,
}
```

Its `digest()` hashes a domain tag, the five 32-byte fields
length-prefixed, and the three strings — "One digest covering every
identity field. The resident radix scopes its keys by this digest, so an
entry written under another identity — or under none — is structurally
unreachable from a lookup, never a best-effort hit" (`layout.rs:32-38`;
the test at `:182-206` shows a one-bit model change, a precision swap, and
even a field-boundary shift each producing a different digest). The
descriptor builder then derives the qualified layout and *validates the
geometry against the live config* before any state is described:

```rust
// crates/muser-kvpack/src/layout.rs:138
fn validate_geometry(cfg: &MuseConfig, cached: u32) -> Result<(), LayoutError> {
    if cfg.n_layers != MUSE_LAYER_COUNT {
        return Err(LayoutError::Geometry("layer count"));
    }
    if cfg.n_kv_heads != MUSE_KV_HEAD_COUNT || cfg.head_dim != MUSE_HEAD_DIM {
        return Err(LayoutError::Geometry("KV head geometry"));
    }
    if cfg.sliding_window != MUSE_SWA_WINDOW {
        return Err(LayoutError::Geometry("SWA window"));
    }
    if cfg.context_length != MUSE_MAX_CONTEXT {
        return Err(LayoutError::Geometry("maximum context"));
    }
    // … (cached-token bound and the 39-SWA/13-NoPE partition check,
    //     :151-161 — any layer violating `is_swa() == (layer % 4 != 3)`
    //     refuses with "39-SWA/13-NoPE partition") …
}
```

Duplication is normally a smell, so it is worth saying why this one is
deliberate. The NoPE layer table exists in three places — engine
config, the Mac sink, the transfer schedule — because those three live in
different processes, in different languages, on different machines, and no
single one of them can be made authoritative without a network round trip
in the middle of a hot path. The reason it matters for the handoff is that
a silent disagreement between the copies would not look like an error: it
would produce a structurally perfect transfer of the wrong planes. So the
copies are checked against each other instead of trusted. This cross-check
(plus a fail-closed GGUF cross-check, `layout.rs:154-161` per the merge
map) is what keeps them honest `[docs/kvpack-merge-handoff §4]`.

`descriptor` also binds the K4 scalar math (`qk_scale_factor_bits`,
`output_multiplier_bits`, `final_logit_softcapping_bits`,
`post_norm_eps_bits` as f64 bit patterns, `layout.rs:96-104`) and appends
the exact-logits state — a full vocab-width f32 plane at the synthetic
layer 52 (`MUSE_EXACT_LOGITS_LAYER`, `layout.rs:107-125`) — which is what
lets [Ch 25](25-warm-reuse.md)'s warm hit resume generation, not just
attention.

Durable sessions go through `session.rs`: `save` exports the interchange
snapshot (refusing any cut without final logits, `session.rs:150-153`),
`save_snapshot` writes every plane under its descriptor key and the
logits plane last, and lookup is deepest-prefix by construction —

```rust
// crates/muser-kvpack/src/session.rs:222
pub fn find_deepest(&self, tokens: &[u32]) -> Result<Option<DurableHit>, SessionCacheError> {
    // … (derive the requested descriptor; resolve the 256-token-block
    //     prefix chain; `resolve_prefix` returns the deepest committed
    //     cut, :228-243) …
}
```

— and the whole ladder is ordered by one module line: "Ordered exact-prefix
reuse: current session, resident, durable, then remote"
(`crates/muser-kvpack/src/reuse.rs:1`). The resident tier is a
content-interned, identity-scoped token radix (`resident.rs`); the durable
tier caps it — "the durable tier is the sole authentication authority …
an unauthenticated resident entry can never serve deeper than the
authenticated chain" (`reuse.rs:322-328`). [Ch 25](25-warm-reuse.md) walks
the ladder end to end.

## 24.6 The sealed manifest in the cluster: identities bound at enrollment

Who does the receiver think it is talking to, and when was that decided?
The answer is the organizing idea of this section: it was decided at
enrollment, before any transfer existed, and it is recorded on disk rather
than negotiated on the wire. A protocol that negotiates identity can be
talked out of it; a protocol that reads identity from a file it was
configured with cannot.

On the disaggregated lane, the same sealed-manifest discipline is wired
into muser-cluster's receiver configuration, and the architecture doc
states the security model in one paragraph: "GX10 Handoff V2 uses mutually
authenticated TLS plus an HMAC-sealed manifest. Enrollment generates each
TLS private key on the machine where it remains; the HMAC is a shared
secret transferred over known-host-verified SSH. Replay admission durably
reserves the generation with file and directory fsync before
target+DFlash publication and ACK. Any durability failure degrades the
route until repair and restart" `[docs/muser-architecture.md §Durable and
remote KV]`.

The receiver's config is the enrollment artifact made concrete — every
identity the seal will be checked against, held by path, never inlined:

```rust
// crates/muser-cluster/src/config.rs:20
pub struct ReceiverConfigV2 {
    pub listen: SocketAddr,
    pub certificate_chain: PathBuf,     // mTLS: this node's chain
    pub private_key: PathBuf,
    pub peer_ca: PathBuf,
    pub peer_leaf_sha256: BTreeSet<String>,  // leaf-pin set for the producer
    pub hmac_key_file: PathBuf,         // the shared seal key, by path
    pub hmac_key_id: String,
    pub minimum_hmac_epoch: u64,        // replay floor
    pub replay_ledger: PathBuf,
    // … (timeouts, producer control, producer mode) …
    pub identity: ExactIdentityV1,
    pub target_cache_identity_sha256: String,
    #[serde(default)]
    pub dflash_identity_sha256: Option<String>,
    /// Context shape stamped during enrollment from the digest-verified
    /// DFlash sidecar. It is paired with the component digest so a receiver
    /// can never silently substitute its own window.
    #[serde(default)]
    pub dflash_context_geometry: Option<DFlashContextGeometry>,
}
```

Note what is *not* here: no secret material inline — keys are paths
(secrets hygiene is repo law, `AGENTS.md`). The struct is a list of things
the receiver refuses to be talked into: which producer leaf it will accept,
which seal key, which floor for the epoch, which model identity, which
context window. Each field is a door that enrollment nailed shut.

The wire framing itself is
Muser's reimplementation beside the vendored crate — frames
`Begin/Segment/Seal/Ack/Abort` over the TLS stream with magic
`KVPKV2\0\0` and a 20-byte preamble (`crates/muser-cluster/src/transport.rs:15-16`),
plus one deliberate extension: the delta `prefix_cut` travels *beside* the
typed manifest, lifted from raw JSON at the frame boundary, because the
typed `BeginManifestV2` drops unknown keys (`transport.rs:35-46`) — a
live illustration of §24.2's "two implementations, one seal" reality, and
the hook [Ch 26](26-delta-handoff-and-migration.md) pulls on.

None of that is worth much until somebody has watched it refuse. So we
made it refuse, deliberately, in both directions. We replayed a request one
generation below the watermark, expecting the ledger to notice, and got the
explicit stale/replayed refusal rather than a quiet re-serve of stale
state. Then we handed the receiver a config that was well-formed in every
respect except a flipped adapter digest — the case that is dangerous
precisely because everything else about it looks right — and got an
identity-mismatch error rather than a restore that would have seemed fine
until the output drifted.

Both runs are retained, and the receipts discipline around them is stricter
than it first sounds: "receipts bind every attempt — command, exit status,
retained log — including the
refused ones. A producer timeout is recorded as *invalid evidence*, never
counted as a refusal" `[docs/kvpack.md §Proven live — refusal receipts,
not claims]`. That last clause is the one to hold on to. A refusal only
counts as evidence when the thing that refused was healthy enough to have
said yes.

## 24.7 The receiver refuses slow volumes before any transfer

Before a receiver even binds its listener, it probes the volume its replay
ledger will live on — and refuses to serve from a slow one. That sentence
contains two terms this book has not yet defined, and the refusal only
makes sense once both are in hand.

The **replay ledger** is the receiver's durable record of the highest
generation ever committed per HMAC key: the watermark that makes an old,
validly-signed handoff refuse as a replay; [Ch 30 §30.6](30-handoff-v2-transport.md)
tells its full story. **fsync** is the operating-system call that forces
buffered writes out of volatile cache onto the storage device itself — it
is what turns "we wrote the watermark" into "the watermark survives a power
cut", and [Ch 30 §30.6](30-handoff-v2-transport.md) walks why the
*directory* variant is the load-bearing one here.

Put those together and the constraint appears: the watermark must be
durable *before* the ACK goes out, so the speed of a disk sits directly in
the latency path of every transfer. The code carries its own incident
report about what that cost us:

```rust
// crates/muser-cluster/src/receiver.rs:108
/// The commit path durably reserves every generation with
/// write+fsync+rename+directory-fsync before the ACK leaves. On a volume with
/// a slow directory-fsync tail this stalls the ACK and first decode by
/// hundreds of milliseconds at random (the 2026-08-18 p4 seal stall), so a
/// receiver whose replay ledger sits on such a volume is refused at bind
/// time. `scripts/gx10/durable_fsync_probe.py` is the standalone operator
/// check for the same pattern.
const LEDGER_RESERVE_PROBE_ITERATIONS: usize = 20;
const LEDGER_RESERVE_PROBE_MAX_TAIL: Duration = Duration::from_millis(100);

fn check_ledger_volume(ledger: &Path) -> Result<(), String> {
```

`probe_ledger_reserve` (`receiver.rs:150-178`) runs the exact reserve
pattern twenty times — create temp file, write 4 KiB, `sync_all`, rename,
directory `sync_all` — and `check_ledger_volume` refuses the bind if the
worst sample exceeds 100 ms, with an error that tells the operator what to
do: "point replay_ledger at the internal disk (see
scripts/gx10/durable_fsync_probe.py)" (`receiver.rs:139-146`).

The backstory is worth telling in the order we lived it, because we spent a
while chasing the wrong suspect. What showed up on 2026-08-18 was a bimodal
stall in the commit paths: most transfers were fine, and then one would sit
for ~1 s for no reason visible anywhere in the transfer itself. A random
stall on a network path looks like a network problem, and that is where we
looked first. It was not the network. Root cause was the directory-fsync
tail of the *evidence* volume — the volume we had, quite reasonably,
pointed the replay ledger at, because that is where receipts go and the
ledger felt like a receipt.

The lesson went straight into the repo's working agreements, and it is a
placement rule rather than a tuning knob: operational state (replay
ledger, sockets, locks) belongs on the internal disk, evidence on the
append-only volume `[AGENTS.md; ledger Arc 2]`. The bind-time gate is that
lesson turned into a fail-closed preflight — the receiver refuses *before*
any transfer rather than stalling *during* one, which is the difference
between an operator reading an error message that names the fix and an
operator staring at an intermittent mystery.

One separation is worth making explicit, because the two failures are easy
to blur. This fsync tail is not the other deep-payload stall of the same
campaign (EEE link-idle retransmission blackouts,
[Ch 31](31-the-wire-discipline.md));
both punished the deep burst schedule, by different mechanisms, and both
got operationalized — one as this bind-time probe, one as the EEE-off link
invariant.

## 24.8 Tradeoffs

Each of the decisions above had a cheaper alternative, and in most cases
the cheaper alternative is the one you would reach for first. This section
is the accounting: what we gave up, and what the measurements say we bought
with it.

**One format authority, twice implemented, verified by a seal.** The
obvious architecture is a shared crate on both ends; Muser deliberately
does not have one (the producer reimplements the format in C++/Python,
`lib.rs:8-14`). The measured consequence of divergence is not corruption
but refusal: the sorted-keys incident of §24.2 dropped live transfers on
`descriptor_sha256` mismatch — the seal doing its job — and produced a
patch with a regression test. A shared crate would have made both ends
silently consistent *and* silently coupled to one implementation's bugs;
cross-verification makes disagreement loud. The cost is real and known:
format fixes must land twice (F1 in the merge audit tracks that `prefix_cut`
still has no home in the typed protocol, so "delta handoff … has a
Python-only producer" `[docs/kvpack-merge-handoff §3 F1]`).

**Fail-closed identity vs best-effort reuse.** Every identity dimension of
§24.4–§24.5 could, in principle, be advisory — restore anyway, let quality
absorb it. The format refuses, and the campaign's receipts show what the
refusal buys: bit-identical warm hits at depth
([Ch 25](25-warm-reuse.md)) are only claimable *because* a wrong
tokenizer, template, layout, or scalar-math constant is a miss, not a
restore. The cost is the opposite of convenience — "a different model,
tokenizer, template, quantization, layout, or engine ABI is a miss, not a
best-effort restore" `[third_party/kvpack/README.md §What kvpack provides]`
means upgrading any of them orphans every pack, by design.

**Integrity hashes vs authenticated sealing.** Unencrypted pack hashes
detect accidents cheaply and nothing more; the honest line is written down
("pack hashes prove integrity, not authenticity" `[docs/kvpack.md]`). The
defense-in-depth ordering — `kvenc` envelope at rest, mTLS + HMAC seal in
transit — lets each layer carry only the threat it can actually prove.
Skipping the distinction would be the security version of the "~7 GB"
payload mistake: reading an allocation as if it were traffic.

**Vendoring vs a live dependency.** The vendored, per-file-hashed snapshot
trades upstream motion for reviewability: every byte the engine relies on
is in-tree, auditable by one script, and — the merge ruling's direction —
canonical for upstream `[docs/kvpack-merge-handoff §1]`. The alternative, a
git dependency, would re-introduce exactly the silent-drift class the
canonical-JSON patch closed.

**Where the gap lives.** None of this is on the Metal decode graph; kvpack
is not the dispatch gap. Its costs live elsewhere and are measured
elsewhere: receiver-side verify/install/seal/commit phases (~0.2 s constant
in the N2 diagnostic `[ledger N2]`), and the wire amortization schedule of
[Ch 30](30-handoff-v2-transport.md).

## 24.9 What comes next

The format is now fully assembled: a provenance-pinned vendored tree, a
crash-safe layered-integrity container, a component × segment wire protocol
with a mandatory keyed seal, a Muse adapter that scopes every entry by a
digest over eight identity dimensions, and a receiver that refuses slow
volumes before accepting a single byte. But a format is only as valuable as
the hits it serves. The whole apparatus — identities, seals, ledgers —
exists so that one question can be answered *fast and provably*: "have I
already computed this prefix?" What a hit is actually worth, at two very
different depths, with controls that prove it is reuse and not
cache-forever, is [Ch 25](25-warm-reuse.md).

## References

- `[docs/kvpack.md]` — the stance ("exactness is the product"), the three
  questions, the security model in full, refusal receipts, the honest
  threat-model line.
- `third_party/kvpack/README.md` — replay semantics (quoted), the
  integration boundary, layered integrity, concurrency model, conformance.
- `third_party/kvpack/provenance.json` — schema, upstream commit/tag/tree,
  per-file SHA-256 map, the two recorded patches (canonical-JSON
  sorted-keys quoted abridged).
- `third_party/kvpack/crates/kvpack-handoff/src/handoff_v2.rs:28-91` —
  `HmacIdentityV2`, `SegmentRoleV2`, `SegmentDescriptorV2`,
  `BeginManifestV2` (quoted); `:100-160` begin validation; `:233-246`
  `SealCoreV2` / `SealManifestV2` (quoted); `:248-287` signing;
  `:307-314` the `HandoffSinkV2` shadow contract.
- `third_party/kvpack/crates/kvpack-handoff/src/manifest.rs:56-66` —
  `ExactIdentityV1` (quoted); `:85-129` `LayoutClassV2` (quoted) with the
  windowed-class rule; `:347-368` `LayerHeaderV1`; `:397-410` the
  pre-RoPE canary contract.
- `crates/muser-kvpack/src/lib.rs:3-32` — the one-shared-dependency ruling
  (quoted), K1/K3/K4/K5, the adapter's three additions.
- `crates/muser-kvpack/src/layout.rs:20-58` — `MuseIdentity` and its
  digest (quoted); `:69-136` the descriptor builder incl. K4 binding and
  the exact-logits plane; `:138-163` `validate_geometry` (quoted).
- `crates/muser-kvpack/src/session.rs:144-244` — durable save/find_deepest;
  `crates/muser-kvpack/src/reuse.rs:1-7, 322-328` — ladder order and the
  durable-caps-resident rule.
- `crates/muser-cluster/src/config.rs:20-50` — `ReceiverConfigV2`
  (quoted): pins, key paths, replay floor, identity fields.
- `crates/muser-cluster/src/transport.rs:15-16, 35-46` — `KVPKV2` framing;
  the raw-JSON `prefix_cut` lift.
- `crates/muser-cluster/src/receiver.rs:108-178` — the slow-volume
  refusal (comment and thresholds quoted), `probe_ledger_reserve`.
- `[docs/kvpack-merge-handoff-20260820]` — §1 the merge ruling, §3 F1
  (`prefix_cut` has no typed home), §4 the muser cache/reuse map, §5 pack
  internals, §6 the handoff architecture and pacing reality.
- `[docs/muser-architecture.md §Durable and remote KV]` — enrollment key
  hygiene, durable replay reservation (quoted in §24.6).
- `[AGENTS.md]` — the 2026-08-18 durability lesson (operational state on
  the internal disk) and `scripts/gx10/durable_fsync_probe.py`.
- `[ledger N2]` — receiver phases constant ~0.2 s (the non-wire cost).
- `[receipt phase4-disagg-20260820/130815-g900091/]` — the 1,823,184,896 B
  deep payload (§24.1's scale anchor).
- [Ch 22](22-the-price-of-context.md), [Ch 23](23-the-swa-ring-and-the-growing-cache.md)
  — the byte economics and the interchange snapshot this format carries.
- [Ch 25](25-warm-reuse.md), [Ch 26](26-delta-handoff-and-migration.md),
  [Ch 30](30-handoff-v2-transport.md), [Ch 31](31-the-wire-discipline.md)
  — the hits, the deltas, the transport, and the wire discipline this
  chapter foreshadows.
