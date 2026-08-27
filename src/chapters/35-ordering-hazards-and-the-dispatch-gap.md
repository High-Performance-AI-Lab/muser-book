# Chapter 35 — Ordering, hazards, and the dispatch gap
> **status:** polished  ·  **path:** Muse Glimmer, pinned Muser tree

*Prerequisites: [Ch 2](02-metal-compute-model.md) (command buffers,
encoders, dispatch — this chapter uses "closure," "encoder," and "commit"
fluently), [Ch 3](03-unified-memory-and-buffers.md) (unified memory and the
buffer substrate), [Ch 10](10-the-forward-pass-at-a-glance.md) (the one-token
graph whose dispatches we will count), [Ch 34](34-scheduler-and-slots.md)
(who submits). No GPU-synchronization background is assumed; the hazard
taxonomy is built from zero.*

---

[Ch 34](34-scheduler-and-slots.md) ended with a submission: the batcher packs
rows, a permit is acquired, one concurrent encoder records a whole token, one
commit, one wait. That chapter treated the recorded tape as if pressing
"record" and "play" were all there was to correctness. They are not. The
moment more than one kernel writes GPU memory inside one command buffer — and
Muser's token graph runs hundreds of dispatch groups — you must answer a
question the hardware will not answer for you on every path: *when do this
kernel's writes become visible to that kernel?* Get it wrong in one direction
and you read stale bytes; get it wrong in the other and you stall the GPU for
no reason.

This chapter does three things. It teaches the hazard taxonomy — RAW, WAW,
WAR — with timelines small enough to check by eye. It inventories Muser's
*actual* ordering tools, verified in source: one command buffer per token,
tracked buffers by default, two barrier forms, a serial/concurrent encoder
switch, and queue-ordered submissions instead of fences. Then it spends that
vocabulary on the campaign's most instructive measurement: the bounded
one-token diagnosis that reconciled a +196 dispatch-closure gap into four
named families and rejected the largest of them for changing logprobs beyond
contract. The recurring question of this book — what may be moved without
breaking the exactness contract? — has never had a sharper answer than
"not these 104 groups."

---

## 35.1 What one queue buys — and what it does not

Recall the Metal execution model of [Ch 2](02-metal-compute-model.md). You
record kernel dispatches onto a `MTLComputeCommandEncoder`, end the encoding,
commit the [command buffer](../glossary.md#command-buffer), and the queue
executes it. Two guarantees come free:

1. **Within one serial encoder, dispatches run in recorded order.** Whatever
   ordering problems exist, they are not *scheduling* problems: kernel B
   recorded after kernel A does not start before A on a serial encoder.
2. **Command buffers on one queue complete in commit order.** The engine
   leans on this for its teacher-forced lane — the comment at the submission
   site: "The queue serializes GPU work so token i+1 cannot race token i's
   residual/KV, while the host encodes i+1 during i's GPU interval"
   [crates/muser-engine/src/decode.rs:2151-2155].

So where is the problem? In *memory visibility*. A dispatch "completing"
does not mean its writes are visible to the next dispatch — caches, and the
freedom a **concurrent** encoder grants its dispatch closures to overlap, sit
between the two. [Ch 10](10-the-forward-pass-at-a-glance.md) already showed
the serving token graph uses one *concurrent* encoder with explicit barriers
precisely because the four independent Q/K/V/gate matvecs are allowed to
overlap [crates/muser-engine/src/decode.rs:5448-5458]. Ordering within
overlap is the subject of this chapter. And even on a serial encoder, the
fine print matters: Metal orders *dispatch execution*, but Muser's own
diagnostic notes teach that relying on implicit visibility instead of saying
what you mean is how [Ch 34](34-scheduler-and-slots.md)'s clean design
acquires invisible state.

## 35.2 The three hazards, from zero

A **hazard** is a pair of memory operations whose *order* changes the result.
Three kinds can hurt you; classify every buffer relationship into exactly one
of four buckets (Figure 35.1). Two kernels A (first) and B (second), one
buffer X:

```
  RAW — Read After Write   (true dependency: B reads what A wrote)

     kernel A          ████ write X
     kernel B                              ▒▒▒▒ read X   ← must see A's bytes
                       ────────── time ──────────→
     Rule: A's writes must be visible to B. THE decode hazard: the token
     graph is a producer→consumer chain (norm → matvec → gate → …).

  WAW — Write After Write  (order of writers decides final bytes)

     kernel A          ████ write X = 1
     kernel B                    ████ write X = 2   ← final value must be 2
                       ────────── time ──────────→
     Rule: writers serialize. Rare on the token path (buffers have one
     writer per dispatch group) but lives in workspaces reused across
     dispatch groups and in-place kernels.

  WAR — Write After Read  (B overwrites bytes A still needs)

     kernel A          ▒▒▒▒ read X (old)
     kernel B                    ████ write X (new) ← must wait for A's read
                       ────────── time ──────────→
     Rule: B waits. Appears when a ring buffer's producer laps a slow
     consumer — Ch 15's ring rotations are the in-engine near-miss.

  RAR — Read After Read   (both readers: order irrelevant)
     NOT A HAZARD. No barrier, ever.
```

*Figure 35.1: the hazard taxonomy as timelines. ████ = write, ▒▒▒▒ = read. The
rule column is the whole discipline: name the bucket, then place (or omit) a
barrier accordingly.*

Worked micro-example, one buffer `x`, two dispatch closures on a concurrent
encoder. Recorded order A then B, but the encoder may overlap them:

```
  A: x = x + 1        (read-modify-write — contains a RAW against itself)
  B: y = x * 2        (reads x)

  If B's read overlaps A's write:  y may see the old or the new x.
  This is a RAW hazard from A to B → a barrier between the closures is
  REQUIRED. Without one, the graph is correct only by scheduling luck.
```

Now the same pair with `B: x = x - 1` instead: both closures write `x`, a WAW
hazard — the final value of `x` depends on which write lands last, so the
closures must serialize even though neither *reads* the other. And if `A`
reads `x` while `B` writes it, you have WAR — on the decode path, think "the
KV ring's next row is also a row someone is still attending from." [Ch 15](15-kv-store-and-the-ring.md)'s
explicit `origin_logical`/`origin_physical` bookkeeping exists so that the
writer and the readers name *different rows* and the WAR never fires.

One more non-hazard worth naming because the engine exploits it: **reads of
immutable data never need ordering.** All four packed decode rows of
[Ch 34](34-scheduler-and-slots.md) read the same weights concurrently — RAR
against the weight arena, zero barriers, by construction.

## 35.3 Muser's actual ordering tools

The ancestor Ferrite engine solved this problem with a compiled `Vec<Op>`
program and a frozen per-model barrier plan (§35.8). Muser has no such
machinery — there is no `Op` enum, no compiler pass, no plan. The ordering
model is five explicit devices, each verifiable in source.

### Tool 1 — one command buffer per token, one queue

The serving token graph is one command buffer: encoder opened, whole 52-layer
graph recorded, `end_encoding`, `commit`, bounded wait
[crates/muser-engine/src/decode.rs:5448-5460]. The packed group graph of
[Ch 34](34-scheduler-and-slots.md) is the same shape with four rows
[crates/muser-engine/src/decode.rs:4920-4937]. Cross-token ordering is the
queue's commit-order guarantee (`decode.rs:2151-2155`); cross-*graph*
ordering is the scheduler's single-owner rule of [Ch 34 §34.2]. There are no
`MTLFence` or `MTLSharedEvent` objects anywhere in the engine — a grep for
fence construction across `crates/muser-engine/src/` finds the word only in
a comment [crates/muser-engine/src/metal/buffer.rs:10]. Where a boundary must
be *crossed*, Muser ends the command buffer and waits — and even the wait is
bounded: `wait_for_completion` parks the blocking call on a watcher thread
with a condvar so "a hang becomes a logged `Deadline` error, not a frozen
box" [crates/muser-engine/src/metal/context.rs:153-175].

### Tool 2 — tracked buffers by default

Metal buffers can be created *tracked* (the driver inserts the visibility
synchronization your dispatches imply) or *untracked* (it does not; you owe
every dependency an explicit fence or barrier — see [Metal-PG]). Muser's
allocation path routes every buffer through one function:

```rust
// crates/muser-engine/src/metal/buffer.rs:7-14
fn shared_tracked() -> MTLResourceOptions {
    // Several accepted Muse paths still cross compute encoders (notably
    // target-hidden prefill/capture). Untracked resources are only valid when
    // every such dependency has an explicit fence/barrier. b9678d4 enabled
    // untracked mode globally before that contract existed and empirically
    // changed DFlash conditioning while leaving final greedy IDs unchanged.
    MTLResourceOptions::StorageModeShared
}
```

That comment is a small incident report. An earlier revision (`b9678d4`)
flipped the engine to untracked globally, before any explicit-dependency
contract existed; the final greedy token IDs stayed identical, but the DFlash
draft's *conditioning* — the hidden states it reads — changed. Something as
distant as hazard-tracking mode perturbed numerics through scheduling
overlaps, and that is disqualifying in an engine whose contract is
reproducible bits. The default since is tracked. One documented exception:
multi-gigabyte KV planes may allocate `uninitialized` — and that is an
initialization contract, not a tracking one; debug builds *poison* the bytes
(`0xDEAD`) so a premature read is conspicuous, and the ring metadata
guarantees every read row was written first
[crates/muser-engine/src/metal/buffer.rs:246-277].

### Tool 3 — two barrier forms

With tracking on, why barriers at all? Because the concurrent encoder still
must be told where dispatch *closures* depend on each other, and because the
engine wants dependencies to be auditable in source rather than implied.
Form one, broad scope, is inserted automatically between closures:

```rust
// crates/muser-engine/src/decode.rs:6256-6266
impl EncodeTarget for GraphEncoder<'_> {
    fn before_dispatch(&self) {
        if self.concurrent && self.has_dispatch.replace(true) {
            // Broad buffer scope exactly matches llama.cpp's dependency reset.
            // Independent kernels are deliberately grouped into one dispatch
            // closure, so every closure boundary is a real graph dependency.
            unsafe {
                let _: () = objc::msg_send![self.encoder, memoryBarrierWithScope: 1u64];
            }
        }
    }
```

Every `dispatch(command, |encoder| { … })` call routes through
`before_dispatch` [crates/muser-engine/src/decode.rs:6273-6279]: on a
concurrent encoder, each closure after the first is preceded by a
whole-buffer-scope memory barrier — deliberately the same reset llama.cpp
performs between its own kernel groups [crates/muser-engine/src/decode.rs:6259].
The cost of the broad form is that it also orders *unrelated* reads; which is
why form two exists:

Form two is the **targeted resource barrier**, naming exactly the buffers a
dependency flows through. The clearest example is on the decode attention
route — KV store then attention, the RAW hazard made visible as two lines:

```rust
// crates/muser-engine/src/decode.rs:5660-5671
dispatch(command, |encoder| {
    self.kernels.encode_kv_store_f16(
        encoder,
        &self.activations.k,
        &self.activations.v,
        &plane.key,
        &plane.value,
        write_physical,
    );
    let kv: [&metal::ResourceRef; 2] = [plane.key.metal(), plane.value.metal()];
    encoder.memory_barrier_with_resources(&kv);
    self.kernels.encode_llama_flash_attn_decode_vec_f16(
```

Store writes K/V; barrier names K/V only; attention reads them. The scoped
form has its own didactic comment on the splitk route, where the producer's
*partials* scratch (not the whole world) must reach the reducer: "Scope the
dependency to that allocation instead of stalling every buffer used by the
52-layer command buffer" [crates/muser-engine/src/metal/encode/attn.rs:771-775].
Targeted barriers appear at each genuine RAW boundary — mask-before-attention
and pad-before-vec on the llama routes, staged-shadow-before-attention on the
SWA route, partials-before-reducer on the split routes
[attn.rs:247, 256, 293, 316, 428, 558, 614, 632, 747, 783; decode.rs:4305].

### Tool 4 — serial versus concurrent encoder, as a flag

The prefill encoder can run either way, and the choice is an environment
variable with the A/B history in a comment:

```rust
// crates/muser-engine/src/decode.rs:975-979
    // Prefill-only concurrent Q/K/V/gate and FFN gate+up. Decode already
    // groups those projections; serial prefill paid a launch tax on PP128.
    // `MUSER_SERIAL_PREFILL_DISPATCH` restores the previous encoder for A/B.
    concurrent_prefill_dispatch: bool,
```

Default: concurrent grouping of the independent projection GEMMs (the
`MUSER_SERIAL_PREFILL_DISPATCH` flag at `decode.rs:1331-1333` restores the
serial encoder for exact A/B comparisons). The same select-by-graph structure
appears in `new_prefill_graph_encoder`, which also documents the command
buffer's reference contract — unretained references, "This matches pinned
llama.cpp's commandBufferWithUnretainedReferences contract"
[crates/muser-engine/src/decode.rs:6102-6122].

### Tool 5 — discipline as documentation

Count the tools and you notice what is missing: no barrier planner, no
analysis pass, no proof type. The contract is maintained the way [Ch 34](34-scheduler-and-slots.md)'s
isolation contract is — in the structure of the code and its comments.
Independent kernels are *grouped into one closure* so that "every closure
boundary is a real graph dependency" (`decode.rs:6260-6261`); the barriers
you find are therefore exactly the hazard statements, no more. When a new
hazard class appeared — the SWA staging shadow needing its staged bytes
visible to the vec kernel — the fix was three named resources and a comment
[crates/muser-engine/src/decode.rs:4300-4305].

That is the whole ordering model. Now spend it on the measurement.

## 35.4 The instrument, corrected first

On 2026-08-15 the campaign asked: where does the one-token decode deficit
live? The instrument was `MUSER_METAL_PHASE_PROFILE` and its `PhaseProfiler`
— and the first finding of the investigation was that the instrument itself
was wrong in two ways, which the note corrects before presenting any number
[docs/decode-dispatch-gap-20260815.md].

**What a "closure" is.** The profiler counts calls to the Rust `dispatch`
closure, each submitted as its own command buffer with a synchronous wait —
it does *not* count raw `dispatch_thread_groups` calls. One `qkvg` closure
contains four kernel dispatches but contributes one count. Production serving
encodes the graph into one shared encoder with a single wait, so
"diagnostic wall time minus reported GPU time" mostly measures the
diagnostic's own hundreds of submit/wait cycles. Any statement like "the gap
is N dispatches of overhead" is therefore a category error; the honest unit
is *profiling closures*.

**Two label defects had shifted timings.** Production labels omitted
`lm_head` (its time silently attributed to the following `softcap` label),
and the legacy schedule declared a separate SWA `kv_store` in all 39 SWA
layers although the implementation at position 2,048 combines KV publication
and attention in one closure — 603 labels printed for 564 samples, "every
legacy per-label timing after the first extra label was shifted." The
profiler was fixed to derive labels from post-append ring state and to abort
on label/sample count mismatch [docs/decode-dispatch-gap-20260815.md].

This is the book's measurement culture in miniature, and it recurs in
[Ch 38](38-measuring-against-llama-cpp.md): before a number can mean
anything, the instrument's own error model must be on the table.

## 35.5 The +196 reconciliation

With a corrected instrument, the bounded one-token diagnostic (pinned
16,756,681,056-byte target, 2,048-token fixture, one teacher token) counted
**760 profiling closures** for the production graph against **564** for the
legacy schedule — a difference of **+196** — and then reconciled the
difference *exactly* into four families
[docs/decode-dispatch-gap-20260815.md §Corrected closure-count diff]:

| Family | Δ closures | What it is | Disposition |
|---|---:|---|---|
| Norm-boundary groups | **+104** | 51 entry/attn-norm boundaries + 52 post-attn/FFN-norm pairs + 1 post-FFN/output boundary, separated instead of fused | Existing fusion not exact; **reject** |
| SWA wrapped-ring staging | **+39** | one per SWA layer after ring wrap: stage old rows into the shadow ([Ch 36](36-prefill-vs-decode-paths.md)) | Keep until a bit-exact ring-aware replacement exists |
| KV-publication/attention splits | **+52** | store dispatch and attention dispatch as separate closures, once per layer | Session/publication structure; keep — combining closures alone removes no kernel math |
| Last-row copy | **+1** | one bookkeeping copy of the final hidden row | No math; **removed** |
| **Total** | **+196** | | |

*Table 35.1: the +196-closure reconciliation at position 2,048
[docs/decode-dispatch-gap-20260815.md]. Closures are Rust profiling closures,
not raw Metal dispatches (§35.4).*

Two structural reads of this table matter more than the arithmetic. First,
the 52 splits and 39 staging groups are the *price of the ordering and
exactness disciplines you just learned*: the store-then-barrier-then-attend
sequence of §35.3 is what the 52 splits *are*, and the staging groups are
[Ch 16](16-attention-decode-kernels.md)'s decision to reproduce llama's
reduction lanes rather than read a wrapped ring "mathematically equivalently."
Second — and the note says this in bold — "No repeated closure performing
provably identical arithmetic was found" [docs/decode-dispatch-gap-20260815.md].
The gap is not waste shaped like duplicated work. It is boundaries, each of
which exists for a reason.

## 35.6 What was landed, what was rejected

The same note then ran the reduction candidates, all on the same fixture
(pinned target, 2,048-token fixture, one teacher token, bounded phase
diagnostic) [docs/decode-dispatch-gap-20260815.md §Landed and rejected
reductions]:

| Step | Groups | GPU ms | Full-logit SHA-256 changed? | Verdict |
|---|---:|---:|---|---|
| Baseline | 760 | 40.330 | — | historical pre-J0 reference |
| Direct one-row LM-head input (copy elision) | 759 | 40.194 | no | **landed** |
| Existing dual-norm fusion | 655 | 40.614 | **yes** | rejected: not exact |
| Pinned-reduction dual norm | 655 | 39.274 | no | historical only |
| Eight-head one-query GQA FA2 | 655 | 37.097 | no | historical only |

*Table 35.2: the reduction table. GPU times are single-run diagnostics; the
SHA-256 is over the full logit row — the exactness gate
[docs/decode-dispatch-gap-20260815.md].*

**The one landed change** is the +1 family: remove the last-row copy. It is
bit-exact by construction (no arithmetic touched — one 6,656-element f32
copy eliminated), its single-run GPU delta is **−0.136 ms (−0.34 %)**, and
its wall sample went *up* 4.380 ms — diagnostic submit/wait noise, no wall
claim made [docs/decode-dispatch-gap-20260815.md]. A lesson in miniature:
the only guaranteed-free lunch was a bookkeeping copy, and it was worth a
third of a percent.

**The 104-group fusion is rejected on exactness, not on speed.** The
available dual-norm fusion collapses the separated norm boundaries into
655 groups — and changes the logit bytes (Table 35.2, row 3). A corrected
fusion that replaces the four-SIMD-group `rsqrt` reduction with the pinned
llama 32-SIMD-group reduction *twice*, preserving the intervening f32
device-memory publication and reread, keeps the bits (row 4) — and its
five-sample streamed serving result was 26.714 tok/s median (CV 0.079 %)
versus llama's 33.428: "the exact reduction is useful but does not by itself
close Stage A" [docs/decode-dispatch-gap-20260815.md]. The one-query GQA
specialization measured 28.290 tok/s median (CV 0.193 %), ratio **0.8463×**,
prefill 1.0366× — Stage A still 13.37 points below its 98 % decode bar at
that moment [docs/decode-dispatch-gap-20260815.md]. (How Stage A eventually
closed — by changing the anchor itself, J0/J1 — is
[Ch 38](38-measuring-against-llama-cpp.md)'s story; the gap families
survived it.)

**The hybrid postmortem.** The earlier, more aggressive attempt reused the
legacy retained-activation schedule and selected fast fused boundaries. It
preserved the greedy token and failed the public numerical contract, with
every number worth quoting [docs/decode-dispatch-gap-20260815.md §Rejected
hybrid postmortem]:

- full-logit maximum absolute error: **4.6300888e-4**; mean absolute error
  1.5170774150033564e-4;
- normalized [logprob](../glossary.md#logprob) maximum absolute error:
  **3.197146176834309e-4 — above the 1e-4 contract**;
- **201,970 of 202,048 logits differed**;
- the first KV difference: **layer 1, value plane element 524,115, f16 bits
  39,892 versus 39,893** — a single [ULP](../glossary.md#ulp) flip in one
  value element, one layer in, propagating to a hundred thousand logits.

Read that last bullet slowly, because it is this book's precision thesis in
one measurement. A one-ULP difference in one f16 value — the seventeenth-bit
wobble of a rounding-order change in the fused residual/norm chain — is
invisible in the sampled token and fatal to the contract. The attempt "was
removed rather than hidden behind a tolerance or shipped as an alternate
route" [docs/decode-dispatch-gap-20260815.md]. Evidence retained under
`muser-receipt://pinned-token-parity-20260814-v{3,4}/`.

## 35.7 What the gap reveals about what an engine is for

Step back and state the tradeoff plainly. The available 104-group norm fusion
is the single largest closure family. Fusing it changes normalized logprobs
to 3.2e-4 against a 1e-4 contract. Therefore Muser keeps the boundaries.
The ~3 % of GPU time that fusion-class changes might recover
(40.330 → 39.274 ms for the exactable subset, Table 35.2) is not purchasable
at that price — and the campaign's own ranked list records the principle as
its first item: "Implement an exact boundary fusion only if it reproduces the
standalone reduction and store order bit for bit. The current 104-group
fusion is a negative fixture, not a candidate"
[docs/decode-dispatch-gap-20260815.md §Ranked remaining exact work].

This is the recurring question's hardest answer. What does one token cost?
Some of the cost is *not* removable — not because the techniques are unknown,
but because the engine is *for* something: producing tokens whose full
distributions can be diffed against a pinned reference, byte for byte, so
that every other claim in this book (parity, handoffs, warm reuse, sessions)
has a ground truth to stand on. An engine that quietly rounded differently
under load could not host [Ch 32](32-precision-across-the-handoff.md)'s
bounded-drift gates — it would have already spent its credibility on its own
scheduler. Bit-exactness beats throughput, and not sentimentally: it is the
enabling asset for every measured claim Parts IV–VI made.

## 35.8 Ancestor contrast: the compiled barrier plan that was not ported

The ancestor Ferrite engine ordered its decode differently, and the contrast
is instructive enough to box.

> **The Ferrite design (lineage, not Muser).** Ferrite compiled its decode
> graph into a `Vec<Op>` program at load time — routing decisions resolved
> once, then a pipeline of passes fused, reordered, and emitted a frozen
> per-model *barrier plan* of byte-range hazards, replayed per token; a
> sealed `OverlapProof` type made concurrency certification constructible
> only by the analyzer [ferrite-book Ch 21; ferrite-book Ch 22]. It ran its
> buffers *untracked* under that plan, buying a measured −2 to −4 %
> tracking cost back on the ancestor's A18 Pro hardware [ferrite-book Ch 22].
> **Why none of it ported.** The audit that planned this book verified the
> divergence in the Muser tree: there is no `Op` enum and no VM program —
> decode is hand-written encode methods, the shape of Ferrite's *legacy*
> route, chosen for Muser's fixed single-model, pinned-kernel discipline
> where route reasoning does not benefit from a program representation
> (weights and routes are constant after load). And the ancestor's own
> corrections register later established that the comparison motivating its
> untracked mode was false — "llama.cpp uses untracked hazard tracking by
> default" is marked FALSE, the fabricated-claim finding that forced the
> ancestor book's largest correction pass [ferrite-book CORRECTIONS-2026-07
> §3a]. Muser keeps the lesson and not the machinery: hazard-tracking is
> default-on (§35.3, the `b9678d4` reversal), and barriers are hand-placed
> statements rather than compiler output. What survives as genuine lineage
> is the taxonomy (§35.2), the scope-versus-per-resource barrier insight,
> and the measurement instinct that barrier overhead must be counted, never
> assumed [ferrite-book Ch 22].

The gentle summary: Ferrite's design was a bet that a compiler could own
ordering so the runtime could be free with it. Muser's design is a bet that
a small, fixed, audited graph does not need a compiler — only a taxonomy,
five tools, and the discipline to write every dependency down.

## 35.9 Tradeoffs

**Tracked-by-default vs untracked-plus-plan.** Tracking costs driver-side
dependency analysis on every dispatch; the ancestor measured −2 to −4 % for
turning it off `[A18-neo, ferrite-book Ch 22]` — Ferrite-lineage hardware,
never measured on Muser's M3 Ultra [unverified]. Muser pays the tracking
cost because the one attempt at untracked mode "empirically changed DFlash
conditioning while leaving final greedy IDs unchanged"
[crates/muser-engine/src/metal/buffer.rs:10-12] — a silent-numerics
perturbation, the exact failure class this engine exists to exclude.

**The 104 groups: paid in boundaries, refunded in trust.** The separated
norm-boundary closures cost GPU time relative to a fused schedule (Table
35.2: 40.330 vs 39.274 ms for the bit-exact variant on the diagnostic
fixture), and the hybrid that fused them aggressively breached contract at
3.197e-4 normalized-logprob error [docs/decode-dispatch-gap-20260815.md].
The refund is every exactness-gated claim downstream — most concretely, the
parity matrices of [Ch 38](38-measuring-against-llama-cpp.md) that could not
have been run against an engine with drifting boundaries.

**Broad-scope closure barriers vs targeted resource barriers.** The broad
form is simpler and matches llama's own dependency reset (`decode.rs:6259`);
the targeted form exists where a broad stall would serialize unrelated work
— "instead of stalling every buffer used by the 52-layer command buffer"
[crates/muser-engine/src/metal/encode/attn.rs:771-773]. No measurement
separates the two forms' costs [unverified]; the split is driven by the
shape of the dependencies (closure-granular on the token graph,
allocation-granular inside attention producer/reducer pairs).

**One-CB-per-token vs fences.** Ending the command buffer and waiting is
coarser than an `MTLFence` would be — the teacher-forced lane's comment
shows the trade was considered and taken for host-encode overlap instead
(`decode.rs:2151-2155`). The compensating control is the bounded wait
(context.rs:153-175): the coarser primitive cannot wedge a serving thread
indefinitely.

## 35.10 What comes next

Ordering and hazards were the last invisible ingredient in the token's
journey — and the +196 reconciliation named one family, the 39 SWA staging
groups, that belongs to a graph this book has only glimpsed: the *prefill*
route, where whole prompts flow through batch-shaped kernels, wrapped rings
force a staging shadow, and chunk boundaries yield to waiting decoders.
Decode is a matvec story; prefill is a GEMM story; the same weights star in
both. [Ch 36](36-prefill-vs-decode-paths.md) walks the second graph.

## References

- `crates/muser-engine/src/metal/buffer.rs:7-14, 246-277` — `shared_tracked`
  (quoted, with the b9678d4 incident) and the uninitialized-KV contract.
- `crates/muser-engine/src/decode.rs:5448-5460` — one command buffer per
  token, concurrent encoder, explicit barriers.
- `crates/muser-engine/src/decode.rs:6102-6122, 6256-6279` — unretained-
  references prefill command buffer; serial/concurrent `GraphEncoder`; the
  broad-scope closure barrier (quoted); the `dispatch` trampoline.
- `crates/muser-engine/src/decode.rs:975-983, 1331-1333` — encoder-mode and
  fusion defaults with their A/B flags.
- `crates/muser-engine/src/decode.rs:2151-2155, 5660-5671` — queue
  serialization comment; the store→barrier→attend RAW (quoted).
- `crates/muser-engine/src/decode.rs:4297-4305` — the staged-shadow resource
  barrier on the SWA route.
- `crates/muser-engine/src/metal/encode/attn.rs:247, 256, 293, 316, 428,
  558, 614, 632, 745-783` — the targeted-barrier inventory; the scoped-
  dependency comment (quoted).
- `crates/muser-engine/src/metal/context.rs:142-184` — bounded completion
  waits.
- `[docs/decode-dispatch-gap-20260815.md]` — read in full for this chapter:
  the instrumentation correction, the 760/564/+196 reconciliation (Table
  35.1), the landed/rejected reduction table (Table 35.2), the hybrid
  postmortem numbers, and the ranked remaining work.
- `crates/muser-engine/src/decode.rs:6143-6149, 6224-6236` — `PhaseProfiler`
  as the closure-counting instrument behind §35.4.
- `[ferrite-book Ch 21]`, `[ferrite-book Ch 22]`, `[ferrite-book
  CORRECTIONS-2026-07 §3a]` — the ancestor's compiled-program/barrier-plan
  design and its corrections register (lineage/contrast only; A18 Pro
  numbers are ancestor context, never Muser results).
- [Ch 16](16-attention-decode-kernels.md) §16.9 — the staging and split
  families from the attention side; [Ch 38](38-measuring-against-llama-cpp.md)
  — how the anchor change (J0/J1) closed what the fusions could not;
  [Ch 40](40-what-we-measured-and-rejected.md) — the norm-boundary fusion
  as a catalogued rejection.
