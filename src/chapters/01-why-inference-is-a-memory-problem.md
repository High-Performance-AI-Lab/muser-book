# Chapter 1 — The problem: why inference is a memory problem
> **status:** polished  ·  **path:** Muse Glimmer, pinned Muser tree
>
> *Prerequisites: none. This is the first chapter of the book. It assumes you
> have never run a transformer and have never thought about memory bandwidth.*

---

## 1.1 The promise of this book, and the question that runs through it

This book explains, end to end, how the Muser inference engine generates text
from one pinned model — **Muse Glimmer**, 52 layers, roughly 30 billion
parameters — on one Apple Silicon Mac with an **M3 Ultra GPU and 96 GB of
unified memory**. Every line of Metal and Rust quoted in these chapters is
read from the pinned Muser source tree; every number carries a tag that says
where it was measured. When the book and the code disagree, the code wins.

Here is the thesis, in one sentence, before any definitions:

> Generating a single token of text means streaming **sixteen gigabytes** of
> model weights through the GPU, and the GPU finishes the arithmetic long
> before the bytes have finished arriving.

That is the whole book in one line. LLM inference — the act of a model
producing text — looks like a computation problem. It is not. It is a
**bandwidth** problem wearing a computation costume. This chapter derives
that fact on the back of an envelope, from the model's own geometry, so that
by the end you can reproduce every number yourself. We will do the arithmetic
together, in roughly the order we did it ourselves — including the step where
we went looking for a number we assumed existed, did not find it, and had to
change how we asked the question.

And here is the question that recurs in every chapter after this one — the
book's standing question:

> **What does one token cost, where does the time go, and what may be moved —
> into a draft model, into a cache, or across the wire — without breaking the
> exactness contract?**

Every kernel chapter in Part IV returns to that question ("is this kernel
where the time goes?"), every systems chapter in Parts V–VII returns to it
("what may be moved?"), and Part VIII is about the evidence culture that
keeps every answer honest. Keep the question in your pocket. We will collect
answers as we go.

## 1.2 The cast: a few words you need first

Before the arithmetic, eight one-sentence definitions. We are about to count
two quantities and divide them by two speeds, and every word below is
load-bearing in one of those counts. Each is expanded in
the [glossary](../glossary.md) and, where it matters, in a later chapter.

- A **[parameter](../glossary.md#parameter)** is one learned number inside
  the model — one knob tuned during training. A "30B model" holds on the
  order of 30 billion of these numbers.
- **[Weights](../glossary.md#weights)** are the parameters collectively —
  the giant tables of learned numbers stored on disk. "Reading the weights"
  means loading those tables so the math can use them.
- A **[token](../glossary.md#token)** is one unit of text — roughly a piece
  of a word. "The cat sat" is about four tokens. The model emits tokens one
  at a time.
- **[Decode](../glossary.md#decode)** is the act of generating those tokens
  one by one, after the prompt has been read. Reading the prompt is called
  **[prefill](../glossary.md#prefill)**; the two regimes have opposite
  bottlenecks, and the contrast becomes important in
  [Ch 36](36-prefill-vs-decode-paths.md).
- A **[matvec](../glossary.md#matvec)** (matrix × vector) is the one
  operation decode does, over and over: multiply a big table of weights by a
  single vector representing the current token. The hero kernel of this
  book, [Ch 13](13-the-qkv-gate-matvec-family.md), is a matvec.
- A **[FLOP](../glossary.md#flop)** (Floating-point OPeration) is one unit
  of arithmetic — one multiply or one add. A matvec over a matrix with *N*
  elements costs about *2N* FLOPs.
- A **[GGUF](../glossary.md#gguf)** is the on-disk model-file format Muser
  reads: a small header, then the weight tensors packed end to end.
- **[Quantization](../glossary.md#quantization)** is storing each learned
  number in fewer bits than a full 32-bit float — four-ish bits per weight
  instead of sixteen or thirty-two. Part II
  ([Ch 5](05-quantization-from-scratch.md)) is devoted to it; here you only
  need the fact that it shrinks the file.

The argument below is just: *count the FLOPs, count the bytes, divide each
by the machine's speed at that thing, compare the two times.*

## 1.3 One token, costed by hand

This is the arithmetic the rest of the book rests on. We cost out generating
**one token** of Muse Glimmer on the M3 Ultra. No hand-waving: every number
is derived, and the ones that are measurements say so.

### Step 1 — how many parameters?

The label on the box says thirty billion parameters. Labels round, and we are
about to divide bytes by this number, so we would rather read the model's
true shape off the artifact than off the marketing.

Happily, we do not have to take anyone's word for the shape. Muse Glimmer's
geometry is parsed fail-closed from the GGUF at load time, and a test asserts
every parsed field against the pinned release artifact. The table below is
therefore not a transcription from a model card — it is the geometry the
engine refuses to start without `[crates/muser-engine/src/config.rs:169-181]`
`[crates/muser-engine/tests/muse_golden.rs:97-108]`:

| Field | Value | Source |
|---|---:|---|
| layers | 52 | `muse_golden.rs:97` |
| hidden dim | 6,656 | `muse_golden.rs:98` |
| attention heads : KV heads | 32 : 2 (head_dim 128) | `muse_golden.rs:99-101` |
| attention width (32 × 128) | 4,096 | derived |
| KV width (2 × 128) | 256 | derived |
| FFN intermediate dim | 19,968 | `muse_golden.rs:102` |
| vocab size | 202,048 | `muse_golden.rs:103` |
| embedding / LM head | untied (two separate 6,656 × 202,048 tables) | `config.rs:295-297` |

*Table 1.1: Muse Glimmer geometry, read from the pinned GGUF contract.*

Now count parameters from the shapes. The loader asserts a per-tensor shape
contract at `config.rs:294-318`, so the dimensions multiplied out below are
the ones the engine itself checks on the way in: a wrong figure here fails
the load rather than quietly producing a wrong book. Per layer:

```text
attention projections:
  q      6,656 × 4,096 = 27,262,976
  k      6,656 ×   256 =  1,703,936
  v      6,656 ×   256 =  1,703,936
  gate   6,656 × 4,096 = 27,262,976
  o      4,096 × 6,656 = 27,262,976
                              85,196,800
feed-forward (gate, up, down):
  3 × (6,656 × 19,968)  = 3 × 132,907,008 = 398,721,024
norms (4 × 6,656) + per-head QK norms (2 × 128) = 26,880
                                           per layer: 483,944,704
× 52 layers                             = 25,165,124,608
+ token embedding 6,656 × 202,048       =  1,344,831,488
+ LM head         6,656 × 202,048       =  1,344,831,488
+ final norm                             =          6,656
                                          ─────────────
total                                   = 27,854,794,240
```

So Muse Glimmer is **27.85 billion parameters by hand** — the "~30B" on the
label is rounded up. The exact count matters less than the habit: every
bandwidth number in this book divides through a total like this one, and you
should be able to re-derive it.

### Step 2 — how many bytes on disk?

Parameters are not bytes, and it is bytes that the memory system has to move.
So the next question is what all those learned numbers actually weigh.

Full-precision f32 would be 4 bytes each: 27.85e9 × 4 ≈ 111 GB. That does
not fit the plan. The pinned kquant GGUF stores most weights in 4-to-6-bit
blocks instead (Part II explains the formats), and its exact size is
*asserted by the release-gate test*:

```rust
// crates/muser-server/src/chat_template.rs:250
assert_eq!(metadata.len(), 16_756_681_056, "release GGUF byte size");
```

That assertion is not decoration. The size is written down in two independent
places that have to agree, and the second is the engine's own crate doc,
which states it in words: "The pinned target artifact is 16,756,681,056 bytes
on disk." `[crates/muser-engine/src/lib.rs:14]` If a rebuild ever changed the
artifact, the gate would fail loudly rather than let this chapter's
arithmetic quietly drift.

Units, because they will bite you otherwise:

```text
16,756,681,056 bytes
  = 16.757 GB   (decimal, 10⁹ bytes — the convention this book uses)
  = 15.61  GiB  (binary, 2³⁰ bytes)
```

Cross-check against Step 1: 16,756,681,056 × 8 bits ÷ 27,854,794,240
parameters ≈ **4.81 bits per parameter** — quantized blocks plus their
per-block scale bytes plus the two big tables, averaged out. A 30B model at
~4.8 effective bits is what makes a 96 GB Mac a plausible host at all.

### Step 3 — how much do you read to produce ONE token?

This is the insight that surprises people. To generate one token, **every
weight matrix is used exactly once**: the matvec for each projection touches
every row of every weight table a single time. The weights are 16.76 GB and
used once per token, so they cannot live in any cache — caches are
megabytes. They must be streamed, in full, out of main memory, **for every
single token**.

Say it the other way round, because this is the part that trips people up.
The weights are not a dataset the GPU loads once and then keeps close. They
are a river. One token is one complete pass of that river past the ALUs, and
the next token starts the river again from the top, byte for byte the same.

```text
1 token   → read ~16.76 GB of weights, once.
10 tokens → ~167.6 GB.
500-token answer → ~8.4 TB of reads.
```

Not because the model grows — because every token re-reads all of it.

A reader who already knows what a KV cache is will be objecting by now: the
weights are not the only thing decode reads. Quite right, and the reason it
matters here is that our envelope is only honest if the other traffic is
negligible — if it were comparable, every ratio in this chapter would be
wrong. So it is worth spending a paragraph proving that it is not.

The non-weight traffic exists but is small at shallow context. The KV cache
(a per-layer memory of past tokens; [Ch 15](15-kv-store-and-the-ring.md) is
its chapter) costs 1,024 bytes per layer per cached token — the formula
`2 KV heads × 128 × 2 bytes × (K + V)` is derived in
`[docs/memory-footprint.md §KV formula]`. At the benchmark cell we use below
(a ~98-token context), attention reads 52 × 98 × 1,024 B ≈ 5.2 MB per token
— 0.03 % of the weight stream. At the full 131,072-token context it grows
to ~1.83 GB per token `[docs/memory-footprint.md]` — still 9× smaller than
the weights; [Ch 22](22-the-price-of-context.md) costs it out properly. The
objection is real, then, but it does not change the shape of the answer: at
the depths this chapter reasons about, the weight stream is the traffic.

### Step 4 — how much arithmetic is that?

Bytes counted. Now the other side of the comparison, the side everyone
assumes is the expensive one: how much arithmetic does a single token
actually demand?

A matvec `y = W·x` over a matrix with *N* elements costs ~2N FLOPs — one
multiply-add pair per element. Summing only the matrices (norms and the
embedding lookup contribute ~nothing):

```text
matmul parameters = 52 × (85,196,800 + 398,721,024)   = 25,163,726,824
                  + LM head 6,656 × 202,048           =  1,344,831,488
                                                     ─────────────
                                                     26,508,558,312
FLOPs per token ≈ 2 × 26.5e9 ≈ 53.0 GFLOP
```

(The embedding table contributes 0 FLOPs — the lookup is a gather, not a
multiply.) That is the entire compute budget for one token: ~53 GFLOP.

### Step 5 — what does the machine actually do?

We have the bytes and we have the FLOPs. To turn either into a time we need
the machine's own speeds — and this is where the book departs from the
ancestor text it descends from. The fork is worth walking slowly, because
what we chose here governs every bandwidth claim in the rest of the book.

The Ferrite book, written for a small phone-class chip, could quote a
measured DRAM ceiling for its hardware, and we expected to do the same: run a
read microbenchmark, take the ceiling, divide the weight bytes by it, done.
We could not. For this M3 Ultra, no Muser document records a measured
pure-read DRAM ceiling. That left two roads. We could borrow a specification
figure from a datasheet and let the reader assume it was ours — fluent,
authoritative, and unearned. Or we could work the equation backwards from a
number we had genuinely measured, and say plainly that it is a derivation.

We took the second road, and it is the reason the number below carries a
label everywhere it appears: this book **derives the effective read rate from
measured decode throughput** rather than asserting a bus speed it never
observed.

The measurement we do have is the decode throughput itself. The kquant lane's
headline decode number — five repetitions after warmup, synthetic fixture,
F16 KV, 66-token prefix / 32 teacher-forced tokens (teacher-forced: the
harness feeds known prior tokens rather than model-generated ones) — is:

> **35.440 tok/s** (35.439527527, CV 0.037 %) — kquant plain decode
> `[ledger P1.3]`, echoed in `[docs/benchmarks.md §1]`

("CV" is coefficient of variation, std/mean across the repetitions — the
lower, the steadier the cell.)

Two derived numbers fall out, and both are derivations, not measurements:

```text
token time   = 1 / 35.439527527        = 28.22 ms per token

effective read rate
  = 16,756,681,056 bytes × 35.439527527 tokens/s
  = 593.85 GB/s   (decimal; 553 GiB/s)
```

**This 594 GB/s is an effective rate derived from measured throughput** —
the machine's demonstrated average weight-stream rate at this operating
point. It is *not* a specification number and *not* a measured pure-read
ceiling. For orientation only: the campaign ledger refers to the M3 Ultra's
"~800 GB/s" memory class when discussing kernel occupancy
`[ledger L0]` — a class label, again not a measured ceiling. Against that
class, decode runs at ≈ 74 % (593.85 / 800). Where this book needs an honest
bandwidth reference, it uses the derived effective rate and says so.

### Step 6 — compute time vs. memory time

Now the punchline. At the derived effective rate, just *reading* the weights
for one token costs:

```text
16.756681056 GB / 593.85 GB/s = 28.22 ms
```

That is **the entire measured token time** — 28.22 ms, from Step 5. The
weights alone, streamed at the rate the machine actually sustains, consume
the whole token. The 53 GFLOP of arithmetic has no room of its own; it must
hide completely underneath the byte stream (Figure 1.1).

Put in plainer words: the GPU is not slow, and it is not busy. It is waiting.
The whole engine described in this book is an argument about what to do with
a processor that finishes early.

Sanity-check the roofline direction. The workload's
**[arithmetic intensity](../glossary.md#arithmetic-intensity)** — FLOPs per
byte read — is:

```text
53.0e9 FLOP / 16.757e9 bytes ≈ 3.2 FLOP / byte
```

A machine whose memory moves ~800 GB/s `[ledger L0, class reference]` needs
only ≥ 800 × 3.2 ≈ 2.6 TFLOP/s of sustained FP32 to keep its ALUs busy on
this workload. The M3 Ultra GPU's exact FP32 peak is not recorded in any
Muser document [unverified], but any GPU paired with an ~800 GB/s memory
system clears 2.6 TFLOP/s many times over. Decode sits deep on the memory
side of the roofline — and, more directly: the measured token time *equals*
the weight-stream time. There is nothing left over for compute to own.

```text
     DRAM (one unified 96 GB pool)                 GPU (M3 Ultra)
   ┌────────────────────────────────────┐      ┌────────────────────────────┐
   │  mapped GGUF   16.76 GB (kquant)   │      │  ~53 GFLOP of matvec math  │
   │  ════════════════════════════════  │ ───▶ │  hides entirely under the  │
   │  ════════════════════════════════  │ 594  │  byte stream; the token    │
   │  ════════════════════════════════  │ GB/s │  time IS the read time     │
   │  read ALL of it, ONCE per token    │      │  (28.22 ms measured)       │
   └────────────────────────────────────┘      └────────────────────────────┘
     52 layers × {Q,K,V,gate,O, gate,up,down}
     + final norm + LM head (202k-wide)
```
*Figure 1.1: One token = stream ~16.76 GB once. The bytes leave DRAM at the
~594 GB/s effective rate (derived from the measured 35.440 tok/s
`[ledger P1.3]`); the math finishes underneath. The GPU is byte-starved,
not ALU-starved.*

## 1.4 The bandwidth wall

The consequence is brutal and it dictates the entire engine.

If a workload is **compute-bound**, you speed it up with more ALUs, higher
clocks, a bigger GPU. If it is **memory-bound**, none of that helps — the
ALUs you already have are waiting for bytes most of the time; adding more
adds more waiting. Muse Glimmer decode does 3.2 FLOPs per byte read; the
machine's balance point is an order of magnitude higher. It is on the wrong
side of the roofline, and no kernel cleverness moves it to the right side —
the matvec *is* one multiply-add per weight byte; its intensity is fixed by
the model's format.

So only three things can make decode faster:

1. **Read fewer bytes per token** — smaller weights. This is why Part II
   exists: the kquant blocks that get Muse Glimmer to 4.81 effective
   bits/param, and the NVFP4 native lane. It is also where the argument's
   first tempting expectation died. Fewer bits per weight ought to mean
   fewer bytes per token, and fewer bytes per token ought to mean faster
   decode; we ran the two lanes against each other expecting the newer
   format to pull ahead. NVFP4 decode landed at 35.491 tok/s against
   kquant's 35.440 — **parity within noise, never claimed faster**
   `[ledger P1.3]` `[docs/benchmarks.md §1]`. At batch-1 decode both lanes
   read nearly the same bytes at nearly the same rate, so the win we
   actually bought was *capacity*, not decode speed. That distinction — a
   lane that buys you room is not a lane that buys you time — is one this
   book refuses to let slide, in either direction.
2. **Read bytes faster** — a different memory system. Not a lever you have
   on a fixed Mac; the ceiling is the machine's.
3. **Avoid re-reading** — reuse computed results. The KV cache exists so
   attention does not recompute the past every token
   ([Ch 15](15-kv-store-and-the-ring.md)); kvpack (the durable cache format
   of Part V) warm reuse collapses a
   68.6 s cold prefill at 65,536 tokens to a 0.6132 s warm first token,
   bit-identical text — a single-sample cell, not a distribution
   `[claims #11]` `[ledger "Kvpack ladder stage-5 isolated-depth verdict"]`.

Muser's era adds a **fourth lever** the ancestor book did not have:
**move the work somewhere else.**

- Move it *into a draft model*: DFlash speculative decoding drafts cheap
  tokens and verifies a whole window of them in one batched pass against
  the full model — the verify matvec reintroduces weight-reuse into a loop
  that had none. This lever also produced the book's most instructive
  retraction. An early headline put the speedup at 1.3273×; then the
  measurement window itself turned out to be wrong, and requalifying
  against the fixed window brought the honest figure down to 1.23692× vs
  the pinned llama.cpp comparator at 2,048 depth `[claims #15]`. The old
  headline is superseded and must not be cited as a current result, and the
  famous 107.9 tok/s figure survives only as the kquant spec *bar*
  `[ledger "Spec-prefill funded-fix requalification"]`. Both were kept in
  the ledger rather than quietly deleted, which is the habit Part VIII is
  about: a number you have to explain is worth more than a number you have
  to trust.
- Move it *across the wire*: disaggregated prefill runs the
  compute-friendly half of the job on a remote NVIDIA GB10 node and hands
  the KV back — 4.26× faster to first token at 2,048 depth
  `[docs/benchmarks.md §3]` `[ledger "Phase 4 disaggregated GX10→Mac
  context matrix", 5 reps]`.
- Move only *what's new*: delta handoff ships just the suffix — 54.2851 %
  of full bytes with bit-exact output at the 32,768-of-65,536 cell
  `[claims #12]` `[receipt kvpack-ladder-20260820/attempt-10-…-stage6-delta/
  stage6-delta-65536/stage6-verdict.json]`.

Every chapter of this book is about one of these four levers, or about the
evidence culture that decides whether a lever actually worked.

## 1.5 "96 GB is still a budget"

The ancestor Ferrite book opened with a provocation: on its 8 GB phone-class
chip, "8 GB is a lie" — capacity was never the binding constraint,
bandwidth was `[ferrite-book Ch 1]`. On this machine the provocation flips
sign but keeps its shape: **96 GB is real, and it is still a budget** —
capacity answers "does it fit, and what else fits"; bandwidth answers "how
fast". You need both, and the second one governs decode.

What must fit in the 96 GB `[docs/memory-footprint.md]`? The arithmetic,
all of it derivable:

| Resident thing | Size | Notes |
|---|---:|---|
| target GGUF (weights) | 16,756,681,056 B | mmap/page-cache backed |
| KV planes, 4 slots × 131,072 ctx | 7.306 GB | `4 × (39 × 2,048 + 13 × 131,072) × 1,024 B` |
| DFlash draft GGUF | 1,631,205,312 B | loaded only when configured |
| vision projector | 1,400,328,928 B | loaded only when configured |
| f32 batch-activation widths | ~0.99 GB | reused scratch; prefill chunks at 512 |
| macOS + everything else | the rest | not the engine's to spend |

*Table 1.2: The 96 GB budget `[docs/memory-footprint.md]` — with that
document's own caveats: these are on-disk/topology-derived numbers,
"summing artifacts + KV is only a lower bound," and no smaller-memory
configuration may be advertised as supported.*

Two budget facts worth internalizing now. First, the release contract is
**four full-context slots** on this one machine
`[docs/memory-footprint.md §Release requirement]` — KV (7.3 GB at the
ceiling) plus weights (~16.8 GB) plus optional draft/vision (~3 GB) plus
activations is what "four slots" costs; the KV term, not the weights, is
what decides how many slots fit ([Ch 22](22-the-price-of-context.md)
derives this). The engine shares one mapped weight arena across all slots
precisely so the 16+ GiB target is loaded once, not once per slot
(`[crates/muser-engine/src/decode.rs:954-957]`). Second, capacity and
bandwidth fail differently: run out of bandwidth and you get 28 ms tokens;
run out of capacity and you get nothing.

## 1.6 How to read every number in this book

One thing remains before the descent: how to read a number when you meet one.
None of the rules below is an abstract principle. Each was learned by
watching a figure that was perfectly true of a particular run turn into a
claim about the product, and then having to take it back. Stated once here,
obeyed everywhere after:

- **Ratios are `llama ÷ muser`** — above 1.0 means muser is faster
  `[docs/benchmarks.md §Methodology]`. Absolute tok/s drifts with machine
  state; the same-session interleaved ratio is the trustworthy cross-engine
  statistic.
- **Counted cells are five repetitions** after the stated warmup
  convention, means with coefficient of variation; single-sample cells say
  so ([Ch 38](38-measuring-against-llama-cpp.md) covers the protocol).
- **Synthetic vs. natural is load-bearing.** The spec-decode ratios above
  are synthetic-fixture numbers; on natural text, cross-engine outputs
  diverge, and spec decode *loses* on high-acceptance shallow text (rust at
  2,048: 0.931) `[ledger "Spec re-measurement at the fixed window"]`. Never
  let a synthetic number become a workload claim.
- **Never cite** the all-accept 110.59 tok/s control as serving
  performance, the retired 5.83× remote-prefill figure, the superseded
  1.3273× spec headline, or the barred 1.64960× decode-at-131k accounting
  `[docs/launch-claims.md]`. Every figure on that list was once the true
  result of a real run; each became misleading the moment it was quoted
  outside the cell that produced it. They are retained, and retained
  visibly, so nobody has to rediscover on their own why they were pulled.
- Ferrite-lineage numbers (the ancestor lab's A18-class measurements) are
  labeled as lineage when they appear, never as Muser results.

## 1.7 Where the book goes from here

You now hold the book's central derivation. The rest is a guided descent
into the machinery that lives under it.

- **Part I (this part)** teaches the Metal compute model —
  [Ch 2](02-metal-compute-model.md) (devices, queues, command buffers,
  threads, threadgroups, SIMD groups),
  [Ch 3](03-unified-memory-and-buffers.md) (unified memory and the buffer
  substrate that maps 16.76 GB of weights with zero copies), and
  [Ch 4](04-pso-and-three-kernel-sources.md) (how `.metal` source becomes
  runnable kernels — from three distinct sources).
- **Part II** is quantization: how 27.85 billion parameters fit in
  16.76 GB, and what each lane pays for it.
- **Parts III–IV** build the model and walk the decode loop kernel by
  kernel — the place where "one token = one pass over the weights" becomes
  fifty-two layers of named dispatches.
- **Part V** is the KV cache as an asset; **Part VI** is the disaggregated
  lane — the fourth lever, taken seriously; **Part VII** is serving;
  **Part VIII** is measurement and the evidence culture.

By the end, the sentence at the top of §1.1 will be something you have
proven, kernel by kernel, from source — not something you were told.

But there is a prerequisite. Everything in this book after this page
happens on the GPU, and to follow it you must speak Metal: what a device
is, what a command buffer records, what a threadgroup is, why the *SIMD
group* is the unit that actually matters on Apple Silicon. That language is
the next chapter.

---

## References

- `[crates/muser-engine/tests/muse_golden.rs:97-108]` — the pinned-artifact
  geometry assertions (52 layers, hidden 6,656, heads 32:2, head_dim 128,
  FFN 19,968, vocab 202,048).
- `[crates/muser-engine/src/config.rs:169-181]` — fail-closed GGUF metadata
  parsing of the same fields; `:294-318` the per-tensor shape contract used
  in the parameter count.
- `[crates/muser-engine/src/lib.rs:14]` — "The pinned target artifact is
  16,756,681,056 bytes on disk."
- `[crates/muser-server/src/chat_template.rs:237-261]` — the `release_gguf`
  test: byte size, chat-template length (7,167 B), and the three SHA-256
  identities.
- `[docs/memory-footprint.md]` — KV formula, the 96 GB budget table, the
  four-slot release contract, and its own "lower bound only" caveat.
- `[docs/benchmarks.md]` — §1 (35.44/35.49 parity-within-noise),
  §Methodology (ratio and repetition conventions), §3 (disaggregated
  payoff band).
- `[ledger P1.3]` — `docs/goal-parity-ledger-2026-08.md`, kquant/NVFP4
  plain-decode table: 35.440 (CV 0.037 %) / 35.491 (CV 0.130 %), five reps,
  66-token prefix / 32 teacher-forced tokens, F16 KV.
- `[ledger L0]` — same ledger, "microbenchmark-first apparatus and the
  occupancy bound": the "~800 GB/s M3 Ultra" memory-class reference.
- `[claims #11]`, `[claims #12]`, `[claims #15]` — `[docs/launch-claims.md]`:
  warm reuse at depth, delta handoff, current spec-decode restatement.
- `[receipt kvpack-ladder-20260820/attempt-10-…-stage6-delta/
  stage6-delta-65536/stage6-verdict.json]` — the 54.2851 % delta-share
  evidence (verified `delta_share_of_full: 0.5428507652…`,
  `exact_against_full_handoff: true`).
- `[ferrite-book Ch 1]` — the ancestor's "one token, costed by hand" device
  this chapter ports; its 1 GB / 45.95 GB/s / 33.20 tok/s numbers are
  Ferrite-lineage and do not transfer.
- [glossary](../glossary.md) — terms introduced this chapter: parameter,
  weights, token, decode, prefill, matvec, FLOP, GGUF, quantization,
  bandwidth, arithmetic intensity.
