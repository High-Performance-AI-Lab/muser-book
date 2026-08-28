# How to Write an Inference Engine

### The Muser book — Muse Glimmer on Apple Metal, kvpack, and disaggregated prefill

---

> **Scope of this book.** This book explains, end to end, how the Muser
> inference engine generates text from the pinned **Muse Glimmer** model
> (52 layers, ~30B parameters) on **Apple Silicon**, with a **kquant DFlash
> speculative lane**, a **native NVFP4 lane**, and a **disaggregated lane**
> where a remote NVIDIA GB10 ("GX10") node prefills NVFP4 under vLLM and hands
> the KV cache to the Mac over the authenticated **Handoff V2** transport.
> Every line of Metal and Rust quoted here is read from the pinned source
> tree; every number is tagged with its measurement source.

This is a zero-to-hero book. The first time a concept appears — a *softmax*, a
*SIMD group*, a *nibble*, a *residual stream*, a *rotary embedding*, a *KV
cache page*, an *NVFP4 block* — it is defined in place, with a diagram and a
worked example, before it is ever used to explain Muser code. A reader who
knows Rust but has never written a GPU kernel, and who has never read a
transformer paper, should be able to finish this book understanding the *math*
of LLM inference, the *Metal* that makes it fast on Apple Silicon, the
*precision discipline* that makes remote prefill trustworthy, and the
*evidence culture* that keeps all of it honest.

We wrote it the way we worked. When a chapter reaches a real decision, it does
not hand you the winning answer and move on. It sets up the fork, tells you
what we tried and what we expected, lets the attempt fail on the page, names
what the failure taught us — and only then shows what shipped. Put the other
way round: the dead ends are not confessions tucked into an appendix, they are
where the reasoning is visible, and they are the whole reason the shipped
answer deserves your trust instead of merely your patience. Wherever we tell
you one of those stories, the run that proved it is retained and cited, so you
can go and check us.

## How to read this book

You are holding eight parts, and they are not all the same kind of reading.
Which order, then, and how much do you have to swallow before you are allowed
to skip around?

**Linearly, the first time.** The chapters build on each other:

| Part | What it teaches |
|---|---|
| I — The problem and the Metal compute model | *No transformer knowledge assumed. By the end of Part I you can read any Metal shader in this book and know what every line means.* |
| II — Quantization | *How ~30 billion weights fit in a ~17 GB GGUF, and what each lane pays.* |
| III — The model | Muse Glimmer's 52-layer graph, and the forward pass at a glance. |
| IV — The decode path, kernel by kernel | *One chapter per stage, in execution order. The spine of the book.* |
| V — The KV cache and kvpack | *The cache is the engine's second life: what it costs, how it's laid out, and how it becomes a durable, portable asset.* |
| VI — The disaggregated lane | *Why a Mac and a GB10 are better together than either alone — and what it costs to trust someone else's prefill.* |
| VII — Orchestration and serving | *How 52 layers × a dozen kernels × 4 slots become one disciplined owner of one accelerator.* |
| VIII — Measurement and evidence | *How to know anything at all. The part that makes the other seven honest.* |

The appendices (A–F) hold the glossary, the kernel dispatch table, the lane
and flag surface, the bibliography, the pin record, and the writing contract.

**As a reference, afterwards.** Each kernel chapter is self-contained: it
states the math, the access pattern, the Rust dispatch, the Metal shader, the
measured tradeoffs, and its own bibliography. You can jump to any one chapter
once you have read Part I.

## The measured reality

A book about performance is worth exactly as much as its numbers, so the fair
question to ask before reading another sentence is: which machine made them,
under what load, and can anyone else check? We would rather answer that here,
at the front, than have you wonder about it for eight parts.

All Muser numbers in this book come from the retained evidence of the 2026-08
campaigns — five-repetition synthetic matrices, the kvpack ladder, the pacing
ladder, and the qualification wizard runs — recorded in the campaign ledger
(`docs/goal-parity-ledger-2026-08.md` in the Muser repo) and cited here with
receipt paths under `muser-receipt://`. The decode hardware is
one Apple Silicon Mac with an **M3 Ultra and 96 GB of unified memory**; the
prefill hardware is one **ASUS GX10 (NVIDIA GB10)** node on the same wired
10GbE fabric; the comparator is **pinned llama.cpp**. The shipped lane matrix
at the pin `[docs/muser-architecture.md]`:

| Lane | Prefill | Decode | Speculative | Intended use |
|---|---|---|---|---|
| Native NVFP4 | Spark tensor-core FP4 | Mac NVFP4 weights, FP16 KV, 35.491 tok/s | Rejected (fail-closed) | Fast product lane |
| kquant/reference | Reference path | kquant, 35.440 tok/s | 107.9 tok/s | Speculative + reference lock |
| Exact NVFP4 flag | Integer-dot verification producer | Mac NVFP4 | Verification only | Deterministic anchor |

Read that table for its gaps as much as for its throughputs. Only the
kquant lane carries a speculative number; the fast product lane's speculative
cell says *rejected (fail-closed)*, which is this engine's way of saying that
we tried the combination, it did not hold, and the code now refuses it outright
rather than quietly serving you something weaker under the same name. That
refusal has a chapter of its own. It is the shape of most stories in this
book: an attempt we believed in, a measurement that disagreed, and a piece of
machinery that now says no on purpose.

**The book's recurring question:** *what does one token cost, where does the
time go, and what may be moved — into a draft model, into a cache, or across
the wire — without breaking the exactness contract?* Every kernel chapter
returns to the dispatch-gap accounting that keeps Muser's decode at
parity-within-noise while rejecting fusions that would change logprobs
`[docs/decode-dispatch-gap-20260815.md]`; every systems chapter returns to
what the evidence actually permits us to believe.

## The pinned source of truth

Every book written against living code goes stale. The only choice an author
has is whether it goes stale visibly or invisibly, and a book whose line
numbers drift silently is worse than no book at all. So we pinned it.

This book is written against one pinned revision of Muser (see
[PINNED.md](PINNED.md)). When the book and the code disagree, **the code
wins**. Every quoted line is tagged `file:line` relative to the Muser
repository root so you can verify it. The canonical paths:

- **Metal shaders:** `crates/muser-engine/src/shaders/` (including the
  `ferrite/` lineage directory)
- **Engine and forward pass:** `crates/muser-engine/src/`
- **Benchmarks and harness:** `crates/muser-bench/`
- **kvpack adapter:** `crates/muser-kvpack/`
- **Server and sessions:** `crates/muser-server/`
- **Cluster/transport:** `crates/muser-cluster/`
- **Vendored kvpack:** `third_party/kvpack/`
- **Evidence:** `muser-receipt://` (append-only; cited by path)

## Lineage and attribution

Muser is a from-scratch Rust workspace with no Ferrite *runtime* dependency,
but its CPU text-extraction lineage and select shaders/kernels were adapted,
with attribution, from the private Ferrite research tree
(`docs/extraction-manifest.md`, `NOTICE`). This book has its own ancestor
too: the *Inference Book on Apple Metal* written against Ferrite and
Qwen2.5-1.5B on an A18 Pro. This book keeps that book's pedagogical spine —
zero-to-hero, define-everything-on-first-use, cite-everything — and rebuilds
the content for Muser's model, lanes, and measured reality. Where a Ferrite
lesson survives on Muser's live path, the book says so and cites it.

The reason that genealogy matters while you read, rather than only in the
credits: the ancestor ran a far smaller model on a phone-class chip. Its
numbers describe a different machine and do not transfer to this one. So when
an A18 Pro figure appears in these pages it is labelled as ancestry, never as a
Muser result, and the two never share an unmarked sentence.

## What this book is *not*

Some promises are easiest to keep as prohibitions. Each of the three below is a
genre this book could have slid into without anyone noticing, so we named them
early and held each other to them.

- It is **not** marketing. Every number states what was measured and under
  what scope — device, model, quantization, reps — and cites its receipt.
  If you want a sales pitch, this is the wrong book.
- It is **not** a survey of dead ends for their own sake. Rejected designs
  (the linear distributed-speculative lane, native NVFP4 speculative decode
  under Fallback B, the ANE route for v0.1) appear where their *failure*
  teaches a tradeoff that survives. When one does appear, it is told as what
  it was — a fork we walked down, an expectation we were holding, the
  measurement that ended the argument — and not as a verdict in a list, because
  a list of verdicts teaches nobody how to make the next decision. The path is
  as important as the destination.
- It is **not** speculation. Where a "why" cannot be cited from source or a
  measurement, it is marked **`[unverified]`** rather than smoothed over.

## Bibliography convention

Each chapter ends with a **References** section of its own, because a reader
finishing one kernel chapter should not have to hunt the back of the book.
You will usually meet these tags at the end of a paragraph, or gathered in a
section's evidence trail, rather than wedged into the middle of a sentence: the
evidence is not optional, but neither is being readable, and a claim you cannot
follow is no better cited than one you cannot check. Citation tags:

- `[crates/.../file.rs:LINE]` — Muser source, pinned revision.
- `[docs/<file>.md]` — Muser engineering documents.
- `[ledger §N]` — `docs/goal-parity-ledger-2026-08.md`, the campaign ledger.
- `[claims #N]` — `docs/launch-claims.md` row N.
- `[receipt <path>]` — evidence under `muser-receipt://`.
- `[ferrite-book Ch N]` — the ancestor book (pedagogical lineage).
- `[Metal-SS §N]` / `[Metal-PG §…]` — Apple Metal Shading Language
  Specification / Programming Guide.
- `[CUDA §…]` / `[PTX …]` — NVIDIA CUDA documentation, where the
  disaggregated lane demands the comparison.
- `[arxiv:XXXX.YYYYY]` — the paper.
- `[vLLM …]` / `[llama.cpp …]` — upstream projects Muser interoperates with.

## Status of each chapter

The book is written in passes, and we would rather admit which pass a chapter
is in than let you assume it has been checked. So every chapter declares itself
before it says anything else — a status line at the top:

- **`status: polished`** — reviewed, citable, ready to read.
- **`status: draft`** — written, not yet through review passes.
- **`status: stub`** — placeholder, to be filled.

## Building

The book lives in `src/` as plain Markdown with an mdBook-style
`SUMMARY.md`. To read linearly, start at
[Ch 1](chapters/01-why-inference-is-a-memory-problem.md). With `mdbook`
installed, `mdbook build` and `mdbook serve` work out of the box
(`book.toml` pins the metadata and the mermaid preprocessor; the site is
built without the repository's `_research/` working artifacts, which sit
outside `src/` by design). GitHub Pages deploys on every push to `main`
via `.github/workflows/pages.yml`.

That is the apparatus — the scope, the machines, the pin, the tags. The rest of
the book is the walk itself, and Part I starts where the work started: with the
question of where a token's time actually goes, and the discovery that the
answer has almost nothing to do with arithmetic.
