# STYLE — the writing contract for this book

> Every chapter-writing agent and every review agent reads this file first.
> If a chapter violates a rule below, the reviewer flags it and the writer fixes
> it before the chapter is marked `polished`.

---

## 0. The single overriding principle

**Didactic above all.** Imagine the reader has never seen a GPU shader and has
never read a transformer paper. The first time *any* technical term appears —
`softmax`, `nibble`, `SIMD group`, `residual`, `GQA`, `f16`, `Q4_K`, `matvec`,
`threadgroup`, `KV cache`, `NVFP4`, `speculative decoding` — it is **defined
in place** with one sentence and (where useful) one tiny diagram or one worked
number, *before* it is used to explain anything. A term already defined in an
earlier chapter is cross-linked to that chapter and not redefined, but a
one-line reminder in parentheses is encouraged.

## 1. Chapter status line

Every chapter's second line (right under the `#` title) is:

```
> **status:** stub | draft | polished  ·  **path:** Muse Glimmer, pinned Muser tree
```

Never delete this line; reviewers flip `draft → polished`.

## 2. The standard kernel-chapter skeleton

A chapter about one kernel follows this structure (sections in this order):
1. **What it computes** — the math, in plain words + one formula block.
2. **Why it exists** — the role in the transformer; what breaks if you skip it.
3. **The matrix operation, explained** — first time a matmul/dot/softmax/reduction
   appears, draw the shapes and show a 2×2 worked example by hand.
4. **The Metal kernel** — quote the `kernel void` signature and the inner loop
   from source (with `file:line` tags). Explain line by line.
5. **The Rust dispatch** — the encode/dispatch wrapper, grid/threadgroup sizes,
   buffer binding. Quote it.
6. **The access pattern** — what memory is read, in what order, how much. This is
   where the bandwidth story lives.
7. **Tradeoffs** — *required section*. At least two of:
   - "Why this way and not the obvious alternative" + the measured consequence.
   - "It may seem better to do X, but doing X destroys Y" with a citation.
   - A dead alternative (rejected/fail-closed) and why it died, if relevant.
8. **Where the gap lives** (kernel chapters only) — how this kernel contributes
   to the measured decode-gap story (see `[docs/decode-dispatch-gap-20260815.md]`),
   or explicitly "this kernel is not the gap."
9. **References** — the chapter's own bibliography (see §5).

Non-kernel chapters (Metal model, quantization, architecture, KV cache,
transport, orchestration, measurement) adapt this skeleton but keep sections
1, 2, 4–5, 7, 8 where applicable.

## 3. Diagram rules

- **Mermaid** for: flow diagrams, pipeline diagrams, state machines, decision
  trees, the forward-pass block diagram, the handoff transaction, the compile
  pipeline.
- **ASCII** for: byte layouts, memory maps, buffer-binding tables, wire-frame
  layouts, register/SIMD lane decompositions, anything where character-columns
  carry meaning.
- **Mermaid never carries a byte offset.** If a diagram needs `0x00`, `0x10`,
  offsets, it is ASCII.
- Every diagram has a caption (`*Figure N.M: ...*`) and is referenced in prose
  ("see Figure 11.2").
- Prefer *annotated* diagrams over walls of text. A reader should be able to
  study the figure and get the gist before reading the paragraph.

## 4. Code blocks

- Quote **real source** with `file:line` tags, read from the pinned Muser
  tree at `<muser-checkout>` (see [PINNED.md](PINNED.md)).
  Convention:
  ```
  // crates/muser-engine/src/shaders/ferrite/matvec_multicol.metal:17
  kernel void matvec_q4k_f32_v4(...) { ... }
  ```
  For `third_party/kvpack`, tag with the kvpack-relative path.
- Keep quotes tight: the kernel signature + the load + the inner loop + the
  write-back. Do **not** dump 200 lines. Link to the file for the rest.
- If you trim lines, insert `// …` on its own line and say so: *(lines 60–120
  elided: scale decoding, see file)*.
- Rust dispatch wrappers: quote the function and the dispatch call. Always
  state grid and threadgroup size in prose too.
- Never paraphrase code as if it were quoted. If it's a paraphrase, write it as
  prose, not in a code fence.
- Never invent a file, function, constant, or number. If you cannot find it in
  the tree, either hunt until you do or write `[unverified]`.

## 5. Citations and per-chapter bibliography

- Every chapter ends with a `## References` section.
- Citation tags in prose:
  - `[crates/.../file.rs:LINE]` — Muser source (pinned revision).
  - `[docs/<file>.md]` / `[docs/<file>.md §N]` — Muser engineering docs.
  - `[ledger §N]` — the campaign ledger `docs/goal-parity-ledger-2026-08.md`.
  - `[claims #N]` — `docs/launch-claims.md` row N.
  - `[receipt muser-receipt://...]` — retained evidence.
  - `[scripts/...]` — tooling, including `scripts/gx10/`.
  - `[ferrite-book Ch N]` — the ancestor Ferrite book, pedagogical lineage only.
  - `[Metal-SS §N]`, `[Metal-PG §…]`, `[CUDA §…]` — vendor documentation.
  - `[arxiv:XXXX.YYYYY]` — arXiv paper.
  - `[vLLM …]`, `[llama.cpp …]` — upstream interop targets.
- A measurement number is **always** followed by its tag.
- Claims about *why* something is the way it is: cite a file, a doc, a ledger
  entry, or a receipt. If you cannot, write `[unverified]` or rephrase as a
  question. **Do not produce fluent authoritative paragraphs from
  pattern-matching.** (The Muser repo's epistemic rule; the book inherits it.)
- Ferrite-lineage numbers (A18 Pro, Qwen2.5-1.5B, 33.20 tok/s, 45.95 GB/s,
  `[precedent-7B-ferrite]` figures) are **ancestor context**, never Muser
  results. If cited, label them as Ferrite-lineage measurements explicitly.

## 6. The tradeoff discipline

Every kernel chapter's "Tradeoffs" section must connect to **measured
reality**, not intuition. Acceptable forms:

- *"Fusing X into Y changed the dispatch-gap accounting by N groups
  `[docs/decode-dispatch-gap-20260815.md]`; the unfused path cost was M
  groups/token `[file:line]`."*
- *"The alternative passed the correctness gate but regressed production by
  −N % `[ledger §K]`."*
- *"The linear distributed-verifier lane reached 110.59 tok/s only on an
  all-accept control; real acceptance collapsed to 9.23–38.07 % and the lane
  was rejected `[claims #14]`."*

Unacceptable: *"This is the standard pattern,"* *"GPU memory works best
when..."*, *"Fusing is generally faster"* — without a measurement.

## 7. Voice

- Second person ("you") is fine and friendly.
- Short sentences. One idea per paragraph.
- No emojis. No marketing tone. No hype.
- When a concept is subtle, say so: *"This is the part that trips people up."*
- When something is genuinely a hack or a wart, name it.
- When something is fail-closed on purpose, explain what failure it prevents
  and what the operator sees when it trips. Muser's fail-closed culture is a
  *subject* of this book, not an obstacle to it.

## 8. Numbers

- Throughput/latency numbers always carry their scope and source tag, and
  carry the campaign's framing (five-repetition means, synthetic vs natural,
  notarial vs non-notarial) when the claim depends on it. Where the scope
  language is long, link to the claims row instead of paraphrasing it loosely.
- Byte sizes and offsets: compute them in the chapter and show the arithmetic
  so the reader can re-derive.
- Never silently mix Ferrite-lineage hardware numbers (A18 Pro) with Muser
  measurements. Say which machine made which number.

## 9. Cross-references and segues

- Link to other chapters by relative path: `[Ch 13](13-the-qkv-gate-matvec-family.md)`.
- **Every chapter ends with a "What comes next" transition** (one short
  paragraph, no heading, before References or as the last prose) that tells
  the reader what question is now open and which chapter answers it. Every
  chapter's **introduction** opens by recalling where the previous chapter
  left off (one or two sentences; Part-openers recall the previous *Part*).
- Maintain the [glossary](glossary.md): when you define a term in a chapter,
  add it to the glossary with a back-reference to the chapter that introduced
  it.
- The first time a term appears in a chapter, link it to its glossary entry:
  `[SIMD group](glossary.md#simd-group)`.

## 10. What kills a chapter (reviewer veto triggers)

- An undefined term used as if understood.
- A "why" with no citation or `[unverified]` tag.
- A mermaid diagram carrying byte offsets.
- A quoted code block that is actually paraphrase, or a `file:line` tag that
  does not resolve in the pinned tree.
- A tradeoff section with no measurement.
- A number with no source tag.
- A missing entry/exit segue.
- Ferrite-lineage numbers presented as Muser measurements.

## Narrative: receipts need a story

The governing essay is ["Notes on the Synthesis of
Labyrinths"](https://qa.increment.com/documentation/notes-on-the-synthesis-of-labyrinths/):
present the labyrinth of the work — the forks, the dead ends, the reasoning
at each junction — not a flattened list of outcomes. Receipts stay, always;
what changes is how they enter the prose.

- Open a section on the question it answers before any apparatus appears.
- A failed attempt is a war story, in order: the fork, what we tried, what
  we expected, what happened, what it taught, and only then what shipped.
  Never a staccato list of verdicts with citations.
- Weave evidence into the sentence ("the run that proved this is retained:
  [...]") instead of stacking bracket tags mid-thought. One tag per
  sentence is a good ceiling; the rest can close the paragraph.
- A digression must announce its own relevance in its first sentence.
- Restate the one or two genuinely hard ideas of a chapter in fresh words;
  pay for it by abbreviating inventory.
- The voice is "we": we tried things, expected things, and were surprised,
  and the reader is walking beside us.
