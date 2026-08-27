# PINNED — the exact trees this book quotes

This book is written against pinned revisions. When the trees move, the book
does not silently follow; a re-pin is an explicit event.

## Muser (the engine this book explains)

- **Repository:** the muser repository (`<muser-checkout>` at pin time)
- **Pinned commit:** `6d0807da975d3628f874df6b36ac9cc2af3723f2`
  (`feat(dashboard): live chat pane with token streaming`)
- **Branch:** `main`
- **Tree state at pin:** clean (working tree == HEAD), so quoting from the
  working tree is quoting the pin.
- **Model target:** the pinned Muse Glimmer GGUF (identity validated at
  startup: revision, byte size, SHA-256 — see `docs/muser-architecture.md`).
- **Compatibility reference:** llama.cpp commit
  `89e0aa6fd362617d9073e0dafc18e41241521572`.
- **Rules inherited from the Muser repo while working against it:** the
  release lock is authoritative (this book creates no seals, tags, or
  candidates); evidence is read from `muser-receipt://`
  (append-only — never written by book work); files under `~/.muser/**/secrets`
  and pki dirs are never read; `scripts/accelerator_safe.py` gates any
  accelerator run (book work is read-only, so none are needed).

## The ancestor book (pedagogical source)

- **Repository:** the private ferrite-rs research repository
- **Path:** `inference-book/`
- **Identity:** "The Inference Book on Apple Metal — How Ferrite runs
  Qwen2.5-1.5B, one kernel at a time" (25 chapters + appendices, STYLE
  contract, `SUMMARY.md`).
- **Use in this book:** structural and pedagogical lineage. Ferrite
  measurements (A18 Pro, 8 GB, 45.95 GB/s ceiling, 33.20 tok/s tg128, …)
  are ancestor context and are always labeled as such, per the Muser launch
  claims register's `[precedent-7B-ferrite]` discipline
  (`docs/launch-claims.md` ground rules).

## Verification recipe

1. `cd <muser-checkout> && git rev-parse HEAD` must print the
   pinned commit. If it does not, quotes in this book may have drifted —
   check `git diff <pin>..HEAD -- <quoted path>` before trusting a `file:line`.
2. Every `[crates/...:LINE]` tag in a chapter must resolve to the quoted
   content at the pin.
3. Every number must carry a tag that resolves to a doc, a ledger entry, or a
   receipt path under `muser-receipt://`.
