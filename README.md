# muser-book

**How to Write an Inference Engine** — a zero-to-hero book about Muser: how
one token of Muse Glimmer is generated on Apple Silicon, how the kquant and
DFlash speculative lanes work, how kvpack saves and replays KV-cache state
exactly, and how a GB10 node prefills NVFP4 and hands the KV to a Mac over
an authenticated transport.

40 chapters, a 424-term glossary, a 322-entry bibliography, and two
appendices (kernel table, environment flags) — every measured number cited
to a retained evidence receipt.

## Read it

The rendered book publishes to GitHub Pages from this repository's `main`
branch: <https://highperformanceailab.com/muser-book/>.

## Build it locally

```sh
cargo install mdbook mdbook-mermaid   # or brew install mdbook mdbook-mermaid
mdbook build                          # output in book/
mdbook serve                          # http://localhost:3000
python3 scripts/check_social_meta.py  # verify rendered social metadata
```

## How the book keeps itself honest

- Chapters are written against **pinned revisions** of the source trees
  ([src/PINNED.md](src/PINNED.md)); a re-pin is an explicit event, never a
  silent follow.
- Every measured number carries a `[receipt …]` citation to the append-only
  evidence volume; nothing is rounded into existence.
- Style rules and citation notation live in
  [src/STYLE.md](src/STYLE.md).

The book is part of the Muser program:

- [muser](https://github.com/High-Performance-AI-Lab/muser) — the engine
- [kvpack](https://github.com/High-Performance-AI-Lab/kvpack) — exact KV-cache replay
- [muser-console](https://github.com/High-Performance-AI-Lab/muser-console) — the dashboard

## License

MIT OR Apache-2.0, at your option (see `LICENSE-MIT` and `LICENSE-APACHE`).
The bundled `src/mermaid.min.js` is MIT-licensed
([mermaid-js](https://github.com/mermaid-js/mermaid)) and carries its own
license header.
