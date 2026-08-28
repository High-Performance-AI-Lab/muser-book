# Chapter 11 — Token embedding lookup
> **status:** polished  ·  **path:** Muse Glimmer, pinned Muser tree
>
> *Prerequisites: [Ch 2](02-metal-compute-model.md) (the Metal compute model),
> [Ch 3](03-unified-memory-and-buffers.md) (mmap'd GGUF, zero-copy weight
> views), [Ch 5](05-quantization-from-scratch.md) and [Ch 6](06-the-kquant-family.md)
> (Q4_K blocks), [Ch 9](09-muse-glimmer-architecture.md) (the model's shapes),
> [Ch 10](10-the-forward-pass-at-a-glance.md) (the decode loop and the residual
> stream). This is the first kernel chapter of Part IV: the operation is small,
> and the chapter is sized to match.*

Chapter 10 ended where the descent begins: box ① of Figure 10.1, the
humblest kernel in the whole graph — one integer in, one 6,656-float
vector out. This chapter opens it.

We put the smallest kernel first on purpose. Every habit the rest of Part IV
leans on — read the shapes before the code, count the bytes before believing
a cost, quote the dispatch instead of your memory of it — can be practised
here, on an operation that has no arithmetic to hide behind. If a chapter
about a table lookup feels like more scaffolding than the lookup deserves,
that is the point: the scaffolding is what we are teaching.

---

## 11.1 What it computes

The kernel answers one question: how does an integer become a vector? The
tokenizer hands the engine a number — the id of a piece of text inside a
fixed vocabulary — and everything downstream wants floats. Something has to
bridge the two, and this is the something.

Given one [token](../glossary.md#token) id — an integer in `0 .. vocab_size` —
the embedding lookup produces the initial hidden vector for that token:

```
hidden = EmbeddingTable[token_id]        // a [hidden_dim] f32 vector
```

That is the whole formula. There is no dot product, no reduction, no
nonlinearity: it is a **row gather** out of a learned table, followed by a
dequantization from [Q4_K](../glossary.md#q4_k) (the 144-byte super-blocks of
[Ch 6](06-the-kquant-family.md)) into f32. For Muse Glimmer,
`hidden_dim = 6,656` and `vocab_size = 202,048`
(`crates/muser-engine/tests/muse_golden.rs:96,101`), so the output of this
step is 6,656 floats — the seed of the [residual
stream](../glossary.md#residual-stream-hidden-state) that all 52 layers will then accumulate
into.

This chapter is about `muser_embedding_q4k`, the Metal kernel that does the
gather and the dequant in one dispatch, and about the one structural way
Muser's embedding differs from the ancestor book's: **Muser embeds on the
GPU**, inside the same command buffer as everything else
(`crates/muser-engine/src/decode.rs:5523-5533`).

## 11.2 Why it exists — the residual stream is born here

Recall the residual stream from [Ch 10](10-the-forward-pass-at-a-glance.md):
one `[6,656]` f32 buffer that every layer reads from and adds a delta to. The
embedding lookup is the **only** step in the entire forward pass that *writes*
the initial value of that buffer rather than adding into it. Everything
downstream — 52 layers of attention and FFN, the final norm, the LM head —
starts from the 6,656 floats this kernel produces.

What breaks if you skip it: nothing downstream has an input. The residual
stream would hold whatever the buffer was zero-filled with, every token would
embed to the same vector, and the model would emit a constant. The lookup is
trivial arithmetic carrying non-trivial content — the "intelligence" is in the
learned numbers of the table, not in the operation.

## 11.3 The row gather, explained from zero

Two questions decide everything about this kernel. Where in memory does a
token's row live, and how many bytes long is it? Answer both and the kernel
more or less writes itself; get either one wrong and you will read a
perfectly plausible row belonging to some other word entirely.

An **[embedding](../glossary.md#embedding)** is a lookup table: a matrix of
learned numbers with one row per vocabulary entry. In the GGUF the tensor is
`token_embd.weight` with shape `[hidden_dim, vocab_size]`. The loader does
not take that shape on trust — it asserts it at load
(`crates/muser-engine/src/config.rs:295`) — and for this checkpoint it
accepts the table in two dtypes only, Q4_K or F16 (`decode.rs:1209-1214`).
Memory is vocab-major: token `t`'s 6,656 values are contiguous
(Figure 11.1), so the byte offset of a row is just `t × row_bytes`.

```
                 token_embd.weight   (vocab-major rows)
               ┌────────────────────────────────────────────┐
        t = 0  │ 6,656 quantized values …  (3,744 B, Q4_K)  │
        t = 1  │ 6,656 quantized values …                   │
          ⋮    │      ⋮                                     │
   t = 202,047│ 6,656 quantized values …                   │
               └────────────────────────────────────────────┘
   To embed token id t:  read row t → dequant → hidden[0..6,656] (f32)
```

*Figure 11.1: The embedding table. One row per token id; one row is read per
token. Drawn in memory order this time (each token's values are contiguous),
unlike the transposed pictures some papers prefer.*

How big is one Q4_K row? Show the arithmetic, because the same multiply
returns in every kernel chapter:

```
  Q4_K super-block = 256 elements in 144 bytes          (Ch 6)
  row_bytes        = (6,656 / 256) × 144 = 26 × 144 = 3,744 B
  table total      = 202,048 × 3,744    = 756,467,712 B ≈ 756 MB
```

That multiply is not ours to invent, incidentally — the dispatch computes
`row_bytes` with the same expression
(`crates/muser-engine/src/metal/encode/qkv.rs:361`).

The whole table is ~756 MB of the 16,756,681,056-byte pinned artifact
(`crates/muser-engine/src/lib.rs:14`) — about 4.5 % of the model — but only
**one 3,744-byte row of it is read per token**. Large in memory, trivial in
bandwidth; both facts are true simultaneously. Put the other way, because the
whole book rests on the distinction: a tensor's size tells you what it costs
to *hold*, and tells you nothing whatever about what it costs to *use*. The
embedding table is the cleanest illustration in the model — three quarters of
a gigabyte resident, under four kilobytes touched per token — and
[Ch 1](01-why-inference-is-a-memory-problem.md) hangs its whole budget on
exactly that gap.

## 11.4 The Metal kernel

Now the code. Read it holding one question: which line is the actual gather?
There is exactly one, a pointer expression, and it is outnumbered about a
dozen to one by nibble bookkeeping. That ratio is not a defect of this
kernel. It is what quantization costs at every kernel in this book, visible
here only because there is nothing else going on to hide it.

Here is the kernel, complete, with the dequant helper it calls:

```metal
// crates/muser-engine/src/shaders/muse_reference.metal:946
inline float muser_q4k_value(device const uchar *row, uint element) {
    uint block_index = element / 256;
    uint within_block = element % 256;
    uint group = within_block / 64;
    uint within_group = within_block % 64;
    uint scale_index = group * 2 + (within_group >= 32 ? 1 : 0);
    uint lane = within_group % 32;
    device const uchar *block = row + block_index * 144;
    uchar2 scale_min = muser_scale_min(block + 4, scale_index);
    uchar packed = block[16 + group * 32 + lane];
    uint quant = within_group < 32 ? uint(packed & 0x0f) : uint(packed >> 4);
    return muser_f16(block) * float(scale_min.x) * float(quant)
        - muser_f16(block + 2) * float(scale_min.y);
}

// crates/muser-engine/src/shaders/muse_reference.metal:961
kernel void muser_embedding_q4k(
    device const uchar *weights [[buffer(0)]],
    device const uint *token_ids [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &hidden_dim [[buffer(3)]],
    constant uint &vocab_size [[buffer(4)]],
    constant uint &tokens [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    uint total = hidden_dim * tokens;
    if (index < total) {
        uint token_slot = index / hidden_dim;
        uint element = index % hidden_dim;
        uint token_id = min(token_ids[token_slot], vocab_size - 1);
        uint row_bytes = (hidden_dim / 256) * 144;
        output[index] = muser_q4k_value(weights + token_id * row_bytes, element);
    }
}
```

Line by line:

- **One thread per output element.** Thread `index` produces
  `output[index]`. For a single decode token that is 6,656 threads; for a
  512-row prefill chunk it is 512 × 6,656 ≈ 3.4 M threads, all from the same
  kernel. `token_slot = index / hidden_dim` picks which token of the batch
  this thread serves.
- **The gather itself** is the pointer line
  `weights + token_id * row_bytes`: one integer multiply, one add. The
  `min(token_ids[token_slot], vocab_size - 1)` clamp is a bounds guard — a
  hostile or corrupt id reads the *last* row instead of running off the table.
- **`muser_q4k_value` is the Q4_K dequant from [Ch 6](06-the-kquant-family.md),
  spelled out scalar-style.** Each element lives in one of the row's 26
  super-blocks (`element / 256`); each super-block has 8 sub-blocks of 32
  (`group = within_block / 64`, two sub-blocks per 64 elements, one per
  nibble-half of each byte); `muser_scale_min` unpacks the 6-bit scale/min
  pair for the sub-block (`muse_reference.metal:32-39`); the nibble is the
  low or high 4 bits of one byte depending on which half of the sub-block the
  element falls in. The value is
  `d × sc × nibble − dmin × m`, where `d`/`dmin` are the two f16 header
  values of the super-block (`muser_f16`, `muse_reference.metal:27-30`).
- Nothing is cached across threads; each thread independently decodes the
  scale bytes it needs. That is ~26 super-block-header reads per thread in
  the worst case, all served from L2 (the GPU's shared on-chip cache) after the first lane touches them —
  fine for a kernel that runs once per token.

Said the other way round, because this is the part that trips people up: an
element index is really a four-level address — which super-block, which
sub-block inside it, which byte inside that, and which half of that byte. The
kernel stores none of those levels. No lookup table, no precomputed offsets:
it re-derives all four from a single integer using divisions and masks,
independently, in every thread. Quantized formats trade memory for address
arithmetic, and this helper is that trade with the lid off.

### A worked example: one element of one row

Take token id `t = 42`, element `e = 300`, and a made-up super-block whose
header is `d = 0.5`, `dmin = 0.01` and whose sub-block scales decode to
`sc = 40`, `m = 12`, with the packed byte holding nibble `9` in the half
that covers element 300. Then:

```
  block_index = 300 / 256 = 1            (the second super-block)
  within      = 300 % 256 = 44
  group       = 44 / 64   = 0            (first 64-element group)
  within_grp  = 44 % 64   = 44  ≥ 32  → high nibble, scale_index 1
  value       = d·sc·nibble − dmin·m
              = 0.5 × 40 × 9 − 0.01 × 12 = 180 − 0.12 = 179.88
```

Multiply that by 6,656 threads and the row is dequantized into
`hidden[0..6,656]`. The numbers here are illustrative — the real bytes live
in the pinned GGUF — but every index computation above is the kernel's own
arithmetic, quoted from `muse_reference.metal:946-959`.

And the thread-to-element picture for one token (Figure 11.2):

```
  thread index:    0     1     2    …   255 |  256   257  …        6,655
  token_slot:      0     0     0    …     0 | (token 1's threads, in batch mode)
  element:         0     1     2    …   255 |    0     1  …          6,655
  reads:        row 42, bytes 0..3,744 — every thread re-derives its own
                block/group/lane indices into the same contiguous row
```

*Figure 11.2: the per-thread decomposition for `tokens = 1`. Grid is the
output shape (`dispatch_1d(hidden_dim × tokens)`); the gather is one
pointer computation per thread.*

## 11.5 The Rust dispatch

The kernel says what to compute. It does not say how many threads run it,
where the weight bytes come from, or what should happen if the buffer you
bound turns out to be the wrong size. That is the wrapper's job. It is the
layer where a mistake is expensive and silent, which is why it is also the
layer carrying the assertions.

The wrapper is `encode_embedding_q4k`:

```rust
// crates/muser-engine/src/metal/encode/qkv.rs:338
pub fn encode_embedding_q4k(
    &self,
    encoder: &ComputeCommandEncoderRef,
    weights: GpuByteView<'_>,
    token_ids: &GpuByteView<'_>,
    output: &GpuBuffer,
    hidden_dim: usize,
    vocab_size: usize,
    tokens: usize,
) {
    // …(F16-table branch elided: a 2-byte-per-element table dispatches
    //    `muser_embedding_f16` instead — see file)…
    let row_bytes = hidden_dim / 256 * 144;
    debug_assert_eq!(weights.len(), row_bytes * vocab_size);
    debug_assert_eq!(token_ids.len(), tokens * std::mem::size_of::<u32>());
    debug_assert_eq!(output.len(), hidden_dim * tokens);
    self.bind(encoder, "muser_embedding_q4k");
    encoder.set_buffer(0, Some(weights.metal()), weights.offset() as u64);
    encoder.set_buffer(1, Some(token_ids.metal()), token_ids.offset() as u64);
    encoder.set_buffer(2, Some(output.metal()), 0);
    set_value(encoder, 3, &(hidden_dim as u32));
    set_value(encoder, 4, &(vocab_size as u32));
    set_value(encoder, 5, &(tokens as u32));
    dispatch_1d(encoder, hidden_dim * tokens);
}
```

And the call site — the first dispatch of every token graph:

```rust
// crates/muser-engine/src/decode.rs:5523
dispatch(command, |encoder| {
    self.kernels.encode_embedding_q4k(
        encoder,
        self.embedding.view(&self.mapped_weights),
        token_view,
        &self.activations.hidden,
        cfg.hidden_dim,
        cfg.vocab_size,
        1,
    );
});
```

Three details matter, and each one is a cost that isn't there. First,
`self.embedding.view(&self.mapped_weights)`: the table is a view into the
single mmap'd GGUF buffer already mapped onto the GPU (`decode.rs:1201`), so
the "upload" of 756 MB of embedding is a no-op — the zero-copy promise of
[Ch 3](03-unified-memory-and-buffers.md), collecting for the first time.
Second, the token id arrives as `token_view` — four bytes in a small staging
buffer the CPU wrote before the command buffer existed
(`decode.rs:5433-5438`), so the GPU never waits on the host. Third, the grid:
`dispatch_1d(hidden_dim × tokens)`
(`encode.rs:1337-1343`) uses `dispatch_threads` with a threadgroup width of
at most 256 — for one token, 6,656 threads in 26+ threadgroups of 256. The
GPU, not the CPU, sizes the launch.

The batch-graph sibling `encode_embedding_q4k_from_u32_buffer`
(`qkv.rs:376`) is the same kernel with the id buffer already GPU-resident —
that is the variant the 512-row prefill chunks use, and the reason one kernel
serves both regimes.

## 11.6 The access pattern

Every kernel chapter reaches this point and asks the same question: where
does the time go? For the embedding the honest answer is "nowhere," and it is
worth seeing precisely how nowhere, because the shape of the accounting is
the one we will reuse on kernels where the answer is not so comfortable.

Per token:

```
  read : 1 row × 3,744 B (Q4_K)  +  4 B (token id)   ≈ 3.75 KB
  write: 6,656 × 4 B = 26,624 B ≈ 26 KB (f32 hidden)
```

The read is a single contiguous 3,744-byte span; the write is a single
contiguous 26 KB span. Against the ~16.76 GB of weight bytes the token graph
reads in total (`lib.rs:14`), the embedding row is
`3,744 / 16,756,681,056 ≈ 2 × 10⁻⁷` of the traffic. The activation write is
larger than the weight read here — the only chapter in Part IV where that is
true — and both are rounding error. Note also what is *not* read: the other
202,047 rows of the table stay untouched until some other token id asks for
them.

### The F16 sibling and the lane matrix

The dispatch you just read has an elided branch: if the table's byte length
says two bytes per element, the wrapper binds `muser_embedding_f16` instead
(`qkv.rs:348-359`). Same one-thread-per-element shape, same gather, no
dequant — a 2-byte half is widened to f32 directly. Which branch runs is a
property of *which lane you loaded*, not of any flag: the kquant
(`q4_k_xl`) artifact carries the table quantized, and a lane whose tables
ship F16 takes the sibling. The lane matrix of
[Ch 7](07-nvfp4-native-lane.md) decides; the kernel follows the bytes. That
is the book's first example of a pattern that recurs through Part IV — the
*dispatch* adapts to the artifact, while the graph structure (one gather,
one norm, four matvecs, …) stays fixed.

## 11.7 Tradeoffs

**GPU gather vs CPU lookup — Muser inverts the ancestor's choice.** We
inherited a decision here, and then we reversed it. The Ferrite book did the
embedding on the CPU — one quantized row dequantized into a unified-memory
buffer, no dispatch at all — and argued the work was too small to amortize a
GPU dispatch over `[ferrite-book Ch 9]`. That argument is sound on its own
terms, and going in we expected to keep the CPU path: it is less code, and
none of the arithmetic had changed.

What had changed was everything around the arithmetic. Muser runs the lookup
on the GPU anyway. Two structural facts, visible in the code rather than in
any A/B ledger, turn the cheap-looking CPU path into the expensive one:

1. **The token id is already a GPU buffer.** Muser's decode graph is one
   command buffer from embedding to softcap (`decode.rs:5448-5460`), fed by
   teacher-forced or batched token arrays that arrive as staging buffers. A
   CPU lookup would need to read the id back, dequantize 3,744 B, and write
   26 KB through unified memory — crossing the host/device boundary to save
   one dispatch that was going to be recorded anyway.
2. **One kernel serves every batch width.** The same `muser_embedding_q4k`
   covers 1-token decode, 64-row teacher-forced blocks, and 512-row prefill
   chunks (`qkv.rs:376`'s `tokens` parameter); a CPU path would need a
   per-width host loop.

There is no measured A/B of CPU-vs-GPU embedding in the campaign ledger
`[unverified]` — the choice is justified structurally, and at ~30 KB of
traffic either side would be invisible against the per-token budget.

The lesson we took from the reversal is worth carrying into the rest of the
kernel chapters: a dispatch's cost is not a property of the dispatch. It is a
property of what is already queued beside it. An argument that held for an
engine recording a handful of command buffers stops holding for one that
records the entire token graph as a single buffer.

**Q4_K table vs an f32 table.** Pre-dequantizing the table to f32 at load
would cost `202,048 × 6,656 × 4 ≈ 5.4 GB` of resident memory — about a third
of the whole artifact again — to save 3,744 bytes of dequant work per token.
The loader instead accepts the table in exactly the dtypes the pinned
artifact carries (Q4_K or F16, `decode.rs:1209-1214`) and dequantizes one row
on demand. The same trade the ancestor book described, at 7× the scale
`[ferrite-book Ch 9]`.

**The clamp instead of a fail.** A Metal kernel has no way to raise an error.
A thread handed a bad index simply reads whatever those bytes happen to be.
So what *should* a shader do with a token id that is out of range? Muser's
answer is the `min` on the pointer line: an out-of-range id silently maps to
the last row rather than erroring (`muse_reference.metal:973`). It is not the
correctness gate — the CPU-side session validates token ids before the graph
is ever built (`Session::decode`,
`crates/muser-engine/src/api.rs:696-741`), so by the time a thread reaches the
clamp a bad id should already be impossible. That makes the clamp a second
line of defense. It also makes it a silent one, and a silent defense is a
defense you never learn has fired. Which of those two descriptions the clamp
was actually written for is `[unverified]` from the tree alone.

### The untied twin at the other end of the model

`token_embd.weight` is not the only `[6,656 × 202,048]` tensor in the
checkpoint. `output.weight` — the LM head — has the same shape and is a
*separate* tensor: Muse Glimmer's embeddings are untied, and the loader
demands both (`config.rs:295-297`). The two tensors live opposite lives.
The embedding table is read one row per token on the GPU, ~3.75 KB at a
time (this chapter). The LM head is read as a *full matvec over all
202,048 rows*, every token, in the tail of the graph
(`decode.rs:5892-5897`) — at 2 bytes/element on the native lane that is
~2.7 GB of traffic and, measured, ~3.46 ms/token versus ~1.75 ms for the
kquant lane's quantized head `[docs/nvfp4-fast-lane-evidence §Measured
product numbers]`. Same shape, three orders of magnitude different
bandwidth. The LM head gets its own chapter,
[Ch 20](20-final-norm-lm-head-softcap.md); this chapter's table is the
quiet one.

## 11.8 Where the gap lives

Every kernel chapter closes by facing the same suspicion: is this the one
eating the decode time? Here the answer is a flat no, and the accounting is
specific enough to say exactly why.

**This kernel is not the gap.** The dispatch-gap accounting of
`[docs/decode-dispatch-gap-20260815.md]` reconciles the production-vs-legacy
closure delta (760 vs 564, +196 at position 2,048) into exactly four
families: 104 norm-boundary groups, 39 SWA staging groups, 52 KV-publication
splits, and 1 last-row copy — the embedding appears in the *common math*
row (406 closures, delta 0, "required math / Keep"). Its bytes are
nanoscopic and its one-dispatch cost is shared by both graphs. If you are
hunting decode time, this is the last place to look.

## 11.9 What comes next

The residual stream now holds 6,656 f32s and the real work can begin. The
first consumer is the entry RMSNorm — and on Muse Glimmer the norm story is
unusual: a *dual-epsilon sandwich* with a llama.cpp hard-coded 1e-8 that the
GGUF does not even carry. That is [Ch 12](12-rmsnorm-and-the-dual-eps-sandwich.md).

## References

- `crates/muser-engine/src/shaders/muse_reference.metal:946-959` —
  `muser_q4k_value`, the scalar Q4_K dequant used by the embedding (and the
  batch matvec fallbacks).
- `crates/muser-engine/src/shaders/muse_reference.metal:961-977` —
  `muser_embedding_q4k`, the kernel this chapter dissects.
- `crates/muser-engine/src/shaders/muse_reference.metal:27-39` — `muser_f16`
  and `muser_scale_min` (the f16 header read and the 6-bit scale/min unpack).
- `crates/muser-engine/src/metal/encode/qkv.rs:338-373` —
  `encode_embedding_q4k`, the dispatch wrapper; `:361` the `row_bytes`
  formula; `:376-411` the batch sibling.
- `crates/muser-engine/src/metal/encode.rs:1337-1343` — `dispatch_1d` (the
  `dispatch_threads` grid used here).
- `crates/muser-engine/src/decode.rs:5523-5533` — the call site, first
  dispatch of `encode_token`.
- `crates/muser-engine/src/decode.rs:1207-1216` — embedding loaded as a
  `Projection` from `token_embd.weight`; dtype gate (Q4_K or F16).
- `crates/muser-engine/src/decode.rs:5433-5438` — the four-byte token staging
  view the kernel reads.
- `crates/muser-engine/src/config.rs:294-297` — `token_embd.weight` shape
  `[hidden_dim, vocab_size]` asserted at load.
- `crates/muser-engine/tests/muse_golden.rs:96,101` — `hidden_dim = 6,656`,
  `vocab_size = 202,048` asserted against the pinned GGUF.
- `crates/muser-engine/src/lib.rs:14` — the 16,756,681,056-byte artifact.
- `crates/muser-engine/src/api.rs:696-741` — `Session::decode`, the host-side
  id validation that runs before any of this.
- `[docs/decode-dispatch-gap-20260815.md]` — the closure reconciliation this
  chapter's gap section cites (embedding in the 406-closure common-math row).
- [Ch 3](03-unified-memory-and-buffers.md) — mmap + zero-copy views.
- [Ch 6](06-the-kquant-family.md) — the Q4_K super-block this kernel
  dequantizes.
- [Ch 10](10-the-forward-pass-at-a-glance.md) — where this dispatch sits in
  the one-command-buffer token graph.
- [Ch 12](12-rmsnorm-and-the-dual-eps-sandwich.md) — the next kernel.
- [Ch 20](20-final-norm-lm-head-softcap.md) — the untied LM head, the other
  boundary tensor.
- `[docs/nvfp4-fast-lane-evidence §Measured product numbers]` — the
  ~3.46 ms/token F16 LM head vs ~1.75 ms kquant comparison (§11.7).
- `[ferrite-book Ch 9]` — the ancestor's CPU-lookup embedding and the
  too-small-to-amortize argument (pedagogical lineage only).
