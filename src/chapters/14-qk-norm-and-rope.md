# Chapter 14 — QK-norm and RoPE — rotating only the layers that rotate
> **status:** polished  ·  **path:** Muse Glimmer, pinned Muser tree
>
> *Prerequisites: [Ch 9](09-muse-glimmer-architecture.md) (the two layer
> classes, GQA), [Ch 12](12-rmsnorm-and-the-dual-eps-sandwich.md) (the RMSNorm
> machinery this chapter reuses), [Ch 13](13-the-qkv-gate-matvec-family.md)
> (the Q/K/V/gate projections that produce this chapter's inputs). No prior
> exposure to positional encodings is assumed.*

---

## 14.1 What this chapter computes

Chapter 13 ended with Q, K, V, and gate holding raw projection outputs —
un-normalized, un-rotated, positionless. This chapter is what happens to
Q and K next, and it turns on a question that sounds trivial until you try
to answer it inside a kernel: how does a machine built entirely out of dot
products know which token came first? Muse Glimmer answers that question
twice, differently, in the same forward pass — and the second answer is
the one the rest of the book keeps cashing in. Two operations run between
the projections and the KV store, and a third deliberately does not:

1. **Per-head QK-norm** — an RMSNorm applied to *each attention head's*
   128-wide slice of Q and K separately (`rms_norm_per_head` family; the
   live dispatch reuses [Ch 12](12-rmsnorm-and-the-dual-eps-sandwich.md)'s
   wrapper). On Muse Glimmer it is parameterless in effect: the weight
   tensors are converter-synthesized constant broadcasts, verified at load.
2. **RoPE** (Rotary Positional Embedding) — a per-position *rotation* of
   Q and K, applied **only on the 39 sliding-window layers**.
3. **NoPE** — the 13 full-attention layers (`{3, 7, …, 51}`) apply **no
   positional rotation at all** (`config.rs:51-60`). Their position
   information comes entirely from the causal mask and the KV layout.

In one formula, for one pair of coordinates of a head vector at position
`pos` on a sliding layer, with pair frequency `θ_i`:

```
┌ x₀' ┐   ┌ cos(pos·θᵢ)  −sin(pos·θᵢ) ┐ ┌ x₀ ┐
└ x₁' ┘ = └ sin(pos·θᵢ)   cos(pos·θᵢ) ┘ └ x₁ ┘
```

Nothing is added, nothing is learned at runtime; each 2D pair is spun by an
angle that grows with position. The payoff — proved in §14.4 — is that
attention's `Q·K` ends up depending only on the position *difference*.

The two layer classes make this chapter's structure unusual among Muse
engines: the position question has **two different answers in one model**,
and the second answer (NoPE) is what makes the disaggregated lane of
Part VI possible.

## 14.2 Why position must be injected at all

This is the part that trips people up the first time. Attention — the one
operation that reaches across tokens ([Ch 16](16-attention-decode-kernels.md))
— is **permutation-invariant**. At its core it is dot products and a
weighted sum, and a dot product does not care which token came first. So to
raw attention, the sentences

```
the cat sat on the mat
mat the on sat cat the
```

look identical — same tokens, order differs, attention cannot see order.
But order is the whole point of language. Something must tell the model
"this token is at position 0, this one at position 5." Skip it and the
model is order-blind; it will produce plausible-looking gibberish that
no logprob gate would forgive. The three classical schemes, in one
paragraph each:

1. **Absolute embeddings** — learn one vector per position, add it to the
   token embedding. Simple; generalizes poorly past trained positions, and
   distances must be *learned*.
2. **Relative biases** — add a per-pair bias to each attention score.
   Generalizes better; modifies the attention kernel itself.
3. **RoPE** — rotate Q and K so their dot product depends only on the
   position difference. Relative position falls out of the math for free,
   and the attention kernel is untouched.

Muse Glimmer picks RoPE for its sliding layers — and picks *nothing* for
its full layers, trusting the window/mask structure to carry order where
the context is unbounded. Why a hybrid like this works is a model-design
claim the engine inherits `[unverified]`; what the engine must do is
implement each layer's choice exactly.

## 14.3 QK-norm first — the parameterless cousin

Before anything spins, a smaller question has to be settled: are all the
heads speaking at the same volume? Attention's softmax is a competition,
and a head whose Q and K happen to leave the projection with a large
magnitude wins that competition for reasons that have nothing to do with
meaning. QK-norm is the answer to that, and on this model it is a strange
one — a normalization with no learned parameters, wearing a learned
parameter's clothes. Immediately after the projections, before any
rotation, each head of Q and K is RMSNormed across its own 128 dimensions:

```rust
// crates/muser-engine/src/decode.rs:5599
dispatch(command, |encoder| {
    self.kernels.encode_qk_norm(
        encoder,
        &self.activations.q,
        &layer.q_norm,
        &self.activations.q,
        cfg.head_dim,
        cfg.rms_eps,
        cfg.n_heads,
    );
    self.kernels.encode_qk_norm(
        encoder,
        &self.activations.k,
        &layer.k_norm,
        &self.activations.k,
        cfg.head_dim,
        cfg.rms_eps,
        cfg.n_kv_heads,
    );
});
```

`encode_qk_norm` (`metal/encode/norm.rs:286-303`) delegates to
`encode_rms_norm_mul` with `dim = head_dim = 128` and `rows = 32` (Q) or
`2` (K) — i.e. **the per-head norm is dispatched as a batched 128-wide
RMSNorm whose rows are heads**, in place. The dedicated
`rms_norm_per_head` kernel exists in the pipeline registry
(`shaders/ferrite/rms_norm_per_head.metal:15`, one threadgroup per head,
tree reduction in threadgroup memory), and it is the right place to read
the algorithm — but the live decode path reaches the same math through the
shared norm wrapper of [Ch 12](12-rmsnorm-and-the-dual-eps-sandwich.md)
(whose pinned-ggml preference applies at this width too).

The Muse-specific wrinkle is the γ. The upstream checkpoint has *no
learned* q/k norm weights; the GGUF converter materializes
`full(qk_scale_factor)` for `attn_q_norm.weight` and `ones(...)` for
`attn_k_norm.weight` so llama.cpp's weighted-RMSNorm op can carry a scalar
(`config.rs:383-395`).

That leaves a fork at load time, and it is worth walking both branches.
The comfortable one is to shrug and multiply: the tensors are present, the
norm kernel already takes a weight vector, so read whatever is in the file
and move on. The uncomfortable one is to ask what happens the day the file
stops being what we assume — the day a converter writes a genuinely
learned per-channel norm into those same two slots. Nothing would crash.
The kernel would happily consume a vector where we expected a broadcast,
the math would quietly become a different model's math, and the output
would stay fluent. That is the failure mode this book keeps meeting:
plausible text is not evidence of a correct engine.

Muser takes the uncomfortable branch and *proves* the assumption instead
of holding it. `QkNormProbe` fails the load unless both tensors are exactly
the constant broadcasts the converter emits (`config.rs:397-403`), so a
learned per-channel norm aborts startup instead of silently redefining the
attention scores (`loader.rs:28-37`). The same probe pins the values it
accepts: `qk_scale_factor ≈ 3.87` and `k_norm = 1.0` (`config.rs:139-141`),
with an arithmetic companion test at `config.rs:426-430` checking
`3.87 × 1/√128 ≈ 0.342`. The lesson outlives this one tensor: when a number
reaches the engine from a converter rather than from training, the only
safe way to depend on it is to assert it at the boundary.

So Q's norm is "normalize, then scale by ≈3.87" and K's is a plain
normalize. And note what the scale is **in addition to**: the attention
softmax scale is still `1/√128 ≈ 0.0883883` (`config.rs:277-281`),
independent of the folded-in 3.87 — two different scales living at two
different points of the graph, easy to conflate, asserted apart by test.

Back to the question this section opened with, now that the machinery is
on the page. Per-head normalization exists to stabilize the score
distribution *within* each head before the dot product, so that one
hot-headed head cannot dominate the softmax on the strength of its
magnitude alone. On this model the "learned" part of that
stabilization was folded into a single scalar by training upstream
`[unverified]` for the quality rationale — what is verified is the probe,
the values, and where they are applied.

## 14.4 RoPE from zero

### 14.4.1 Pair up, rotate, at a per-pair frequency

So how do you tell a dot product where a token sits, without adding a
parameter and without touching the attention kernel? You spin the vector.
The mechanism is easier than RoPE's reputation suggests; what ruins engines
is never the rotation itself but the bookkeeping around it, so build the
mechanism first and meet the bookkeeping immediately after.

Take one head's 128-wide vector and split it into 64 pairs. Muse Glimmer
uses the **interleaved** convention — pair `i` is the *adjacent* couple
`(x[2i], x[2i+1])` — not the half-split ("NEOX") convention of Llama/Qwen.
The shader file's own comment block is the authoritative statement and is
worth quoting whole:

```metal
// crates/muser-engine/src/shaders/ferrite/rope.metal:610
// ── NORM-convention RoPE (LLAMA_ROPE_TYPE_NORM) ─────────────────────────
//
// Every other kernel in this file uses the NEOX convention: rotate the pair
// (x[i], x[i + half_hd]). llama.cpp's `rope_norm` instead rotates the
// *interleaved* pair (x[2i], x[2i+1]). Both read the SAME frequency table —
// freq[j] = base^(-2j/head_dim) — so only the element pairing differs, and a
// model that needs one and gets the other still produces fluent text with
// silently wrong positions. That is why these are separate kernels rather
// than a runtime flag threaded through the NEOX ones.
//
// Muse Glimmer is the first NORM-rope architecture in the tree. Its GGUF
// converter un-permutes Q/K at conversion time precisely so the interleaved
// form is the correct one for the stored weights.
```

Read the warning twice: rotating with the wrong pairing **still runs and
still produces fluent text** — with silently wrong positions. There is no
crash to catch; only a parity gate against pinned llama.cpp can see it.
A bug with no symptom cannot be found in production, so it has to be made
unrepresentable at build time instead — the defense the tradeoffs section
at the end of this chapter returns to.

Pair `i` has a fixed frequency, set once at load:

```
θ_i = rope_base_swa^(−2i/head_dim)     (i = 0..63)
```

`rope_base_swa` is read from the GGUF key `rope.freq_base_swa` (falling
back to `rope.freq_base`, default `10,000` if absent,
`config.rs:120-129, 199-205`). The pinned checkpoint's value is not
asserted anywhere in the tree **[unverified]** — what is verified is the
key it is read from and the table formula built in Rust:

```rust
// crates/muser-engine/src/decode.rs:1256
(0..cfg.head_dim / 2)
    .map(|index| {
        1.0 / cfg
            .rope_base_swa
            .powf(2.0 * index as f32 / cfg.head_dim as f32)
    })
    .collect::<Vec<_>>()
```

The `powf` happens once per engine; no kernel ever calls `pow` — that is
what the `_cached` in the kernel name means.

### 14.4.2 The rotation, and one worked pair

The rotation of a 2D vector `(v₀, v₁)` by angle `a` is the 2×2 matrix every
graphics programmer has memorized:

```
   v₀' = v₀·cos(a) − v₁·sin(a)
   v₁' = v₀·sin(a) + v₁·cos(a)
```

Length is preserved; the vector spins counterclockwise. Work one toy case,
`head_dim = 4` (two pairs), using the config-default base `10,000` so every
number is real:

```
  θ₀ = 10,000^(−0/4) = 1.0        (fastest pair: 1 radian per position)
  θ₁ = 10,000^(−2/4) = 0.01       (100× slower)

  At pos = 2, take x = [1, 0, 1, 0]  (both pairs pointing at their x-axis):

    pair 0, angle 2×1.0  = 2.0 rad :
        (1, 0) → (cos 2.0, sin 2.0)          ≈ (−0.4161,  0.9093)
    pair 1, angle 2×0.01 = 0.02 rad :
        (1, 0) → (cos 0.02, sin 0.02)        ≈ ( 0.9998,  0.0200)

  x' ≈ [−0.4161, 0.9093, 0.9998, 0.0200]
```

*Figure 14.1: RoPE at pos 2 on a head_dim-4 toy. The fast pair has been
thrown nearly onto the negative x-axis; the slow pair barely moved. Every
position leaves a different fingerprint of angles.* At real width 128,
frequencies `θ_i` span many orders of magnitude — a bank of 64 clocks at
wildly different speeds, from "full turn every ~6 positions" (`i = 0`) to
"one turn in ~10⁶ positions" for the slowest pair at base 10⁴ (and further
out at larger bases). A larger base slows the slow clocks, keeping them
from aliasing — completing a full turn and landing two distant positions
on the same angle — within a long context. That is why long-context models
choose large bases `[arxiv:2104.09864]`; which trade-off Muse Glimmer's
authors weighed for 131,072 positions is theirs, not the engine's
`[unverified]`.

### 14.4.3 The magic: the dot product sees only (m − n)

Why rotate *both* Q and K? Suppose Q sits at position `m`, K at position
`n`; RoPE spins pair `i` of Q by `m·θᵢ` and of K by `n·θᵢ`. Their dot
product over one pair, with raw values `(q₀,q₁)` and `(k₀,k₁)`, expands
and — using `cos(A−B) = cos A cos B + sin A sin B` and
`sin(A−B) = sin A cos B − cos A sin B` — collapses:

```
q_rot · k_rot  =  (q₀k₀ + q₁k₁)·cos((m−n)θᵢ)
               +  (q₀k₁ − q₁k₀)·sin((m−n)θᵢ)
```

Every `m` and `n` appears only as the difference `m − n`. Absolute
positions have vanished; relative position falls out for free, with zero
extra parameters and no change inside the attention kernel.

Worth restating in different words, because this is the idea the whole
chapter hangs on: RoPE never tells attention where a token *is*. It
arranges matters so that attention cannot ask anything else but how far
apart two tokens are. The absolute coordinates go in, cancel against each
other on the way through the dot product, and only the gap comes out.

That single identity is why RoPE is the decode-era default — and, in this
book, it has a second life in §14.6: it is *also* the reason NoPE's cache
bytes are relocatable and RoPE's are not.

## 14.5 The Metal kernel and the SWA-only dispatch

That is the theory; here is all of it in hardware. The kernel's job is
narrower than the derivation makes it sound — for this token, which floats
get spun, and by how much? — and everything interesting in the listing is
about answering that cheaply enough for the work to be invisible against
the weight stream.

```metal
// crates/muser-engine/src/shaders/ferrite/rope.metal:624
kernel void rope_norm_batch_cached(
    device       float* Q          [[ buffer(0) ]],
    device       float* K          [[ buffer(1) ]],
    device const float* freq_table [[ buffer(2) ]],  // [half_hd]
    constant     uint&  n_heads    [[ buffer(3) ]],
    constant     uint&  n_kv_heads [[ buffer(4) ]],
    constant     uint&  head_dim   [[ buffer(5) ]],
    constant     uint&  start_pos  [[ buffer(6) ]],
    uint2 tgid [[ threadgroup_position_in_grid ]],
    uint  lid  [[ thread_index_in_simdgroup ]])
{
    const uint batch   = tgid.y;
    const uint pair_id = tgid.x * 32u + lid;

    const uint half_hd       = head_dim / 2u;
    const uint total_q_pairs = n_heads    * half_hd;
    const uint total_pairs   = total_q_pairs + n_kv_heads * half_hd;
    if (pair_id >= total_pairs) return;

    const uint pos = start_pos + batch;

    const bool is_q  = (pair_id < total_q_pairs);
    const uint local = is_q ? pair_id : (pair_id - total_q_pairs);
    const uint head  = local / half_hd;
    const uint pi    = local % half_hd;

    const uint q_stride = n_heads    * head_dim;
    const uint k_stride = n_kv_heads * head_dim;

    device float* base = is_q
        ? (Q + batch * q_stride + head * head_dim)
        : (K + batch * k_stride + head * head_dim);

    float angle = float(pos) * freq_table[pi];
    float cos_a = precise::cos(angle);
    float sin_a = precise::sin(angle);

    // NORM convention: the pair is adjacent, at 2*pi and 2*pi + 1.
    const uint i0 = 2u * pi;
    float v0 = base[i0];
    float v1 = base[i0 + 1u];
    base[i0]      = v0 * cos_a - v1 * sin_a;
    base[i0 + 1u] = v0 * sin_a + v1 * cos_a;
}
```

One thread per pair, in-place on Q and K:

- **The decomposition** `pair_id → (is_q, head, pi)` — flat index into
  "all pairs of Q then all pairs of K"; `head = local / 64`,
  `pi = local % 64` at width 128. `pi` is exactly the `i` of `θᵢ`.
- **No `pow`** — one multiply against the cached table.
- **`precise::cos` / `precise::sin`** — Metal's fast-math transcendentals
  can be a few ULP off; `precise::` selects the higher-accuracy variant
  *for these calls only*, without giving up fast-math elsewhere in the
  library ([Ch 4](04-pso-and-three-kernel-sources.md)). At large bases and
  large positions the *angle* `pos·θᵢ` is itself large and its sine/cosine
  must be exact to the bit the comparator computed — accuracy, not speed,
  is the reason (the ancestor book carries the same lesson
  `[ferrite-book Ch 4, Ch 13]`).
- **The interleaved addressing** `i0 = 2·pi` — the NORM convention of
  §14.4.1, two adjacent floats rotated as one 2D vector, written straight
  back.

The dispatch (`metal/encode/rope.rs:139-151`) launches
`ceil(total_pairs/32) × batch` threadgroups of 32 — one thread per pair,
and with `total_pairs = (32 + 2) × 64 = 2,176` for Muse Glimmer that comes
to 68 threadgroups of 32 threads, once per sliding layer, per token.

The wrapper has more to decide than a grid size, though. The same rotation
exists three times in the tree, and the copies are not interchangeable —
same math, different provenance for the bits, and provenance is precisely
the kind of difference the convention warning above says never surfaces
in the output. When the pinned metallib is loaded, the wrapper prefers
llama's own `kernel_rope_norm_f32` with a packed `GgmlMetalKargsRope`:
same convention, llama's own arithmetic, no daylight between the engine
and the comparator it is scored against (`rope.rs:88-138`). Without that
metallib, the ferrite kernel quoted above is the fallback. Under the
cross-vendor flags the choice changes once more, to the no-fast-math NCO
table route with explicit per-token positions (`rope.rs:62-86`) — the
seam [Ch 32](32-precision-across-the-handoff.md) needs when the tensors
on the other side of the handoff were produced by somebody else's
hardware.

### The dispatch condition — SWA only

Which layers actually pay for all this? Not most of them — and the whole
position apparatus hangs off a single predicate in the token graph:

```rust
// crates/muser-engine/src/decode.rs:5621
if cfg.layer_kinds[layer_index].uses_rope() {
    dispatch(command, |encoder| {
        self.kernels.encode_rope_norm_batch_cached(
            encoder,
            &self.activations.q,
            &self.activations.k,
            &self.rope_frequencies,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.head_dim,
            position,
            1,
            // …(positions view + freq_base + n_ctx_orig elided)…
        );
    });
}
```

and `uses_rope()` is one line, with provenance to the comparator:

```rust
// crates/muser-engine/src/config.rs:66
/// RoPE runs iff the layer is a sliding layer (`muse-glimmer.cpp:93`:
/// `const bool use_rope = hparams.is_swa(il);`).
pub const fn uses_rope(self) -> bool {
    matches!(self, Self::SlidingRope)
}
```

On layers `{3, 7, …, 51}` no rotation dispatch exists at all. Q and K
flow from QK-norm straight to the KV store, positionless by design. The
layer-kind partition itself is fail-closed: the loader panics rather than
guess if the `sliding_window_pattern` key is missing, because "an
all-full model runs and emits plausible text while being wrong"
(`config.rs:378-380`). Notice what kind of defense that is. The loader is
not saving itself from a crash — it is saving us from the *absence* of
one, which is the same reason the QK-norm probe earlier in this chapter
exists. Two different keys, two different tensors, one shared conviction:
on this engine a missing assumption must stop the process, because the
model will never complain on its own.

## 14.6 NoPE and relocatable KV — the consequence

Here is why the two-class asymmetry is the most consequential layout
decision in the engine. A cached K row for a *sliding* layer is a function
of its absolute position: the rotation angle `pos·θᵢ` is baked into the
bytes. Move that row to a different position and the (m−n) identity of
§14.4.3 breaks — the stored rotation is the wrong one. A cached K row for
a *NoPE* layer is just a projection of the token's hidden state: it
carries no positional information whatsoever, so the same bytes are valid
at any position. Two facts follow, both load-bearing for Part V and VI:

- **NoPE tiles relocate by `memcpy`.** A 512-token NoPE KV tile can be
  planted anywhere in any cache of the same identity — prefix reuse, warm
  sessions, remote handoff — with no recompute and no re-rotation. The
  engine's own summary: the 13 NoPE layers are "position-free (relocate =
  memcpy) — the whole kvpack free lunch" (`lib.rs:9-10`), and kvpack's
  layout keys enforce it fail-closed (the NoPE identity requires
  `theta = 0`; a RoPE layer cannot claim it — `crates/muser-kvpack/src/layout.rs`,
  K1 keys `[crates/muser-engine/src/lib.rs:7-15]`).
- **SWA planes are position-bound but bounded.** The 39 sliding layers
  only ever need the last 2,048 tokens (§14.7 previews the ring), so their
  "where" question is a window offset, not an absolute position — that is
  [Ch 15](15-kv-store-and-the-ring.md)'s whole subject.

When [Ch 24](24-kvpack-the-format.md) and
[Ch 26](26-delta-handoff-and-migration.md) move KV tiles around the lab,
this section is the reason NoPE bytes move freely and SWA bytes move as
logical tails with explicit origins.

## 14.7 Tradeoffs

Four decisions in this chapter could plausibly have gone the other way,
and they share a family resemblance: in every one of them the wrong branch
still runs, still returns floats, and still reads like language. That is
what makes them worth walking rather than tabulating.

**Interleaved vs half-split — a convention you must match, not choose.**
Same rotation math, different pairing (Figure 14.2). The checkpoint was
trained with one; the GGUF converter un-permuted Q/K "precisely so the
interleaved form is the correct one for the stored weights"
(`rope.metal:620-622`), and llama.cpp's `rope_norm` matches. Mixing
conventions is the classic silent failure — fluent text, wrong positions
(`rope.metal:614-617`) — and the defense is structural: separate kernels
(`rope_norm_batch_cached` vs `rope_batch_cached`, `rope.metal:566`) rather
than a runtime flag, so no configuration can accidentally cross-wire the
two.

```
  head vector:   x0  x1  x2  x3 │ x4  x5  x6  x7          (head_dim = 8)
                                │
  INTERLEAVED (NORM / Muse):    │   HALF-SPLIT (NEOX / Llama-family):
  pair 0 : x0 ── x1             │   pair 0 : x0 ─────────── x4
  pair 1 : x2 ── x3             │   pair 1 : x1 ─────────── x5
  pair 2 : x4 ── x5             │   pair 2 : x2 ─────────── x6
  pair 3 : x6 ── x7             │   pair 3 : x3 ─────────── x7
```
*Figure 14.2: the two RoPE pairings. Same frequency table, same rotation,
different addressing. Match the checkpoint or corrupt position silently
(`rope.metal:610-618`).*

**Rotate-before-store, and rotate Q and K only.** RoPE runs before the KV
store so the cache holds *already-rotated* keys — attention then needs no
position argument beyond the mask, and the store kernel of
[Ch 15](15-kv-store-and-the-ring.md) is a pure copy. V is never rotated
(position lives in the Q·K score, not in the payload), which is why V's
plane is pure payload bytes on both layer classes.

**`precise::` trig vs fast-math trig.** The engine compiles its main
library with fast-math on for speed ([Ch 4](04-pso-and-three-kernel-sources.md))
but pays for exact trig here, per call. The cost is a handful of cycles on
2,176 threads × 39 layers — nanoseconds — and the benefit is that
`sin(pos·θ)` matches llama's own evaluation of the same angle to the bit.
There is no measured A/B in the ledger `[unverified]`; the choice is
contract-driven, like most precision decisions in this book.

**The QK-norm seam across the handoff.** The producer side (vLLM on the
GB10) materializes its weightless QK-norm in F16 and applies the scale as
a second F16 operation; a fused RMS+weight kernel "loses that intermediate
rounding point and is therefore not seam-exact"
(`metal/encode/norm.rs:280-284`). Under `MUSER_CROSS_VENDOR_QK` the
QK-norm therefore splits into two strict-f32 dispatches with a barrier
between — slower, deliberately, to preserve the producer's rounding. One
more example of the book's recurring rule: exactness is always exactness
*against a specific anchor*.

## 14.8 Where the gap lives

Every kernel chapter owes the same accounting answer: does this work show
up in the decode gap, or does it disappear into the noise of the layers
around it? Here the arithmetic settles it quickly.

**This kernel is not the gap.** Bytes per token: Q 4,096×4 + K 256×4 read
and written in place, ≈ 17.4 KB in and the same out, ×39 sliding layers ≈
1.4 MB of traffic — six orders of magnitude under the weight stream of
[Ch 13](13-the-qkv-gate-matvec-family.md). In the closure accounting of
`[docs/decode-dispatch-gap-20260815.md]`, the `qk_norm` and `rope` labels
sit in the common-math row (delta 0). The chapter's gap-adjacent fact is
the one already banked in [Ch 12](12-rmsnorm-and-the-dual-eps-sandwich.md):
the *norm-boundary* families (+104) are where fusions go to die on
exactness, and QK-norm's dispatch shape is one of the boundaries that
inherits that discipline.

## 14.9 What comes next

Q is rotated (on sliding layers), K likewise, V untouched — and the
current token's K and V must now be written into the cache that attention
will read. One engine, two storage regimes: a 2,048-slot ring for the 39
sliding layers, a growing head-major plane for the 13 NoPE layers. That is
[Ch 15](15-kv-store-and-the-ring.md).

## References

- `crates/muser-engine/src/shaders/ferrite/rope.metal:610-667` — the NORM-
  convention comment block and `rope_norm_batch_cached` (quoted).
- `crates/muser-engine/src/shaders/ferrite/rope.metal:566-608` —
  `rope_batch_cached`, the NEOX sibling kept deliberately separate.
- `crates/muser-engine/src/metal/encode/rope.rs:45-152` — the dispatch:
  cross-vendor NCO route, pinned `kernel_rope_norm_f32` route, and the
  ferrite fallback.
- `crates/muser-engine/src/decode.rs:5599-5641` — QK-norm and the
  SWA-gated RoPE call sites.
- `crates/muser-engine/src/decode.rs:1217-1270` — the frequency-table
  build (and the retained RoPE-cache file check).
- `crates/muser-engine/src/config.rs:51-71` — `MuseLayerKind`,
  `uses_rope()` and the `muse-glimmer.cpp:93` provenance.
- `crates/muser-engine/src/config.rs:84-102, 336-381` — the `il % 4 == 3`
  partition and the fail-closed pattern resolution.
- `crates/muser-engine/src/config.rs:139-141, 277-281, 383-403, 426-430` —
  `qk_scale_factor`, `attn_scale`, `QkNormProbe`, and the scale test.
- `crates/muser-engine/src/metal/encode/norm.rs:280-303` — `encode_qk_norm`
  and the vLLM F16 seam comment.
- `crates/muser-engine/src/shaders/ferrite/rms_norm_per_head.metal:15-47` —
  the per-head tree-reduction kernel (algorithm reference).
- `crates/muser-engine/src/lib.rs:7-15` — the position-free-NoPE / kvpack
  free-lunch summary.
- `crates/muser-engine/src/loader.rs:28-37` — the fail-closed QK-norm probe
  at load.
- [Ch 12](12-rmsnorm-and-the-dual-eps-sandwich.md) — the RMSNorm machinery
  QK-norm reuses.
- [Ch 15](15-kv-store-and-the-ring.md), [Ch 24](24-kvpack-the-format.md),
  [Ch 26](26-delta-handoff-and-migration.md) — where the NoPE/SWA
  asymmetry pays off.
- `[arxiv:2104.09864]` — Su et al., *RoFormer* (the RoPE origin paper).
- `[arxiv:1910.07467]` — RMSNorm (for §14.3's normalization).
- `[ferrite-book Ch 13]` — the ancestor's RoPE chapter (permutation-
  invariance motivation, (m−n) proof, convention warning — ported with
  Muse's NORM convention and dual-class twist).
