# Chapter 17 — The sigmoid gate and the output projection
> **status:** polished  ·  **path:** Muse Glimmer, pinned Muser tree
>
> *Prerequisites: [Ch 2](02-metal-compute-model.md) (threads, SIMD groups,
> `dispatch_threads`), [Ch 9](09-muse-glimmer-architecture.md) (the
> attention-output gate in the architecture), [Ch 12](12-rmsnorm-and-the-dual-eps-sandwich.md)
> (the sandwich norms and the fused dual-eps tail), [Ch 13](13-the-qkv-gate-matvec-family.md)
> (the pinned ggml matvec family — this chapter reuses it verbatim),
> [Ch 16](16-attention-decode-kernels.md) (what attention writes into
> `activations.attention`). Sigmoid is defined from zero here; no prior
> exposure is assumed.*

---

## 17.1 What it computes

[Ch 16](16-attention-decode-kernels.md) ended with attention writing one
vector: `activations.attention`, the concatenation of 32 head outputs of 128
elements each — `attn_dim = 32 × 128 = 4,096` floats
(`config.rs:268-270`). In most transformer families that vector goes straight
to the output projection. Muse Glimmer does something extra first: before the
attention result is allowed to rejoin the residual stream, the layer asks a
second, learned question about every one of those channels — *should this one
be heard at all?*

The answer arrives as another vector. A fourth
attention projection — the **gate**, sibling of Q, K, and V from
[Ch 13](13-the-qkv-gate-matvec-family.md) — produced a second 4,096-vector,
and this chapter's kernel multiplies the two element-wise:

```
attn_out[i]  ←  attn_out[i] · σ(gate[i])        for i in [0, 4096)
```

where **σ** is the logistic sigmoid, defined from zero:

```
σ(x) = 1 / (1 + e^(−x))
```

The sigmoid maps any real number into the open interval (0, 1):

| `x`     | `e^(−x)` | `σ(x)`  | meaning                       |
|--------:|---------:|--------:|-------------------------------|
| `−4`    | `54.6`   | `0.018` | almost fully closed           |
| `−1`    | `2.72`   | `0.269` | mostly closed                 |
| `0`     | `1.0`    | `0.500` | half open                     |
| `+1`    | `0.368`  | `0.731` | mostly open                   |
| `+4`    | `0.018`  | `0.982` | almost fully open             |

*Table 17.1: the sigmoid at five points. Positive gate values pass the
attention output through nearly unchanged; negative values suppress it; the
transition is smooth, never a hard zero.*

So the gate is a learned, per-channel volume knob on the attention result,
and it lives in attention-output space — `MuseConfig::attn_dim`'s doc comment
calls 4,096 "the space the attention-output gate lives in"
(`crates/muser-engine/src/config.rs:268-270`). After the gate, the
**output projection** (`o_proj`, the tensor `attn_output.weight`) maps the
gated 4,096-vector back into the 6,656-wide [residual
stream](../glossary.md#residual-stream-hidden-state):

```
delta = W_o · (attention ⊙ σ(gate))        W_o : [6656 × 4096], Q4_K
residual ← residual + post_norm(delta)     (the fused tail — §17.7)
```

Two operations, one chapter: a pointwise gate and a matvec you already know.
Keep an eye on the asymmetry between them, because it is the chapter's point.
The cheap operation is the one that turned out to need a careful safety
argument; the expensive one is the one with nothing to confess.

## 17.2 Why it exists — a gated attention output

There are two questions hiding in this section's title, and they have very
different evidence behind them. *What is the gate, mechanically?* is settled by
reading the tree. *Why did somebody train a model to want one?* is not in the
tree at all. Separating them is the whole discipline of this book, so we do it
out loud: the verifiable facts first, the honest shrug second.

The gate is part of the model contract, asserted
in the engine's own header — "Parameterless QK-RMSNorm, **sigmoid
attention-output gate**, Gemma-2-style sandwich norms"
(`crates/muser-engine/src/lib.rs:12`). It is a real learned projection with
its own tensor `blk.{l}.attn_gate.weight` of shape `[6656, 4096]`
(`config.rs:307`), and it rides in the same concurrent four-projection set as
Q, K, and V ([Ch 13](13-the-qkv-gate-matvec-family.md);
`decode.rs:5569-5598`). The CPU oracle applies it strictly between attention
and `o_proj`:

```rust
// crates/muser-engine/src/reference.rs:446
// ── sigmoid gate, then o_proj ─────────────────────────────────
for g in gate.iter_mut() {
    *g = 1.0 / (1.0 + (-*g).exp());
}
// … (capture elided) …
for (a, g) in attn_out.iter_mut().zip(gate.iter()) {
    *a *= *g;
}
matmul(
    &self.w(&format!("blk.{il}.attn_output.weight")),
    &attn_out,
    t,
    &mut proj,
);
```

Now the second question. **Why gate the attention output at all?** The
structural reading — the same
decoupling you will meet again in the FFN of [Ch 18](18-swiglu-ffn.md) — is
that the gate separates *whether* a channel's attention result flows onward
from *what* that channel carries. A sigmoid-squashed multiplier in (0, 1) can
near-zero a head-output channel that the layer has decided is noise for this
token, while passing useful channels through. But the model's own training
rationale is not in the Muser tree, and this book does not invent psychology:
**[unverified]** why the authors chose a sigmoid (rather than tanh or ReLU)
gate for Muse Glimmer specifically. What the code proves is the wiring, the
shape, and that skipping it is not an option — remove the gate and every
downstream bit changes, because the oracle order above is the parity
specification the Metal graph must reproduce (`reference.rs` is "the
correctness gate", `lib.rs:99-109`).

A second-order effect worth naming: because σ(x) > 0 always, the gate can
*damp* a channel but never invert it — and because the gate is applied before
`o_proj`, it scales each attention-head channel while they are still separate,
before the heads are mixed by `W_o`.

## 17.3 The operation, explained — a worked gate by hand

Before trusting a kernel, it is worth doing its job once by hand, at a size
small enough to check on paper. The gate makes that easy: it is as simple as
GPU work gets — element-wise, no reduction, no mixing, every output element
depending on exactly one attention value and one gate value. A hand example
with a 4-element slice of the two vectors:

```
  i :    0      1      2      3
  gate:  +2.0   −1.0   0.0    −4.0
  σ(g):  0.881  0.269  0.500  0.018

  attn:  1.50   2.00   0.25   8.00
  out:   1.32   0.54   0.125  0.14     ← attn[i] · σ(gate[i])
```

Every element is independent — element 3's attention value of 8.0 is large,
but its gate is nearly closed (σ(−4) = 0.018), so 98.2 % of it is suppressed.
That is the whole gate.

The operation that follows is where all the bytes are, and it asks nothing new
of you. The `o_proj` after the gate is the [matvec](../glossary.md#matvec) of
[Ch 13](13-the-qkv-gate-matvec-family.md) at a new shape — 6,656 output rows,
each a dot product over 4,096 inputs, which is `4096/256 = 16` Q4_K
super-blocks of 144 bytes = 2,304 bytes of weight per row. No new math; the
shapes, in Figure 17.1:

```
  gated attn [4096]        W_o [6656 × 4096] (Q4_K)          projected [6656]
  ┌           ┐   ┌──────────────────────────────┐   ┌            ┐
  │  4,096 f32│ × │ 6,656 rows × 2,304 B/row    │ = │ 6,656 f32  │
  └           ┘   └──────────────────────────────┘   └            ┘
    16 KiB read        15,335,136 B ≈ 15.34 MB          26 KiB write
                       (read once per layer)
```

*Figure 17.1: The gate (pointwise, 4096 wide) feeding the o_proj matvec
(4,096 → 6,656, Q4_K). The weight stream dominates: 15.34 MB against 42 KiB
of activation traffic.*

## 17.4 The Metal kernel — `sigmoid_gate_inplace`

How much Metal does a per-channel volume knob need? Less than the paragraph
that describes it. Here is the whole kernel, verbatim — it is nine lines and
worth reading as one piece before we take it apart:

```metal
// crates/muser-engine/src/shaders/ferrite/sigmoid_gate.metal:4
// Element-wise sigmoid gating: attn_out[i] *= sigmoid(gate[i])
// dispatch: (ceil(n/1024), 1, 1) × (1024, 1, 1)

kernel void sigmoid_gate_inplace(
    device       float* attn_out [[ buffer(0) ]],
    device const float* gate     [[ buffer(1) ]],
    constant     uint&  n        [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]])
{
    if (gid < n) {
        attn_out[gid] *= 1.0f / (1.0f + exp(-gate[gid]));
    }
}
```

Line by line:

- **Buffer 0, `attn_out`** — the attention result, and note it is `device
  float*`, *not* `const`. This kernel mutates its input in place: the gated
  value overwrites the ungated one. There is no third buffer.
- **Buffer 1, `gate`** — the gate projection's output, read-only.
- **Buffer 2, `n`** — the element count (4,096 on this model), bound as a
  4-byte inline constant.
- **`gid`** — the global thread index; each thread owns exactly one element
  `gid`, and the `if (gid < n)` guard covers the ragged tail when the grid
  rounds up past `n`.
- **The body** — literally the formula of §17.1: one `exp`, one add, one
  divide, one multiply-into-place. Per element: two global reads (`attn_out`,
  `gate`), one global write (`attn_out`).

That `*=` is the only subtlety, and §17.7 is about why it is safe.

Now back to the top of that listing, because its header comment is a small
trap and it is worth walking into deliberately once. We read
`dispatch: (ceil(n/1024), 1, 1) × (1024, 1, 1)` as a specification and went to
the Rust wrapper expecting to find threadgroups of that width being launched.
They are not there. `dispatch_1d` calls `dispatch_threads` with threadgroup
width `min(n, 256)` (`crates/muser-engine/src/metal/encode.rs:1337-1342`), so
the comment describes a dispatch shape the engine no longer uses. The detour
taught more than the discrepancy is worth on its own: the half of the comment
that constrains correctness — one thread per element, ragged-tail guarded —
holds either way, and the half that drifted is the half nothing checks. So we
flag it code-wins style and move on. A kernel header is a hypothesis about the
code, dated the day it was typed; we kept the pointer to both sides of this one
[crates/muser-engine/src/shaders/ferrite/sigmoid_gate.metal:5 versus
crates/muser-engine/src/metal/encode.rs:1337].

One digression before the wrapper, and it announces its own relevance: the
pattern it shows up in recurs in nearly every kernel chapter after this one.
This kernel has a twin. A strict-arithmetic sibling exists for the
cross-vendor comparison lane: `muser_cross_vendor_sigmoid_gate`
(`shaders/muse_reference.metal:680`) computes the same thing through the
no-fast-math library's controlled `expf` so a CUDA producer's bytes can be
matched bit for bit — same formula, different compile flags
(`gate.rs:14-15` selects it under `MUSER_CROSS_VENDOR_QK`). That is what an
exactness lane looks like throughout Muser: never a different algorithm,
always the same algorithm with the fast paths refused.

## 17.5 The Rust dispatch

The kernel is trivial, so the interesting question moves to the encoder: what
must the wrapper guarantee before it is allowed to launch anything? Two things.
The two vectors have to be the same length, or the element-wise product is
quietly meaningless; and the launch has to land between attention and `o_proj`
and nowhere else, because that is the order the CPU oracle fixed. The wrapper
is four statements:

```rust
// crates/muser-engine/src/metal/encode/gate.rs:7
pub fn encode_sigmoid_gate(
    &self,
    encoder: &ComputeCommandEncoderRef,
    values: &GpuBuffer,
    gate: &GpuBuffer,
) {
    debug_assert_eq!(values.len(), gate.len());
    if std::env::var_os("MUSER_CROSS_VENDOR_QK").is_some() {
        encoder.set_compute_pipeline_state(&self.cross_vendor_sigmoid_gate);
    } else {
        self.bind(encoder, "sigmoid_gate_inplace");
    }
    encoder.set_buffer(0, Some(values.metal()), 0);
    encoder.set_buffer(1, Some(gate.metal()), 0);
    set_value(encoder, 2, &(values.len() as u32));
    dispatch_1d(encoder, values.len());
}
```

In prose: bind the PSO (the strict twin only under the cross-vendor flag),
bind `values = activations.attention` and `gate = activations.gate` (both
4,096 floats — the `debug_assert` enforces the equal lengths the element-wise
product requires), push `n` as an inline constant, and launch one thread per
element via `dispatch_threads`, i.e. grid `(4,096, 1, 1)` with 256-wide
threadgroups and a guarded ragged tail (`encode.rs:1337-1342`).

Its call site in the token graph sits exactly where the oracle puts it — one
dispatch closure after the attention route, before `o_proj`:

```rust
// crates/muser-engine/src/decode.rs:5793
dispatch(command, |encoder| {
    self.kernels.encode_sigmoid_gate(
        encoder,
        &self.activations.attention,
        &self.activations.gate,
    );
});
self.project(
    command,
    &layer.output,
    &self.activations.attention,
    &self.activations.projected,
);
```

`self.project` (`decode.rs:5909-5917`) is the same wrapper every projection
uses: it routes by the loaded tensor's dtype to `encode_projection`
(`decode.rs:6044`), which for a Q4_K `attn_output.weight` and one token
dispatches the **pinned llama.cpp metallib kernel**
`kernel_mul_mv_q4_K_f32` — the exact kernel family of
[Ch 13](13-the-qkv-gate-matvec-family.md):

```rust
// crates/muser-engine/src/metal/encode/qkv.rs:429
if tokens == 1 {
    if let Some(pipeline) = self.ggml_matvec(dtype) {
        let (block_bytes, rows_per_group) = match dtype {
            GgmlType::Q4_K => (144, 2),
            // … (Q5_K (176, 1), Q6_K (210, 2) elided) …
        };
        let args =
            GgmlKargsMulMv::for_matmul(n_out, n_in, block_bytes, rows_per_group as i32);
        encoder.set_compute_pipeline_state(pipeline);
        set_value(encoder, 0, &args);
        encoder.set_buffer(1, Some(weights.metal()), weights.offset() as u64);
        encoder.set_buffer(2, Some(input.metal()), 0);
        encoder.set_buffer(3, Some(output.metal()), 0);
        let simdgroups = 2usize;
        encoder.dispatch_thread_groups(
            MTLSize::new(n_out.div_ceil(rows_per_group * simdgroups) as u64, 1, 1),
            MTLSize::new(32, simdgroups as u64, 1),
        );
        return;
    }
    // … (fallback to Muser's own muser_matvec_q4k_4r2s elided;
    //     reached only when the metallib is absent) …
}
```

The PSO comes from the pinned metallib (`encode.rs:278-280` registers
`ggml_q4k: kernel_mul_mv_q4_K_f32` and siblings). For `o_proj`,
`n_out = 6,656`, `rows_per_group = 2`, `simdgroups = 2`, so the launch is
**6,656 ÷ 4 = 1,664 threadgroups of 64 threads** (two SIMD groups, two rows
per group — every output row covered exactly once). The per-dtype table
`Q4_K → (144 B, 2 rows)`, `Q5_K → (176 B, 1)`, `Q6_K → (210 B, 2)` is the
same one [Ch 13](13-the-qkv-gate-matvec-family.md) introduced; the o_proj is
Q4_K on the release artifact (`attn_output 4096->6656 q4k`,
`crates/muser-bench/src/m16.rs:163-166`).

Put that another way, because it matters for everything the rest of the chapter
argues: the expensive half of this stage is not a Muser kernel. Muser computes
the shape arguments, binds three buffers, picks a grid — and then hands the
actual arithmetic to a pipeline state compiled from someone else's pinned
metallib. Whatever the engine can be blamed for here happens before
`set_compute_pipeline_state`, never inside the loop.

## 17.6 The access pattern — where the bytes go

Where does the time go in this stage? For every decode kernel in this book the
honest first answer is bandwidth rather than arithmetic, so the way to read a
stage is to count what it drags across the bus. Start with the cheap half. The
gate kernel, per layer per token:

```
  read  gate[4096]        16,384 B   (16 KiB)
  read  attn[4096]        16,384 B   (16 KiB)
  write attn[4096]        16,384 B   (16 KiB)
                       ─────────────────────
                        49,152 B   (48 KiB)
```

The `o_proj` matvec after it:

```
  read  W_o (Q4_K)       6,656 rows × 16 blocks × 144 B = 15,335,136 B ≈ 15.34 MB
  read  gated attn        16,384 B  (16 KiB)
  write projected[6656]   26,624 B  (26 KiB)   (f32; 6,656 × 4)
```

The arithmetic for the weight figure, shown so you can re-derive it: a row of
`W_o` spans `cols = 4,096` inputs; Q4_K packs 256 elements per 144-byte
super-block ([Ch 6](06-the-kquant-family.md)), so a row is
`4096/256 × 144 = 2,304 B`; there are `rows = 6,656` of them;
`6,656 × 2,304 = 15,335,136 B`. Across all 52 layers that is
`52 × 15,335,136 = 797,427,072 B ≈ 797 MB` of the pinned
16,756,681,056-byte artifact — **4.76 %** (arithmetic against the artifact
size asserted at `lib.rs:14`). For scale, the whole five-projection attention
block (Q + gate + K + V + O) is ≈ 48.4 MB per layer, while the FFN of
[Ch 18](18-swiglu-ffn.md) is 224–259 MB per layer — Figure 17.2 itemizes
the layer:

```
  per-layer weight read (kquant lane, derived arithmetic):
    attn_q      15.34 MB  (Q4_K)      ffn_gate  74.76 MB  (Q4_K)
    attn_gate   15.34 MB  (Q4_K)      ffn_up    74.76 MB  (Q4_K)
    attn_k       0.96 MB  (Q4_K)      ffn_down  74.76 MB  (Q4_K; 109.03 MB on
    attn_v       1.40 MB  (Q6_K)                 the Q6_K-down layers, Ch 19)
    attn_o      15.34 MB  (Q4_K)
    ──────────────────────            ─────────────────────────────
    attention  ≈ 48.4 MB              FFN      ≈ 224–259 MB
```

*Figure 17.2: The weight budget of one Muse Glimmer layer, from the
shape/dtype table of `crates/muser-bench/src/m16.rs:139-198`. The o_proj this
chapter covers is a mid-size slice; the FFN is ~4.6–5.3× the whole attention
block. The gate's own projection (`attn_gate`) costs as much as the Q
projection — gating is not free, it is another 15.34 MB stream.*

The pattern to internalize: **the gate's 48 KiB is rounding error against the
15.34 MB that follows it**. Like every decode kernel in this book, this stage
lives or dies on weight bandwidth, and the gate barely moves bytes.

## 17.7 Why the in-place `*=` is safe — ownership and ordering

The `attn_out[gid] *= …` is a read-modify-write on a shared buffer. Two
questions: can two threads collide *inside* the dispatch, and can another
dispatch read `attention` while the gate is still writing it? Both are worth
answering carefully, because a wrong answer does not crash — it produces a
plausible token, sometimes, on some runs, which is the most expensive kind of
bug this engine can have.

**In-dispatch: no collision by construction.** Each global thread index `gid`
touches exactly one element, `attn_out[gid]`, and no two threads share a
`gid`. The write set is partitioned as finely as it can be — one element per
thread. There is no reduction, no `simd_sum`, no `threadgroup_barrier`, and
none is needed. (Contrast the attention kernels of [Ch 16](16-attention-decode-kernels.md),
where many threads cooperate per head and partials must be combined.)

**Across dispatches: ordering is explicit and tracked.** The token graph runs
all 52 layers inside *one* command buffer on *one* concurrent compute encoder
— the source records the contract at the top of the token route:

```rust
// crates/muser-engine/src/decode.rs:5449
// One concurrent encoder owns the complete token. Graph dependencies
// are explicit barriers; independent projection groups share a barrier
// interval and may overlap, matching the accepted Ferrite/llama route.
```

Muser's buffers are allocated `StorageModeShared` *with* Metal's automatic
hazard tracking. That is a deliberate default, and we know it is deliberate
because the other branch of the fork was taken first. Untracked buffers promise
less driver bookkeeping between dispatches, and this graph already declares its
dependencies explicitly, so we expected the tracking to be redundant work that
could simply be switched off — free latency, no behaviour change. It was not
redundant. With tracking off, the engine "empirically changed DFlash
conditioning" — no kernel had been edited, and the observable behaviour moved
anyway — so the experiment was reverted. The allocator still carries the name
of the mode that survived
[crates/muser-engine/src/metal/buffer.rs, `shared_tracked`]. The lesson
generalizes well past this kernel, which is why it belongs in a chapter about
a nine-line gate: in a fail-closed engine, an optimization that changes bits
is not an optimization.

So the tracking stays, and a dispatch that reads `attention` after the gate's
write is ordered by the driver's
tracked-resource dependencies; where the graph needs finer control it issues
targeted `memory_barrier_with_resources` calls (you saw them around the KV
store in [Ch 16](16-attention-decode-kernels.md), `decode.rs:5669-5670`).
The o_proj that consumes the gated `attention` is therefore sequenced after
the gate, and the next layer's attention (which overwrites `attention`) after
that — a producer→consumer chain, which is the dominant hazard shape of the
whole decode graph ([Ch 35](35-ordering-hazards-and-the-dispatch-gap.md)
formalizes the taxonomy).

**What Muser deliberately does *not* do here** is fold the residual add into
the o_proj matvec itself. This is a real fork, and the ancestor took the
other branch of it: the Ferrite book taught exactly that
device — a `y[row] += dot` write-back in the matvec kernel
[ferrite-book Ch 16] — and it is a good device, cheaper by a dispatch and a
buffer. Muser's gate fuses nothing into the pinned
matvec. The o_proj writes to a scratch buffer, `activations.projected`, and
the residual add happens one dispatch later inside the fused dual-eps tail
(`encode_fused_norm_residual_rms_norm_32sg`, `decode.rs:5806-5818`),
where the *in-place* mutation actually lives: that kernel reads
`activations.normed` (the running residual), adds the post-normed projection
into it, and writes the result back in place.

Why give up the cheaper shape? The reason is the book's recurring one, and the
source states it outright: the legacy one-token graph
with its Ferrite-lineage fused kernels "diverges from the source-pinned
llama Metal graph enough to breach public logprob tolerance", so serving
routes one-token work through the batch graph that "dispatches the exact
pinned kernels" (`decode.rs:2085-2091`). Keeping the o_proj a *stock*
pinned-metallib matvec — unmodified write-back, `=` not `+=` — is what makes
its bytes match llama.cpp's. Turn that around and it becomes the sentence to
carry into the rest of the book: Muser buys exactness with dispatches. Every
fusion the ancestor took for free, this engine pays for at the encoder,
because it does not own the kernel it has to agree with. The fusion temptation
is paid for elsewhere
([Ch 19](19-downproj-and-residual.md) prices it).

## 17.8 Tradeoffs

Three forks meet at this stage. Two of them were the engine's to decide; the
third was decided by whoever trained the checkpoint, and knowing which is which
saves an afternoon of re-litigating a choice you cannot touch.

**In-place gate vs materialized sigmoid buffer.** The alternative — write
`σ(gate)` to a fresh buffer, then a separate multiply — would traffic
`16 + 16 + 16 + 16 + 16 = 80 KiB` per layer instead of 48 KiB, and add one
dispatch. The in-place form saves 32 KiB/layer ≈ 1.66 MB/token across 52
layers — real but tiny against the ~15.34 MB/layer o_proj weight stream, and
tiny again against the ~800 MB/token whole-model read. This is a "no
downside, take it" fusion of the pointwise kind: no reduction is moved, no
rounding changes (each element's arithmetic is identical; only the buffer
destination differs), so it carries none of the exactness risk the norm
fusions of [Ch 19](19-downproj-and-residual.md) do. [unverified] whether the
32 KiB/layer saving is individually measurable in end-to-end tok/s — no
retained A/B isolates this kernel, and at 0.002 % of per-layer bytes it would
be below noise by construction.

**A separate gate closure vs fusing the gate into attention or o_proj.** The
gate could in principle be folded into the attention kernel's epilogue
(compute `σ(gate[h·128+d])` while writing the head) or into the o_proj's
input load. The intuition pulls hard toward doing it: this book's whole gap
story is told in dispatch counts, and here is a dispatch that moves rounding
error and computes one exponential per channel. Muser keeps it a standalone
closure on every route anyway — teacher
forced (`decode.rs:5793-5799`), batched serving, and the packed decode group
(`decode.rs:5112-5116`, where per-row attention/gate buffers are gathered
into shared batch buffers and one gate serves all rows).

We went to the closure-count accounting expecting to find the gate implicated
somewhere, and it is not there. The measured
consequence is structural rather than a timing number: in the one-token
dispatch-gap accounting, the sigmoid-gate closures fall in the "common math
closures" family that is *identical* in both the production and legacy graphs
— 406 closures on each side, delta 0 [docs/decode-dispatch-gap-20260815.md,
closure-count table]. So fusing the gate would trade a bit-exact match against
the pinned graph for no reduction in the +196-closure
gap at all, on the route that actually serves traffic. It stays separate
because separate is exact.

**Gate before o_proj vs gate after o_proj.** The third fork was never ours to
take, and it is worth saying so before anyone spends a week on it. The model
applies the gate in
attention space (4,096 channels, pre-mixing) rather than in residual space
(6,656 channels). This is the checkpoint's choice, mirrored by the oracle at
`reference.rs:446-464`; an engine has no say in it. The consequence for the
engine is only that the gate kernel is 4,096 wide rather than 6,656 wide —
and that the gate projection `[6656 → 4096]` is one of the four concurrent
matvecs of [Ch 13](13-the-qkv-gate-matvec-family.md), not a fifth sequential
one.

## 17.9 Where the gap lives

Every kernel chapter in this part owes the same answer, and the question
behind it is one of suspicion: does this stage help explain the extra dispatch
closures the campaign is hunting? Here the answer is short, and it is the
boring one.

**This kernel is not the gap.** Both of this chapter's operations appear in
the "Common math closures (including LM head/softcap)" row of the corrected
closure-count diff — 406 production versus 406 legacy, delta 0
[docs/decode-dispatch-gap-20260815.md]. The one-token graphs differ in *norm
boundaries, SWA staging, KV publication, and one copy* — none of them here.
The o_proj streams 15.34 MB/layer through the same pinned llama.cpp matvec
the comparator itself runs ([Ch 13](13-the-qkv-gate-matvec-family.md)), so
there is no engine-specific bandwidth story to tell about it either. The
gate's 48 KiB is four orders of magnitude below the o_proj's weights. When
the dispatch-gap chapters ([Ch 35](35-ordering-hazards-and-the-dispatch-gap.md),
[Ch 40](40-what-we-measured-and-rejected.md)) hunt the +196, this stage is
already exonerated.

The attention half of the layer is closed: context was gathered, gated, and
mixed back toward the residual stream. What remains of the layer is the part
with most of the bytes — the feed-forward block, two 74.76 MB Q4_K streams
and a learned gate of its own. [Ch 18](18-swiglu-ffn.md) teaches the FFN from
zero.

---

## References

- `crates/muser-engine/src/shaders/ferrite/sigmoid_gate.metal:7-16` —
  `sigmoid_gate_inplace`, the whole kernel (this chapter's primary source;
  the header dispatch comment at `:5` is stale versus the wrapper —
  `dispatch_1d` uses `dispatch_threads`, `encode.rs:1337-1342`).
- `crates/muser-engine/src/metal/encode/gate.rs:7-23` —
  `encode_sigmoid_gate`, the Rust dispatch (cross-vendor twin selected at
  `:14-15`).
- `crates/muser-engine/src/shaders/muse_reference.metal:680-688` —
  `muser_cross_vendor_sigmoid_gate`, the strict no-fast-math sibling.
- `crates/muser-engine/src/decode.rs:5793-5805` — the gate dispatch and
  `o_proj` call in `encode_token`; `:5449-5451` the one-encoder contract;
  `:2085-2091` the serving-versus-legacy exactness comment;
  `:5112-5116` the packed decode group's shared gate.
- `crates/muser-engine/src/decode.rs:6044-6091` — `encode_projection`, the
  dtype router; `crates/muser-engine/src/metal/encode/qkv.rs:429-450` the
  one-token pinned-metallib matvec path; `:451-459` the
  `muser_matvec_q4k_4r2s` fallback.
- `crates/muser-engine/src/metal/encode.rs:278-280` — `ggml_q4k/q5k/q6k`
  PSO registration against the pinned llama.cpp metallib;
  `:1337-1342` `dispatch_1d`.
- `crates/muser-engine/src/config.rs:266-273` — `attn_dim`/`kv_dim`
  ("the space the attention-output gate lives in");
  `:300-314` the per-layer tensor shape contract including
  `attn_gate.weight [h, attn]` and `attn_output.weight [attn, h]`.
- `crates/muser-engine/src/lib.rs:7-15` — the model-facts header asserting
  the sigmoid attention-output gate.
- `crates/muser-engine/src/reference.rs:446-464` — the CPU oracle's
  gate-then-o_proj order (the parity specification).
- `crates/muser-bench/src/m16.rs:139-198` — the per-projection
  shape/dtype table (`attn_q/gate 6656->4096 q4k`, `attn_v-q6k 6656->256`,
  `attn_output 4096->6656 q4k`) backing §17.6's budget.
- [docs/decode-dispatch-gap-20260815.md] — the closure-count reconciliation;
  this stage is common math (406 = 406).
- [Ch 13](13-the-qkv-gate-matvec-family.md) — the pinned ggml matvec family
  this chapter reuses; [Ch 16](16-attention-decode-kernels.md) — what
  produced `activations.attention`.
- [Ch 12](12-rmsnorm-and-the-dual-eps-sandwich.md) — the fused dual-eps tail
  that consumes this chapter's `projected` output.
- [Ch 6](06-the-kquant-family.md) — the Q4_K 144-byte super-block behind the
  15.34 MB arithmetic.
- [Ch 35](35-ordering-hazards-and-the-dispatch-gap.md) — the hazard taxonomy
  behind §17.7's ordering argument.
- [ferrite-book Ch 16] — the ancestor's residual-fused o_proj matvec; the
  device Muser deliberately does not port onto the pinned kernel (§17.7).
