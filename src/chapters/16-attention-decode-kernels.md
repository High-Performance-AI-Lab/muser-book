# Chapter 16 — Attention: the decode kernel ladder
> **status:** polished  ·  **path:** Muse Glimmer, pinned Muser tree
>
> *Prerequisites: [Ch 2](02-metal-compute-model.md) (SIMD groups,
> `simd_sum`, barriers), [Ch 9](09-muse-glimmer-architecture.md) (GQA 32:2,
> the two layer classes), [Ch 14](14-qk-norm-and-rope.md) (rotated Q/K),
> [Ch 15](15-kv-store-and-the-ring.md) (the ring, the growing plane, and the
> explicit-origin arithmetic this chapter's kernels index with). Attention
> math is built from zero here; no transformer background is assumed.*

---

## 16.0 First: which kernel actually runs — the route ladder

The previous chapter left the keys and values sitting in the cache:
appended, addressable through explicit origins, and so far unread. This
chapter is where something finally reads them back. The natural way to
write it would be to open the engine, find the attention kernel, and walk
you through it line by line.

That is the one thing we cannot honestly do. There is no single attention
kernel on Muser's decode path. There are four routes, selected **per
layer, per token** by predicates computed after the ring `append` of
[Ch 15](15-kv-store-and-the-ring.md). Presenting any one of them as *the*
decode attention kernel would be this book's fastest way to lie to you,
and it is a lie with precedent: the ancestor book made exactly that
mistake once and had to correct it in public `[ferrite-book Ch 15]`. So
we start where the engine starts — with the code that chooses. Here is
the selection, verbatim, from the middle of `encode_token`:

```rust
// crates/muser-engine/src/decode.rs:5643
let write_physical = self.cache[layer_index].append(layer_index, position)?;
let plane = &self.cache[layer_index];
let strict_attention = std::env::var_os("MUSER_CROSS_VENDOR_QK").is_some();
let llama_vec_rows = (strict_attention || self.kernels.has_llama_flash_attn_vec())
    && plane.len > 0
    // The pinned vec kernel rounds KV reads to a 32-row block.
    // A deliberately tiny raw session can have a smaller backing
    // allocation, so taking the vec path would read past it and
    // poison the full distribution with NaNs.
    && plane.capacity >= 32
    && (plane.origin_physical == 0 || plane.len == plane.capacity);
// Token-major SWA cannot use llama's pad kernel (nb11 is a full
// token row, not one head). Only take that path when the window
// is a multiple of 32 so the vec kernel never pads.
let llama_swa = llama_vec_rows && plane.len.is_multiple_of(32);
```

And the four routes those predicates pick between:

```
  layer class      vec-eligible?           route
  ──────────────   ─────────────────────   ─────────────────────────────────────────────
  SWA (39)         llama_swa               kv_store_f16 → barrier → llama vec (pinned)
  SWA (39)         not llama_swa           kv_store_f16 → splitk producer → splitk reduce
  NoPE (13)        llama_vec_rows          kv_store_batch_f16 → barrier → llama vec (pinned)
  NoPE (13)        not llama_vec_rows      ferrite interleaved producer → reduce_v2
```

*Figure 16.0: the decode-attention route ladder. "Pinned" = a kernel from
the llama.cpp metallib of [Ch 4](04-pso-and-three-kernel-sources.md); the
others are Muser-owned (splitk) or ferrite-lineage (interleaved).* Decode
the predicates, because each clause is a lesson:

- **`has_llama_flash_attn_vec()`** — is the pinned metallib loaded? No
  library, no vec route; the ladder falls to Muser's own kernels.
- **`plane.len > 0`** — attention needs at least one visible row.
- **`plane.capacity >= 32`** — the pinned vec kernel *rounds its KV reads
  to a 32-row block*; a tiny diagnostic session with a smaller backing
  allocation would be read past its end, and the comment names the
  failure: "poison the full distribution with NaNs"
  (`decode.rs:5648-5651`). A fail-closed predicate against a silent OOB.
- **`origin_physical == 0 || len == capacity`** — the vec kernel reads a
  contiguous span of cache rows. An unwrapped plane (`origin_physical ==
  0`) is contiguous by construction; a *full* wrapped ring is also usable,
  because every slot is in-window and softmax is permutation-invariant
  (`metal/encode/attn.rs:431-435`). A partially-wrapped ring is not
  contiguous, so it never takes the vec path.
- **`llama_swa = … && plane.len % 32 == 0`** — one more SWA-only clause:
  the token-major SWA plane's row is a *full token row* (`nb11` is 256
  halves, not one head's 128), so llama's padding kernel cannot patch a
  ragged 32-row block; the vec path is taken only when the window length
  never pads (`decode.rs:5654-5657`).

One subtlety about that last-but-one clause, before we trust it too far:
reading a full wrapped ring through the vec kernel is *mathematically*
valid, but it is not *bit-identical* to llama's own SWA addressing — which
is why the serving batch graph, where bit-parity is the contract, stages
wrapped SWA rows into llama's absolute, 256-row-padded indices first, "so
the pinned vec kernel sees the same reduction lanes rather than a
mathematically equivalent compact permutation"
(`metal/encode/attn.rs:140-143`). The teacher-forced graph this chapter
narrates reads the ring directly. Two graphs, two answers to the same
question — the recurring Part IV pattern.

It is worth flagging that distinction now, before a single kernel: the
gap between *the same answer* and *the same floating-point answer* is the
hinge every route decision in this chapter turns on. Keep it in view; it
comes back as the sharpest tradeoff at the end.

## 16.1 What attention computes — from zero

We have just watched four routes argue with each other without saying
what any of them computes. They all compute the same thing, so it is
worth building that thing from nothing before we look at a kernel again.

Everything else in the transformer operates on **one** token's vector.
[Attention](../glossary.md#attention) is the one operation that reaches
sideways: while generating token *t*, it is the moment token *t* may look
at tokens `0..t` and mix information out of them. The folklore naming is
information retrieval: the **Query** (Q) is what this token seeks; each
past token's **Key** (K) is what it offers for matching; its **Value** (V)
is the payload returned on a match.

**Step 1 — the score is a dot product.** For query head `h`, current query
`Q_h ∈ ℝ¹²⁸`, and the key of past token `t` in the same head's group
`K_t[h] ∈ ℝ¹²⁸`:

```
score(t) = Q_h · K_t[h] = Σ_{d=0..127} Q_h[d] · K_t[h][d]
```

**Step 2 — scale by 1/√head_dim.** A sum of 128 random products has
standard deviation growing like √128; unscaled, it would push
[softmax](../glossary.md#softmax) into saturation where one token takes all
the weight. Muse Glimmer's scale is `1/√128 ≈ 0.0883883`
(`config.rs:277-281`; the companion test at `config.rs:426-430` also
checks it against the folded QK-norm factor). Note this is *in addition
to* the `qk_scale_factor ≈ 3.87` baked into the Q-norm weights
([Ch 14](14-qk-norm-and-rope.md)) — two scales, two places, asserted
apart.

**Step 3 — softmax over tokens.** With scaled scores `s_t`, the attention
weights are a probability distribution:

```
w(t) = exp(s_t) / Σ_{t'} exp(s_{t'})
```

**Step 4 — weighted sum of Values.** The head's output is
`out_h = Σ_t w(t) · V_t[h]`. All weight on token 3 → copy token 3's value;
uniform weight → the mean. In one line:

```
out_h = softmax_t( (Q_h · K_t[h]) / √128 ) · V_t[h]
```

### A fully worked example: head_dim = 2, three tokens

Smallest non-trivial case, every number by hand:

```
                    head_dim = 2, √head_dim ≈ 1.4142
  Query:    Q  = [1, 0]
  Keys:     K0 = [1, 0]   K1 = [0, 1]   K2 = [1, 1]
  Values:   V0 = [2]      V1 = [3]      V2 = [4]      (1-D values for readability)

  1) raw scores:      s0 = 1·1+0·0 = 1 ;  s1 = 1·0+0·1 = 0 ;  s2 = 1·1+0·1 = 1
  2) scaled ÷√2:      [0.7071, 0.0000, 0.7071]
  3) softmax:
       row max m       = 0.7071
       s_t − m         = [ 0.0000, −0.7071,  0.0000]
       exp(s_t − m)    = [ 1.0000,  0.4931,  1.0000]
       Z = Σ           = 2.4931
       weights = /Z    = [ 0.4011,  0.1978,  0.4011]      sums to 1 ✓
  4) output:
       out = 0.4011·2 + 0.1978·3 + 0.4011·4
           = 0.8022 + 0.5934 + 1.6044 = 3.0000
```

*Figure 16.1: attention end to end at head_dim 2. Q=[1,0] sees only each
key's first dimension, so K0 and K2 tie at the top; the output sits
between V0=2 and V2=4, pulled toward V1 by the middle weight.* Step 3's
`− m` line is the **max-subtraction trick**: subtracting the row maximum
before `exp` is algebraically exact (the `exp(−m)` cancels between
numerator and denominator) and numerically mandatory — `exp(200)` is
`+inf` in f32 and `inf/inf` is NaN. Every kernel below carries it.

## 16.2 Online softmax — the running-max table

The formula we just wrote has an inconvenient property: its denominator
sums over *every* score, so no weight can be finalized until the last
token has been seen. What does a kernel do about that? The literal answer
is to keep everything. Materializing all `visible` scores and then doing
softmax in a second pass (the ancestor's score-buffer design) costs a
buffer sized by context — comfortable at short depth, and a cost that
grows with exactly the thing you want to grow. The
flash/online formulation instead keeps three running quantities per worker
— max `M`, denominator `S`, accumulator `O` — and folds each new score in:

```
  new_M = max(M, s)
  corr  = exp(M − new_M)              ← rescale: old stats were normalized
                                        against the wrong (smaller) max
  S     = S·corr + exp(s − new_M)
  O     = O·corr + exp(s − new_M)·V
  M     = new_M
```

Run it on Figure 16.1's three scores in arrival order
`[0.7071, 0.0000, 0.7071]`:

```
  start:        M = −∞,  S = 0,  O = 0
  token 0:      new_M = 0.7071; corr = exp(−∞−0.7071) = 0
                S = 0·0 + 1.0000 = 1.0000
                O = 0·0 + 1.0000·2 = 2.0000
  token 1:      new_M = 0.7071 (unchanged); corr = exp(0) = 1
                S = 1.0000 + 0.4931 = 1.4931
                O = 2.0000 + 0.4931·3 = 3.4793
  token 2:      new_M = 0.7071 (unchanged); corr = 1
                S = 1.4931 + 1.0000 = 2.4931   ← matches Figure 16.1's Z
                O = 3.4793 + 1.0000·4 = 7.4793
  output:       O / S = 7.4793 / 2.4931 = 3.0000 ✓
```

*Figure 16.2: online softmax as a running table. The `corr` rescale fired
only on the first row (max moved from −∞); had a later token beaten 0.7071,
every earlier statistic would have been scaled down by `exp(old − new)` in
one multiply.* Two workers each running this over half the tokens merge
the same way — each is a "token" whose `(M, S, O)` combines with the
other's — which is how the split across SIMD groups and workgroups below
stays exact.

Say that the other way round, because it is the load-bearing idea of the
whole chapter. The triple `(M, S, O)` is a *complete summary* of every
token a worker has looked at; nothing else about those tokens is ever
needed again. So attention can be cut into arbitrary pieces, chewed in
any order, on any number of workers, and reassembled with no
approximation anywhere. That single property is what lets three different
kernels share one mathematics while disagreeing about nearly everything
else — and, as we will see, it is also what makes them disagree in the
last bits of the result.

## 16.3 GQA 32:2 — sixteen query heads per KV head

Muse Glimmer has **32 query heads and 2 KV heads** — grouped-query
attention at 32:2, i.e. `heads_per_kv = 16` (`config.rs:274-276`; geometry
from `muse_golden.rs:97-99`). Queries fan in; the cache does not:

```
  query heads:  Q0 … Q15        Q16 … Q31
                   └── KV head 0 ┘└── KV head 1 ┘
  K/V planes:        K0,V0              K1,V1      (2 planes, not 32)
```

*Figure 16.3: the 16:1 fan-in. Every kernel maps a query head to its KV
head by `kv_head = head / heads_per_kv` (`muse_reference.metal:1077-1078`,
`flash_attn_decode_vec_contiguous_f16.metal:519`).* The bandwidth win is
the point: the KV cache — and every attention read of it — is 16× smaller
than full multi-head attention at the same head count
`[arxiv:2305.13245]`. It is also why the store kernels of
[Ch 15](15-kv-store-and-the-ring.md) write 256-element rows (two heads'
worth), not 4,096.

## 16.4 SWA masking against the ring position

Which past tokens are *visible*? For a sliding layer at absolute position
`position` with window `W = 2,048`:

```
  visible      = min(position + 1, W)          (this token included)
  logical_start = position + 1 − visible        (the window's first token)
```

and each visible logical token maps to its ring slot through
[Ch 15](15-kv-store-and-the-ring.md)'s explicit origins:

```
  physical = (origin_physical + logical − origin_logical) % capacity
```

These three lines — not `position % capacity` — are how every Muser-owned
attention kernel addresses the cache; they appear verbatim inside the
splitk kernel below (`muse_reference.metal:1079-1102`). For a NoPE layer
the same code with `window = 0` degenerates to `visible = position + 1`,
`logical_start = 0` — the growing plane. *Masking* here is addressing:
tokens outside the window are never read, so they need no explicit
`−inf` mask. (Explicit masks do exist on Muser's paths — for the pinned
llama prefill kernel's causal blocks and the staged SWA decode route —
`metal/encode/attn.rs:266-317`; the decode ladder itself masks by
address.)

## 16.5 The kernels — three rungs, one math

The rungs come in teaching order here, not in the order the predicates
test them. Muser's own splitk pair goes first, because its source is in
the tree and it wears the online-softmax table on its sleeve. Then the
pinned llama kernel that outranks it wherever bit-parity is the contract.
Then the ferrite-lineage pair that catches NoPE layers when the metallib
is missing. Read the first rung as the reference implementation and the
other two as departures from it: everything that differs between them —
who owns the reduction order, who reads the current token, who can
address a wrapped ring — falls out of the situation each was built for.

### 16.5.1 The Muser splitk producer + reducer (SWA fallback)

This is the default SWA route whenever the pinned vec path is not
eligible, and the clearest exhibit of §16.1–16.2 in code. The producer's
grid is `(n_heads, n_workgroups)` and its threadgroup is
`(32, n_simdgroups)`:

```metal
// crates/muser-engine/src/shaders/muse_reference.metal:1052
kernel void muser_attention_decode_splitk_f16(
    device const float *query [[buffer(0)]],
    device const half *key_cache [[buffer(1)]],
    device const half *value_cache [[buffer(2)]],
    device float *partials [[buffer(3)]],
    constant uint &n_heads [[buffer(4)]],
    constant uint &n_kv_heads [[buffer(5)]],
    constant uint &position [[buffer(6)]],
    constant uint &capacity [[buffer(7)]],
    constant uint &origin_logical [[buffer(8)]],
    constant uint &origin_physical [[buffer(9)]],
    constant uint &window [[buffer(10)]],
    constant uint &n_workgroups [[buffer(11)]],
    constant uint &n_simdgroups [[buffer(12)]],
    constant float &attention_scale [[buffer(13)]],
    threadgroup float *shared [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    const uint head = group.x;
    const uint workgroup = group.y;
    if (head >= n_heads || simdgroup >= n_simdgroups) {
        return;
    }
    const uint head_dim = 128;
    const uint heads_per_kv = n_heads / n_kv_heads;
    const uint kv_head = head / heads_per_kv;
    const uint visible = window > 0 ? min(position + 1, window) : position + 1;
    const uint logical_start = position + 1 - visible;
    const uint block_count = (visible + 31) / 32;
    const uint vector_offset = lane * 4;
    if (simdgroup == 0) {
        *((threadgroup float4 *)(shared + vector_offset)) =
            *((device const float4 *)(query + head * head_dim + vector_offset));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float4 q = *((threadgroup float4 *)(shared + vector_offset));

    float running_max = -3.402823466e+38f;
    float running_sum = 0.0f;
    float4 accumulator = 0.0f;
    for (uint block = workgroup * n_simdgroups + simdgroup;
         block < block_count;
         block += n_workgroups * n_simdgroups) {
        const uint first_offset = block * 32;
        const uint count = min(32u, visible - first_offset);
        float scores[32];
        for (uint item = 0; item < count; ++item) {
            const uint logical = logical_start + first_offset + item;
            const uint physical =
                (origin_physical + logical - origin_logical) % capacity;
            const uint base = (physical * n_kv_heads + kv_head) * head_dim;
            const float4 key =
                float4(*((device const half4 *)(key_cache + base + vector_offset)));
            scores[item] = simd_sum(dot(q, key)) * attention_scale;
        }
        float block_max = scores[0];
        for (uint item = 1; item < count; ++item) {
            block_max = max(block_max, scores[item]);
        }
        const float next_max = max(running_max, block_max);
        const float old_factor = exp(running_max - next_max);
        accumulator *= old_factor;
        running_sum *= old_factor;
        running_max = next_max;
        for (uint item = 0; item < count; ++item) {
            const float weight = exp(scores[item] - running_max);
            running_sum += weight;
            // …(V pass: same logical→physical walk, accumulator += weight * V;
            //    elided — muse_reference.metal:1117-1127)…
        }
    }
    // …(threadgroup merge of the n_simdgroups partials and the per-workgroup
    //    [max, sum, weighted-V] write to `partials`; elided —
    //    muse_reference.metal:1130-1166)…
}
```

*(V-pass and merge elided as marked; the elided lines are the same
online-softmax steps against the value plane and the
Figure 16.2 merge.)* The anatomy:

- **Q staged once** — SIMD group 0 copies the head's 128-wide Q into
  threadgroup memory as `float4`s; every lane then holds one `float4`
  slice (`lane * 4`), reused for all 32-token blocks.
- **The score** — `dot(q, key)` is a 4-lane partial; `simd_sum` folds the
  32 lanes' partials into one score. 32 lanes × 4 = the whole 128-dim dot
  product per token.
- **Logical 32-token blocks, distributed round-robin** — block index
  `workgroup * n_simdgroups + simdgroup`, stepping by
  `n_workgroups * n_simdgroups`: each SIMD group owns every
  `(n_workgroups × n_simdgroups)`-th block, so every block is owned
  exactly once (asserted by test, `metal/encode/attn.rs:1001-1019`).
- **Online softmax per block** — block max, then the `old_factor` rescale
  of Figure 16.2, then weights and the V-accumulate.
- **Partials `[max, sum, weighted-V]`** — one per (head, workgroup),
  stride `2 + 128` floats (`attn.rs:742-743`).

Where do the workgroup and SIMD-group counts come from, and why cap them
at all? `splitk_geometry` (`attn.rs:888-896`) answers the first half:
blocks of 32 visible tokens, workgroups capped at
`MAX_DECODE_SPLIT_WORKGROUPS = 32`, SIMD groups growing 1→4 as visibility
demands. The second half is a fork we lost. The geometry inherited from
Ferrite put occupancy first and capped workgroups far above llama's fixed
launch, on the reasonable theory that more resident work hides more
latency. On the deep, growing planes it did the opposite: the extra
workgroups arrived with too little to chew on. The comment on the
constant records the result in the codebase's own words:

```rust
// crates/muser-engine/src/decode.rs:41
// llama.cpp's Metal `flash_attn_ext_vec` always launches `nwg = 32` and only
// grows simdgroups (1→4) once `2 * nwg * nsg * 32 < visible`. Ferrite's
// occupancy-first cap of 96 oversubscribed the 13 full/NoPE planes and is
// the depth-rent we lose to llama as context grows. Keep short-context
// `nwg = min(blocks, 32)` so TG512 does not pay empty workgroups.
pub(crate) const MAX_DECODE_SPLIT_WORKGROUPS: usize = 32;
```

Read that comment twice. One half is the setting we shipped and the
llama launch rule it now matches. The other half is a confession: the
earlier cap oversubscribed the deep planes, and what we still lose to
llama as context grows is named out loud, in the source, as rent. Nobody
had to write that second half down — a tidier codebase would have left it
in a commit message nobody reads. Muser's house style is that the deficit
lives next to the constant that causes it.

The reducer then finishes the job the producer deliberately left open.
Each workgroup handed up a partial, and the reducer merges those partials
with exactly the combine the producer used inside its own blocks:
`correction = exp(part[0] − global_max)`, then
`global_sum += part[1] * correction`, then accumulate the weighted
values, then divide — once, at the very end. That final single divide is
why the producer normalized nothing along the way: an early
normalization would only have to be undone. The code is
`muser_attention_decode_splitk_reduce_f32`, sibling of
`muser_attention_decode_splitk_reduce_f16`
(`muse_reference.metal:1169-1201`), dispatched at `attn.rs:776-783`. One
detail there is worth stealing for your own encoders: the barrier before
it is scoped to the partials buffer alone, "instead of stalling every
buffer used by the 52-layer command buffer" (`attn.rs:771-773`).

### 16.5.2 The pinned llama vec kernel — `kernel_flash_attn_ext_vec_f16_dk128_dv128`

Why would an engine that has a perfectly good attention kernel of its own
hand two of its four routes to somebody else's binary? Park the question
through the bullets; the answer arrives at the end of the section, and it
is not about speed.

The vec-eligible routes dispatch llama.cpp's own flash-attention decode
kernel from the pinned metallib. Like [Ch 13](13-the-qkv-gate-matvec-family.md)'s
matvec, the body is pinned binary provenance — not in the Muser tree, not
quoted here — and what Muser owns is a meticulously shaped dispatch
(`metal/encode/attn.rs:437-633`):

- **Strides describe the plane.** Head-major NoPE planes bind as
  `ns10 = 128` (one head row per KV row); token-major SWA rings as
  `ns10 = 256` (a full token row per KV row) — the two
  `GgmlMetalKargsFlashAttnExtVec` layouts at `attn.rs:503-511`.
- **Split-K by construction.** llama's kernel launches `nwg = 32`
  workgroups per head (`LLAMA_FA_NWG`, `encode.rs:1066`) and grows SIMD
  groups 1→4 the same way Muser's splitk does (`attn.rs:496-499`); each
  writes a partial, and llama's own `kernel_flash_attn_ext_vec_reduce`
  merges them (`attn.rs:616-630`).
- **Ragged visibility handled llama's way.** If `visible % 32 ≠ 0`, a
  `kernel_flash_attn_ext_pad` dispatch first rounds the tail
  (`attn.rs:525-559`) — which is exactly what the SWA `len % 32`
  predicate avoids needing on the token-major plane, where llama's pad
  kernel "would read past" the wrong-shaped row (`decode.rs:5654-5657`).

Now the parked question. The pinned-vs-own distinction is not performance
vanity — nobody clocked llama's kernel as faster and surrendered. It is
arithmetic identity. The parity ledger's Stage A close-out records two
findings that close the door together: llama's vec kernel "uses an
intentionally different reduction DAG" than any Muser/Ferrite kernel, and
"no untried llama scheduling transplant [was] compatible with the fixed
production hash" `[ledger, Stage A close-out]`. Read them side by side —
you cannot reschedule your way to llama's bits, and the hash that defines
production will not move to meet you. Matching llama's bits meant
adopting llama's kernels on the routes where llama runs them.

### 16.5.3 The ferrite interleaved fallback (NoPE)

The last rung catches the case neither of the others can: a NoPE layer
with no eligible vec route. Here the ladder falls to a ferrite-lineage
pair kept deliberately unmodified — the producer
`flash_attn_decode_vec_f16_gqa_interleaved`
(`shaders/ferrite/flash_attn_decode_vec_contiguous_f16.metal:494`) and
the reducer `flash_attn_decode_reduce_v2`
(`shaders/ferrite/flash_attn_decode_reduce_v2.metal:4`). The wrapper
states the terms of that inheritance in one line: "Exact Ferrite
a85048a90 cache-interleaved producer + LSE reducer for the growing NoPE
planes. These planes are head-major and never wrap; SWA rings remain on
Muser's explicit-origin kernel" (`attn.rs:185-187`). That is a division
of labour, not a preference: the ferrite pair is trusted exactly where
the plane's shape matches the assumption it was written under, and
nowhere else.

Its signature is the prelude ABI — function constants bake
`head_dim = 128` and the SIMD-group count into the pipeline at PSO build
(`encode.rs:798-835`). Its grid is `(n_heads, n_workgroups)`, launched so
that *sibling* Q heads land adjacent; its own header calls this a
"schedule-only interleaved sibling" (`…contiguous_f16.metal:487-493`),
and "schedule-only" is the load-bearing half of that phrase — the
interleaving changes which head runs beside which, never what any of them
computes. Each workgroup then merges its SIMD groups into one legacy
`[M, S, O]` partial, and `reduce_v2` combines those with the same
`precise::exp(p[0] − global_max)` correction of §16.2. (The
"LSE" in that quoted header is log-sum-exp — the `(M, S, O)` statistics of
§16.2 under their textbook name.)

This route also owns the **current-token KV bypass**: the producer takes
`k_cur`/`v_cur` as buffers 4–5, one simdgroup stores them into the plane
as a side effect (`…contiguous_f16.metal:534-543`), and — per the code's
own comment — "Every workgroup still reads the current token from
k_cur/v_cur": the freshest row is consumed as f32 from the activation
buffer, never round-tripped through the f16 plane. The pinned-vec and
splitk routes do **not** bypass: they store first and read the plane back
(the vec route with an explicit barrier between, `decode.rs:5669-5670`).
So the answer to "does Muser bypass the current token?" is *yes on the
ferrite rung only* — a property of which ladder step you stand on, not of
the engine.

## 16.6 The Rust dispatch — the ladder in one table

Everything above lands in four call sites in `encode_token`, one per
rung. Read the table as the ladder seen from the encoder — and read the
last column first, because that is where the previous chapter's store and
this chapter's read have to be ordered against each other, and each route
answers that differently:

| route | wrappers (file:line) | kernels | barrier between store and attention? |
|---|---|---|---|
| SWA vec | `encode_kv_store_f16` `attn.rs:635` + `encode_llama_flash_attn_decode_vec_f16` `attn.rs:437` | `muser_kv_store_f16` → pinned vec + pad + reduce | yes (`decode.rs:5669-5670`) |
| SWA fallback | store + `encode_attention_decode_splitk_f16` `attn.rs:708` | store → `muser_attention_decode_splitk_f16` → `…_reduce_f32` | barrier on partials (`attn.rs:774`) |
| NoPE vec | `encode_kv_store_batch_f16` `attn.rs:787` + vec wrapper | `muser_kv_store_batch_f16` → pinned vec + reduce | yes (`decode.rs:5746-5747`) |
| NoPE fallback | `encode_ferrite_attention_decode_interleaved_f16` `attn.rs:189` | ferrite interleaved (fused store) → `reduce_v2` | fused (`…contiguous_f16.metal:534-543`) |

That last column is the fused-versus-staged argument compressed into four
cells. Three routes store the row and then read it back, so they need an
ordering guarantee — a full barrier, or the narrow one scoped to the
partials. The ferrite rung needs neither, because the value never leaves
the kernel that produced it.

The vec wrapper call itself carries the whole story in its argument list —
`visible`, `capacity`, `origin_physical`, `head_major`, the pad and
partials scratch — `decode.rs:5671-5692` (SWA, `head_major = false`) and
`decode.rs:5748-5769` (NoPE, `head_major = true`).

## 16.7 The access pattern — when attention starts to matter

Now the question this chapter owes the bandwidth story: at what context
length does attention stop being a rounding error and become the thing
that costs? The derivation is short enough to do in full, so do it in
full rather than trusting anyone's intuition about it.

Per token per layer, the attention read is the visible window's KV:
`visible × n_kv_heads × 2 planes × head_dim × 2 B = visible × 1,024 B`
(the per-layer row cost of [Ch 15](15-kv-store-and-the-ring.md)). Derive
the shape across depth:

```
  SWA layers (window caps at 2,048):  39 × 2,048 × 1,024 ≈ 81.8 MB  (constant)
  NoPE layers at depth D:             13 × D × 1,024
  weights per token:                  16,756,681,056 B ≈ 16.76 GB    (constant)

  at D = 2,048:    KV ≈ 0.17 GB   ≈ 1 % of the weight stream
  at D = 32,768:   KV ≈ 0.44 GB + 0.08 GB ≈ 2.7 %
  at D = 131,072:  KV ≈ 1.74 GB + 0.08 GB ≈ 1.83 GB ≈ 10.9 %
  crossover with weights: 13 × D × 1,024 = 16.76 GB ⇒ D ≈ 1.26 M tokens
      — beyond the model's 131,072 limit
```

So unlike smaller models (the ancestor's 1.5 B crossed over near 65 K
context `[ferrite-book Ch 14]` — Ferrite-lineage arithmetic), **weights
dominate Muse Glimmer's decode at every context the model can hold**;
attention's KV read grows linearly but never overtakes within the limit.
That is derived arithmetic from the geometry, not a measurement — and it
is also why the splitk workgroup cap of §16.5.1 ("the depth-rent we lose
to llama as context grows", `decode.rs:41-46`) is about *latency and
occupancy at depth*, not about bytes: 13 NoPE layers × 131 K rows of
strided f16 is plenty to expose an under-parallel kernel even at 10 % of
the byte budget.

## 16.8 Tradeoffs

**A ladder, not a kernel — and the evidence for each rung.** The ancestor
faced this same question and got it wrong first: its position ladder was
ground-truthed by route logging and then corrected once
`[ferrite-book Ch 15]`. Muser's ladder is at least legible, since the
predicates are in source and quoted at the top of this chapter. Legible
is not the same as justified, though, so here is how the rungs earned
their places.

The fork came while the route question was still open and attention was
the prime suspect for the decode gap. The attractive move was to
*specialize*. Decode issues exactly one query, so a one-query GQA FA2
kernel ought to beat a general flash-attention kernel that is still
carrying prefill's machinery around with it. We wrote that kernel
and ran it on the streamed diagnostic, fully expecting the specialist to
win. It came in at 28.290 tok/s median against llama's 33.428 — a ratio
of 0.8463×, the specialist losing to the generalist it was built to
beat. We kept the measurement: it sits in the landed-and-rejected table
`[docs/decode-dispatch-gap-20260815.md §Landed and rejected reductions]`,
banked as evidence rather than shipped as a claim.

What the loss taught is that the kernel's *shape* was never the lever.
The lever was the reduction order — and the only way to get llama's
reduction order is to run llama's kernel. Transplanting the attention DAG
bit-exactly is what moved decode from 0.781× (single-sample, 2026-08-14)
toward the six-depth matrix above parity `[ledger, Arc 1]`. So the ladder
is neither a hedge nor an accident of history. It exists because *each
rung is the exact kernel for its situation*: llama's for parity-critical
contiguous reads, Muser's splitk for wrapped rings the pinned kernel
cannot address, ferrite's for the growing head-major planes without the
metallib.

**Wrapped-ring vec read vs staging — mathematical validity vs bit-parity.**
§16.0's full-ring clause is sound mathematics (permutation-invariant
softmax, `attn.rs:431-435`) and still not what serving does: the batch
graph stages wrapped SWA into llama's padded indices so "the pinned vec
kernel sees the same reduction lanes rather than a mathematically
equivalent compact permutation" (`attn.rs:140-143`). The 39 staging
groups that decision costs live in the gap accounting (§16.9). This is
the book's exactness-vs-equivalence distinction at its purest: two
routes with identical softmax outputs in exact arithmetic, one of which
reproduces llama's floating-point reduction order and one of which does
not. Put plainly, serving pays dispatch groups for an answer it already
had — because it needs that answer to arrive in llama's order, not
merely to be correct.

**Splitk's own kernel vs pinned-everything.** Why keep a Muser-owned
attention kernel at all when the metallib is loaded? Because the pinned
vec kernel cannot read a partially-wrapped token-major ring
contiguously — the predicate excludes it — and re-staging every decode
token (rather than every wrapped prefill chunk) would add a per-token
copy dispatch. The splitk producer walks the ring's explicit origins
natively (`muse_reference.metal:1100-1102`). The price is a different
reduction DAG from llama's on that rung — acceptable on the
teacher-forced/diagnostic graph, and one reason the serving route prefers
eligibility for the pinned kernel whenever the predicates allow.

**32-token blocks, 32-workgroup cap.** The block size matches the pinned
kernel's read granularity (`ncpsg = 32`, `encode.rs:1067`) and the
workgroup cap matches llama's fixed `nwg = 32` — with the in-source
admission that an earlier occupancy-first cap "oversubscribed the 13
full/NoPE planes and is the depth-rent we lose to llama as context grows"
(`decode.rs:41-46`). Owning the tradeoff in a constant's comment is this
codebase's house style; the rent itself is the kind of measured deficit
[Ch 40](40-what-we-measured-and-rejected.md) catalogs.

## 16.9 Where the gap lives

Which of this chapter's dispatches actually show up in the decode-gap
accounting, and could any of them be removed? Two families again, plus
one experiment. The **52 KV-publication splits** — store dispatch +
attention dispatch as separate closures in production — are
"session/publication structure; Keep" `[docs/decode-dispatch-gap-20260815.md]`:
combining the closures would not remove either kernel's math, and the
splits are what make the pinned-kernel-per-route discipline auditable.
The **39 SWA staging groups** are this chapter's wrapped-ring story —
staging exists *because* bit-parity with llama's reduction lanes outranks
the cheaper direct read (§16.8). And the attention-shaped row in the
landed-and-rejected table — the one-query GQA FA2 at 0.8463× — is the
measured trace of the route hunt that J0/J1 eventually resolved by
changing the anchor. Attention is not the gap's largest family; it is the
gap's most instructive one.

## 16.10 What comes next

`activations.attention` now holds 32 heads × 128 floats of weighted-past
mixture. Muse Glimmer does something unusual with it before the output
projection: multiplies it element-wise by `sigmoid(gate)`, where `gate` is
the fourth matvec of [Ch 13](13-the-qkv-gate-matvec-family.md) that has
been waiting in `activations.gate` all along. That sigmoid gate — and the
o_proj that follows it — is [Ch 17](17-sigmoid-gate-and-oproj.md).

## References

- `crates/muser-engine/src/decode.rs:5643-5792` — the route predicates
  (quoted) and all four dispatch branches.
- `crates/muser-engine/src/shaders/muse_reference.metal:1052-1167` —
  `muser_attention_decode_splitk_f16` (quoted in §16.5.1; V-pass and
  merge elided as marked).
- `crates/muser-engine/src/shaders/muse_reference.metal:1169-1201` — the
  splitk reducer.
- `crates/muser-engine/src/metal/encode/attn.rs:437-633` — the pinned vec
  wrapper (strides, ns10 split, pad, reduce).
- `crates/muser-engine/src/metal/encode/attn.rs:707-784` — the splitk
  wrapper and its scoped barriers; `:888-896` `splitk_geometry` (with the
  pinned-schedule test at `:976-983`).
- `crates/muser-engine/src/metal/encode/attn.rs:185-257` — the ferrite
  interleaved wrapper; `:102-183` the SWA staging kernels and their
  llama-lanes comment.
- `crates/muser-engine/src/metal/encode.rs:149-167, 1066-1072,
  1079-1249` — `LlamaFlashAttnPipelines`, `LLAMA_FA_NWG/NCPSG`, and the
  vec/pad/reduce pipeline construction.
- `crates/muser-engine/src/shaders/ferrite/flash_attn_decode_vec_contiguous_f16.metal:487-554`
  — the interleaved producer: header, fused store, and the k_cur/v_cur
  bypass comment.
- `crates/muser-engine/src/shaders/ferrite/flash_attn_decode_reduce_v2.metal:4-48`
  — the LSE reducer.
- `crates/muser-engine/src/decode.rs:41-46` —
  `MAX_DECODE_SPLIT_WORKGROUPS` and the depth-rent comment (quoted).
- `crates/muser-engine/src/config.rs:274-281` — `heads_per_kv`,
  `attn_scale`; `:426-430` the scale test.
- `[docs/decode-dispatch-gap-20260815.md]` — the closure families of
  §16.9 and the 0.8463× GQA-FA2 diagnostic row.
- `[ledger]` (`docs/goal-parity-ledger-2026-08.md`) — Arc 1 (0.781× → the
  six-depth matrix) and the Stage A close-out's pinned-kernel audit.
- `[docs/memory-footprint.md]` — the 1,024 B/row KV constant §16.7
  derives from.
- `[arxiv:1706.03762]` — Vaswani et al., *Attention Is All You Need* (the
  scaled dot-product formula).
- `[arxiv:2305.13245]` — Ainslie et al., *GQA* (the 16:1 bandwidth lever).
- [Ch 15](15-kv-store-and-the-ring.md) — the planes and origins this
  chapter indexes; [Ch 17](17-sigmoid-gate-and-oproj.md) — the gate that
  consumes this chapter's output; [Ch 36](36-prefill-vs-decode-paths.md)
  — the prefill-side attention routes.
- `[ferrite-book Ch 15]` — the ancestor's attention chapter (the
  worked-example and online-softmax devices ported; its position-ladder
  correction is the reason §16.0 leads this one).
