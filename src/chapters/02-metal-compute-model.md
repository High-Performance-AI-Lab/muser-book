# Chapter 2 — The Metal compute model
> **status:** polished  ·  **path:** Muse Glimmer, pinned Muser tree
>
> *Prerequisites: [Ch 1](01-why-inference-is-a-memory-problem.md). You know
> why decode streams ~16.76 GB per token. This chapter assumes you have
> never written a GPU shader.*

---

## 2.1 What a GPU actually is

Chapter 1 ended on a prerequisite: everything after that page happens on the
GPU, and following it means speaking Metal — devices, command buffers,
threads, threadgroups, SIMD groups. This chapter teaches that language from
zero. A CPU is good at doing one hard thing fast. A GPU is good at doing ten
thousand easy things at the same time. Inference is, almost entirely, ten
thousand easy things: multiply this row of weights by this vector, add up
the products, do it again for the next row. That is why the GPU does the
work, not the CPU.

The "ten thousand easy things" are called **[threads](../glossary.md#thread)**.
A thread is one execution lane: it runs your program once, sees its own
small slice of the data, and writes its own small slice of the answer.

But a GPU is not a magical parallel computer where every thread is
independent. Threads are organized, and the organization is what makes a
kernel fast or slow. On Apple Silicon the organization that matters most is
the **[SIMD group](../glossary.md#simd-group)**, which we meet in §2.6.
First, the API.

## 2.2 Metal: Apple's GPU API

**[Metal](../glossary.md#metal)** is Apple's API for talking to the GPU. It
is to Apple GPUs what CUDA is to NVIDIA GPUs: the vendor's own first-class
compute path. If you want compute work done on an Apple Silicon Mac, Metal
is that path.

Metal gives you three things:

1. A **shading language** (MSL, Metal Shading Language) — a C++ dialect in
   which you write the code each thread runs. That per-thread program is
   called a **[kernel](../glossary.md#kernel)**: a single function, written
   once, that every thread in a launch runs once over its own slice of the
   data. Muser's kernels live in `crates/muser-engine/src/shaders/`, and
   the census there is small enough to say out loud: 29 `.metal` files,
   of which two are Muser-authored and 27 sit in the `ferrite/` lineage
   directory, plus one bench-only candidate shader under
   `crates/muser-bench/`. Where an inherited kernel came from is recorded
   rather than remembered; the extraction manifest is retained
   `[docs/extraction-manifest.md]`.
2. A **host API** (Objective-C underneath, wrapped by the Rust crate
   `metal`) — the code the CPU runs to compile kernels, allocate memory,
   and submit work.
3. A **memory model** — on Apple Silicon, a single physical DRAM pool shared
   by CPU and GPU. [Ch 3](03-unified-memory-and-buffers.md) is devoted to
   it.

The single most important mental model for the host API is the next
section.

## 2.3 "Record a tape, then press play"

The CPU does **not** tell the GPU "do this now." Instead the CPU *records*
a sequence of instructions onto an object called a
**[command buffer](../glossary.md#command-buffer)**, and then hands the
whole buffer to the GPU in one shot. Think of recording a cassette tape:
you can record many songs onto it, in order, and only when you press play
does anything actually happen `[Metal-PG, "Command Buffers"]`.

The object that records the tape is the **[compute command
encoder](../glossary.md#compute-command-encoder)**. You ask it to do four
things, over and over:

```text
bind the kernel you want to run             (set_compute_pipeline_state)
bind the memory buffers it will read/write  (set_buffer, repeated)
bind any small constants                    (set_bytes)
launch N copies of the kernel               (dispatch_thread_groups)
```

That four-line sequence is **one dispatch** — one kernel running once
across many threads. A Muser decode token is a dozen-plus dispatches per
layer across the model's **52 layers**, plus a head and tail — hundreds of
dispatches in all, recorded onto **one command buffer per token**, then
played. The per-group counts are reconciled and measured in
[Ch 35](35-ordering-hazards-and-the-dispatch-gap.md); here you only need
the shape.

Here is the CPU side, verbatim — `forward_token`, the function that owns
one whole decode token:

```rust
// crates/muser-engine/src/decode.rs:5448
let command_buffer = queue.new_command_buffer();
// One concurrent encoder owns the complete token. Graph dependencies
// are explicit barriers; independent projection groups share a barrier
// interval and may overlap, matching the accepted Ferrite/llama route.
let serial = GraphEncoder::concurrent(
    command_buffer
        .compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent),
);
self.encode_token(&serial, &token_view, self.n_past)?;
serial.encoder.end_encoding();
command_buffer.commit();
self.context
    .wait_for_completion(command_buffer, Duration::from_secs(300))?;
```

Steps, in order: get a blank tape from the queue; get a recorder with
*concurrent* dispatch semantics (§2.9); record the entire token with
`encode_token` (the 52-layer graph — hundreds of `encode_*` calls, each of
which records one or a few dispatches); stop recording; press play
(`commit`); wait — bounded, in Muser, by a 300-second deadline rather than
an unbounded block. Recording is pure CPU bookkeeping measured in
microseconds; the wall-clock time is paid between `commit` and the wait
returning.

One deliberate difference from the ancestor Ferrite book's engine is worth
a paragraph, because it is the reason the cost model you just learned is
the *only* one you need: Muser does not have a `dispatch_*` family of
test-only wrappers that each pay a full commit-and-wait round trip. In an
engine that has them, a kernel can be submitted two ways — the cheap
shared-tape way on the hot path and the expensive one-tape-per-kernel way
in tests — and the two drift. Here, tests hand-roll the same five steps
inline when they need them; the multi-column kernel tests are the example
to read `[crates/muser-engine/src/metal/encode/multicol.rs:373-390]`.
Everything on the hot path is an `encode_*` call onto a shared encoder,
and there is no second submission shape to keep in your head.

## 2.4 The handles: one struct, five GPU objects

Which pieces of GPU state deserve to be created once and kept forever, and
what does getting that wrong cost? The question is not academic: opening a
device, building a command queue, and compiling a library of shaders are
all slow enough that doing any of them per token would swamp the token.
Muser's answer is to hoard exactly the handles that are expensive to make
and cheap to share, in one struct, `MetalContext`:

```rust
// crates/muser-engine/src/metal/context.rs:32
pub struct MetalContext {
    pub device: Device,
    pub queue: CommandQueue,
    pub library: Library,
    /// Strict-f32 copy of the standalone Muse kernels.  The cross-vendor
    /// Q8 projection and integer NVFP4 routes must match CUDA's explicit
    /// scalar boundaries, while the ordinary serving kernels retain fast math.
    pub cross_vendor_library: Library,
    pub ggml_library: Option<Library>,
    pub ggml_library_path: Option<PathBuf>,
}
```

Walk the fields. The Rust types come from the `metal` crate: `Device`
wraps `MTLDevice`, `CommandQueue` wraps `MTLCommandQueue`, `Library`
wraps `MTLLibrary`.

- **`device`** (`MTLDevice`) — the handle to the GPU. On Apple Silicon
  there is exactly one system-default device:
  `Device::system_default()` at `[crates/muser-engine/src/metal/context.rs:46]`.
  You ask it to allocate memory and to compile shaders.
- **`queue`** (`MTLCommandQueue`) — the queue that accepts command buffers,
  created once at startup: `device.new_command_queue()`
  (`[crates/muser-engine/src/metal/context.rs:47]`). Every command buffer
  comes off this queue.
- **`library`** (`MTLLibrary`) — a bundle of compiled kernel functions,
  addressable by name. This one is built at startup from a concatenation
  of 24 `.metal` source files, with fast math on
  (`[crates/muser-engine/src/metal/context.rs:59-110]`). [Ch 4](04-pso-and-three-kernel-sources.md)
  covers *how*, including why there are three libraries where a simple
  engine would have one.
- **`cross_vendor_library`** — the same two source files recompiled with
  fast math *off*, so a handful of kernels match CUDA's arithmetic
  boundaries bit for bit when the remote-producer lane needs them
  (`[crates/muser-engine/src/metal/context.rs:111-121]`).
- **`ggml_library`** — an optional third library loaded from a prebuilt
  llama.cpp `.metallib` (a kernel library serialized to disk; [Ch 4](04-pso-and-three-kernel-sources.md)
  is its chapter) when `MUSER_GGML_METALLIB` is set — the pinned
  upstream kernels Muser dispatches for numerical parity
  (`[crates/muser-engine/src/metal/context.rs:122-131]`).

There is a timing rule hiding in that list, and it is the part to carry
forward. At *construction* time, every kernel is looked up by name and
compiled once into a cached pipeline state — the `MetalKernels`
constructor and its `PIPELINES` registry of 66 names
`[crates/muser-engine/src/metal/encode.rs:21-88]`, which is
[Ch 4](04-pso-and-three-kernel-sources.md)'s subject again. At *dispatch*
time you never touch a library at all. You reach for the pipeline the
registry already holds, and it costs a line:
`self.bind(encoder, "sigmoid_gate_inplace")`
`[crates/muser-engine/src/metal/encode/gate.rs:17]`.

Ask the registry for a name nobody put in it and the process dies on the
spot: `PsoCache::get` refuses to return silently for an unregistered name
`[crates/muser-engine/src/metal/pso_cache.rs:45-49]`. That loudness is on
purpose — a missing pipeline is a programming error, never a runtime
condition — and the tradeoffs section returns to why the crash is the
kinder outcome.

These five objects are the only long-lived GPU state. Everything else —
buffers, command buffers, encoders — is created per-use or per-token
([Ch 3](03-unified-memory-and-buffers.md) for the buffers).

## 2.5 One queue, one owner

When several sequences are decoding at once, who is allowed to talk to the
GPU, and in what order? Metal answers half of that: command buffers from
one queue run in FIFO order, so the GPU itself will not interleave two
tokens' work. It does not answer the other half. Several resident
sequences can still stampede the queue from many threads, and FIFO then
means only "whoever got there first, repeatedly" — a hot slot can starve
its peers while staying perfectly ordered. Muser's answer is a single
owner:

```rust
// crates/muser-engine/src/decode.rs:1020
/// One owner for the shared Metal queue. Decode work is selected first and
/// resident sequence IDs rotate in ascending cyclic order, preventing a hot
/// slot from repeatedly reacquiring the accelerator ahead of its peers.
struct AcceleratorScheduler {
    state: Mutex<AcceleratorSchedulerState>,
    ready: Condvar,
}
```

The `AcceleratorScheduler` (`[crates/muser-engine/src/decode.rs:1023-1026]`)
is a Mutex+Condvar gate: `acquire(sequence_id, work)` blocks until the
accelerator is free and this sequence is the chosen next one — decode work
is selected before prefill, and decode sequences rotate in ascending
cyclic order (`[crates/muser-engine/src/decode.rs:1040-1059]`). Every graph
— every tape — is recorded and committed while holding an
`AcceleratorPermit`. The shared execution resources live next to it, in
`MetalShared`:

```rust
// crates/muser-engine/src/decode.rs:954
/// Immutable Metal execution resources shared by every resident sequence.
/// Metal command submission is scheduler-serialized; retaining one context,
/// pipeline set, mapped weight arena, and GPU vector set avoids loading the
/// 16+ GiB target once per serving slot.
pub struct MetalShared {
```

That comment is [Ch 1](01-why-inference-is-a-memory-problem.md)'s capacity
argument made structural: four serving slots share *one* context, *one*
pipeline set, *one* mapped 16.76 GB weight arena, and *one* scheduler. The
per-sequence state (KV planes, activations, position) lives elsewhere, in
`MetalMuseModel` (`[crates/muser-engine/src/decode.rs:989-998]`).

## 2.6 Threads, threadgroups, and the SIMD group

This is the part that trips people up. There are **three** nested units of
parallelism on an Apple GPU, and you must understand all three to read
Muser's dispatch sizes.

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  GRID  = the entire launch (what dispatch_thread_groups fixes)          │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐         (n threadgroups)  │
│  │ threadgroup│ │ threadgroup│ │ threadgroup│ ...                      │
│  │ ┌────────┐ │ │ ┌────────┐ │ │            │                          │
│  │ │SIMD grp│ │ │ │SIMD grp│ │ │            │   each SIMD group =      │
│  │ │ 32 lanes│ │ │ 32 lanes│ │ │            │   32 threads in lockstep │
│  │ └────────┘ │ │ └────────┘ │ │            │                          │
│  │ ┌────────┐ │ │            │ │            │                          │
│  │ │SIMD grp│ │ │            │ │            │                          │
│  │ └────────┘ │ │            │ │            │                          │
│  └────────────┘ └────────────┘ └────────────┘                          │
└─────────────────────────────────────────────────────────────────────────┘
```
*Figure 2.1: The three nested units. A grid of threadgroups, each
threadgroup of SIMD groups, each SIMD group of 32 lockstep threads.*

- **Thread** — the unit that runs the kernel once. Each thread has a
  unique `thread_position_in_grid` so it can pick its own slice of the
  data. Nothing is promised about the *order* threads run in.
- **[Threadgroup](../glossary.md#threadgroup)** — a block of threads
  (32–1024) that share on-chip **[threadgroup memory](../glossary.md#threadgroup-memory)**
  and can synchronize with each other using
  `threadgroup_barrier(...)`. A threadgroup is the unit of co-scheduling:
  all its threads live on one compute unit together.
- **SIMD group** *(Apple-specific, the important one)* — exactly **32
  threads that execute in lockstep** on one SIMD ALU. All 32 lanes run the
  same instruction at the same instant. The superpower: lanes within a
  SIMD group can exchange data in **one cycle** via intrinsics like
  `simd_sum(x)` (a 32-way sum) and `simd_shuffle(val, lane)`. A reduction
  that would take `log₂(32) = 5` barrier-separated passes between
  independent threads takes one `simd_sum` inside a SIMD group.

Here is a real Muser kernel that uses all three units — `rms_norm_batch`,
the batched RMSNorm reduction (RMSNorm itself is
[Ch 12](12-rmsnorm-and-the-dual-eps-sandwich.md)'s subject; here it is
just "reduce one row of the residual stream to one scalar, then scale"):

```metal
// crates/muser-engine/src/shaders/ferrite/rmsnorm_batch_tail.metal:1
kernel void rms_norm_batch(
    device const float* x      [[ buffer(0) ]],  // [B × n]
    device const float* weight [[ buffer(1) ]],  // [n] (shared)
    device       float* out    [[ buffer(2) ]],  // [B × n]
    constant     uint&  n      [[ buffer(3) ]],
    constant     float& eps    [[ buffer(4) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint tid [[ thread_index_in_threadgroup ]],
    uint sgitg [[ simdgroup_index_in_threadgroup ]],
    uint lid [[ thread_index_in_simdgroup ]],
    threadgroup float* shared [[ threadgroup(0) ]])
{
    const uint batch = tgid;
    device const float* xb  = x   + batch * n;
    device       float* ob  = out + batch * n;
    device const float4* xb4 = (device const float4*)xb;
    device const float4* wb4 = (device const float4*)weight;
    device float4* ob4 = (device float4*)ob;
    const uint n4 = n >> 2u;

    float sum_sq = 0.0f;
    for (uint i = tid; i < n4; i += 128u)
        sum_sq += dot(xb4[i], xb4[i]);
    sum_sq = simd_sum(sum_sq);
    if (lid == 0u) shared[sgitg] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u)
        shared[4] = rsqrt((shared[0] + shared[1] + shared[2] + shared[3]) / float(n) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = shared[4];
    for (uint i = tid; i < n4; i += 128u)
        ob4[i] = xb4[i] * inv_rms * wb4[i];
}
```

Read it with the vocabulary. Each threadgroup handles **one row** of the
batch (`tgid`), and the launch is 128 threads = **4 SIMD groups**. The
`[[ ]]` attributes are MSL's way of receiving precomputed coordinates:
`tgid` = which threadgroup (which row), `tid` = thread 0..127 within it,
`sgitg` = which of the 4 SIMD groups, `lid` = lane 0..31 within the SIMD
group. The reduction inside is the classic two-stage pattern:

1. Each thread accumulates partial sums of squares over `float4` (4-wide)
   loads, striding by 128 — the strided loop that shares work across the
   threadgroup.
2. `simd_sum(sum_sq)` collapses each SIMD group's 32 partials to one value
   in a single instruction.
3. Lane 0 of each SIMD group writes its group's value into **threadgroup
   memory** (`shared[sgitg]`).
4. `threadgroup_barrier` — every thread in the threadgroup waits until all
   writes to threadgroup memory are visible. *This is the thread-level
   barrier; it is the only way to synchronize across SIMD groups.*
5. Thread 0 combines the 4 group values, computes the inverse root mean
   square, parks it at `shared[4]`.
6. A second barrier, then every thread re-reads `inv_rms` and scales its
   slice of the row.

Say the same thing the other way round, because this asymmetry is the one
idea that shapes every kernel in the rest of the book: inside a SIMD group,
threads talk to each other for free, in a single instruction. Between SIMD
groups, they cannot talk at all — they can only leave messages in
threadgroup memory and agree, via a barrier, on when it is safe to read
them. Two levels of communication, two very different prices. A kernel
that does its reducing inside SIMD groups and its combining across them
pays the expensive price as few times as possible; that is what the
two-stage pattern above is *for*, and it is why the geometry of a launch is
never an arbitrary choice.

The Rust side that launches it:

```rust
// crates/muser-engine/src/metal/encode/norm.rs:270
self.bind(encoder, "rms_norm_batch");
encoder.set_buffer(0, Some(input.metal()), 0);
encoder.set_buffer(1, Some(weight.metal()), 0);
encoder.set_buffer(2, Some(output.metal()), 0);
set_value(encoder, 3, &(dim as u32));
set_value(encoder, 4, &eps);
encoder.set_threadgroup_memory_length(0, 32);
encoder.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(128, 1, 1));
```

In prose: grid = `rows` threadgroups (one per row of the batch), threadgroup
= 128 threads (4 SIMD groups of 32), and 32 bytes of threadgroup memory
(`shared[0..4]`). Note `set_threadgroup_memory_length` on the host side
matching `threadgroup(0)` in the kernel — the buffer-slot contract of §2.8,
one slot namespace over.

### The `32sg` tell

Muser's kernel names carry their geometry. The live decode tail kernel is
`muser_fused_norm_residual_rms_norm_32sg` — "**32sg**" = 32 SIMD groups =
1,024 threads, because it fuses *two* norms plus a residual add over Muse
Glimmer's 6,656-wide rows and wants the whole row resident in the
threadgroup. The dispatch comment says exactly that:

```rust
// crates/muser-engine/src/metal/encode/norm.rs:236
// 32 SIMD groups keep the 6,656-wide Muse tail resident and match the
// accepted Ferrite geometry. 33 floats are padded to Metal's 16-byte
// dynamic-threadgroup-memory alignment.
encoder.set_threadgroup_memory_length(0, 144);
encoder.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(1024, 1, 1));
```

(The kernel body at
`[crates/muser-engine/src/shaders/ferrite/rmsnorm_batch_tail.metal:147-201]`
is the same two-barrier pattern as `rms_norm_batch`, with `1024u` strides
and a 32-entry combine loop — read it now, it will hold no surprises.)
Whenever you meet a Muser kernel in this book, decode the suffix first:
`_4sg`, `_32sg`, `_4r2s` (4 rows, 2 SIMD groups), `_n32` (32 output rows
per threadgroup). The suffix is the geometry.

## 2.7 The grid is the output shape

How many threads should a kernel launch? There is a rule, and it is short
enough to memorize: **the grid is the output shape.** You launch one thread
per thing you intend to write, and the shape of the launch follows the
shape of the answer, never the shape of the input. Getting it wrong is not
a performance bug, it is a correctness bug — too small a grid leaves part
of the output holding whatever was there before, and too large a grid runs
off the end of the buffer unless the kernel guards. Two contrasting real
geometries from Muser's decode path show the rule doing its work:

**An elementwise kernel — one thread per element.** Muse Glimmer's sigmoid
attention-output gate multiplies one 4,096-wide vector by an
elementwise-gated copy of itself (the architecture's oddity, covered in
[Ch 17](17-sigmoid-gate-and-oproj.md)). The whole kernel:

```metal
// crates/muser-engine/src/shaders/ferrite/sigmoid_gate.metal:7
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

And the whole dispatch:

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

`dispatch_1d` is Muser's one-line elementwise helper:

```rust
// crates/muser-engine/src/metal/encode.rs:1337
pub(super) fn dispatch_1d(encoder: &ComputeCommandEncoderRef, count: usize) {
    if count == 0 {
        return;
    }
    let width = count.min(256) as u64;
    encoder.dispatch_threads(MTLSize::new(count as u64, 1, 1), MTLSize::new(width, 1, 1));
}
```

Note it uses `dispatch_threads` — the *non-tiled* form where you state the
exact total thread count and a threadgroup width, and Metal works out the
grid (256-wide groups here, and Metal pads the tail so the `gid < n` guard
matters). One thread per output element; idle guard-exited threads cost a
branch, not a stall.

**A batched pointwise kernel — 1,024-wide groups.** The prefill-side
`residual_add_batch` (`dst[i] += src[i]` over a batch) instead fixes the
geometry explicitly, one thread per element in groups of 1,024:

```rust
// crates/muser-engine/src/metal/encode.rs:515
self.bind(encoder, "residual_add_batch");
encoder.set_buffer(0, Some(destination.metal()), 0);
encoder.set_buffer(1, Some(source.metal()), 0);
set_value(encoder, 2, &(total as u32));
encoder.dispatch_thread_groups(
    MTLSize::new(total.div_ceil(1024) as u64, 1, 1),
    MTLSize::new(1024, 1, 1),
);
```

Same rule — grid = `⌈total/1024⌉` groups of 1,024 threads covers exactly
`total` outputs. And a matvec inverts the aspect ratio entirely: it emits
one output element per weight-matrix row, so it launches *many small
threadgroups* (one or a few per row). The exact matvec geometries are
[Ch 13](13-the-qkv-gate-matvec-family.md)'s subject; remember only the
rule.

## 2.8 Binding memory: slots are the contract

Before launching, you tell each thread where its inputs live. You bind
`MTLBuffer` objects to numbered **buffer slots** (0, 1, 2, …) that the
kernel reads as `[[ buffer(N) ]]`. Look back at `sigmoid_gate_inplace`:
its parameters carry `[[ buffer(0) ]]`, `[[ buffer(1) ]]`, `[[ buffer(2) ]]`,
and those numbers line up one-for-one with the CPU-side calls
`set_buffer(0, …)`, `set_buffer(1, …)`, `set_value(encoder, 2, …)`. The
slot index is the whole contract between a kernel and its dispatch code.

Small constants — a `uint` count, an `eps` float, a small argument struct —
do not get their own `MTLBuffer`; they are inlined into the command buffer
with `set_bytes`, which is what the `set_value` helper wraps
(`[crates/muser-engine/src/metal/encode.rs:1329-1335]`).

The subtle half of `set_buffer` is its third argument, a **byte offset into
the buffer**. Muser's weight dispatch binds *views* — one giant mapped
buffer plus per-tensor offsets — through exactly this argument:

```rust
// crates/muser-engine/src/metal/encode/multicol.rs:192
encoder.set_buffer(0, Some(weights.metal()), weights.offset() as u64);
```

What `weights` is, why one buffer serves the whole model, and what the
offset arithmetic must respect are [Ch 3](03-unified-memory-and-buffers.md)'s
story — the next chapter.

## 2.9 Serial vs concurrent dispatch

So far the encoder looked like a strict sequence: dispatch N, then
dispatch N+1. Metal offers two flavors of compute encoder, and Muser uses
both deliberately:

- A **serial** encoder (`new_compute_command_encoder()`) runs its dispatches
  in order, one at a time. Simple, and leaves the GPU under-occupied when
  consecutive dispatches are independent.
- A **concurrent** encoder
  (`compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent)`)
  may overlap dispatches that have no dependency between them — exactly
  what a decode graph wants, because its four input projections (Q, K, V,
  gate) read the same input and write disjoint outputs.

You saw `forward_token` create the concurrent encoder in §2.3. Freedom is
not free: with overlap, *you* must say where dependencies live. Muser wraps
that rule in one type, `GraphEncoder`, whose entire job is to insert an
explicit **memory barrier** between dispatch groups:

```rust
// crates/muser-engine/src/decode.rs:6256
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
    // …
}
```

Read it as a protocol: the 52-layer graph is encoded as a sequence of
*closures* (`dispatch(command, |encoder| { … })` at
`[crates/muser-engine/src/decode.rs:6273-6279]`); independent kernels are
deliberately packed into one closure (they may overlap inside it); every
closure boundary is a real dependency, and `before_dispatch` plants a
`memoryBarrierWithScope` there so the GPU drains prior writes before the
next group starts. The comment names the provenance: this is llama.cpp's
own dependency-reset discipline, adopted wholesale.

So why does the serial encoder still exist at all? Because which flavor
wins is not obvious from first principles, and the losing side was kept so
the comparison can be re-run. The decode path had already been converted:
its projections sit together in one closure and overlap. Prefill was the
open question, and the argument for leaving it serial was respectable —
prefill kernels are large, a large kernel should saturate the GPU on its
own, and concurrency would then be buying overlap nobody needed while
still paying for barriers. The comment records how that argument fared:

```rust
// crates/muser-engine/src/decode.rs:976 (fields of MetalShared)
// Prefill-only concurrent Q/K/V/gate and FFN gate+up. Decode already
// groups those projections; serial prefill paid a launch tax on PP128.
// `MUSER_SERIAL_PREFILL_DISPATCH` restores the previous encoder for A/B.
concurrent_prefill_dispatch: bool,
```

Serial prefill lost. The launch tax it paid was visible in the
prompt-processing cell, so concurrent became the default — and the old
encoder was not deleted. `MUSER_SERIAL_PREFILL_DISPATCH` restores it
`[crates/muser-engine/src/decode.rs:1332]`, which means the two policies
can still be raced against each other on one binary, on whatever machine
doubts the result. That is the flag-for-measurement culture you will meet
throughout this book: a decision that was made by a number keeps the switch
that produced the number.

The same instinct, pushed to its diagnostic extreme, gives *every* dispatch
group its own command buffer. That is deliberately the expensive shape this
section argued against — and it is exactly why it is useful, because Metal
then reports an exact GPU interval per group instead of one interval per
token. It lives in the `PhaseProfiler`
`[crates/muser-engine/src/decode.rs:6224-6236]`, gated by
`MUSER_METAL_PHASE_PROFILE`, diagnostic-only and never on the serving path.

## 2.10 The full cycle, and a bounded wait

Putting the whole chapter together, the lifetime of one piece of GPU work
in Muser:

1. Acquire the accelerator (`AcceleratorScheduler::acquire`,
   `[crates/muser-engine/src/decode.rs:1040]`).
2. `queue.new_command_buffer()` — get a blank tape.
3. Create the encoder (concurrent on the token path; serial where the A/B
   flag says so).
4. (Hundreds of times) bind pipeline + bind buffers + set constants +
   dispatch — via `encode_*` functions, grouped into barrier-delimited
   closures.
5. `encoder.end_encoding()` — stop recording.
6. `command_buffer.commit()` — press play; the GPU starts, asynchronously.
7. Wait for completion — *bounded*.

Step 7 deserves its own paragraph, because it is where Muser's fail-closed
culture reaches even the wait. `CommandBufferRef::wait_until_completed()`
blocks a thread *unboundedly* — a wedged GPU would freeze the serving
thread forever. Muser's `wait_for_completion` parks that blocking call on a
detached watcher thread and bounds the caller's wait with a condvar, so a
hang becomes a logged `Deadline` error carrying the command buffer's label
and status, not a frozen box (`[crates/muser-engine/src/metal/context.rs:149-211]`).
The 300-second deadline you saw in `forward_token` is that mechanism.

## 2.11 Tradeoffs

Three decisions in this chapter had a plausible alternative. Here is what
happened when we took each alternative seriously.

**One command buffer per token vs. per dispatch.** Start with the easy
half. The obvious way to run a decode graph is one command buffer per
kernel — record, commit, wait, repeat — and it is what a first engine
writes, because every step is independently debuggable. It also pays a
CPU↔GPU round trip per dispatch, hundreds of them per token, which is why
Muser records the whole token (embedding through softcap) onto one buffer
and waits once.

The hard half is that the cheap shape is not free either: batching a token
onto one tape makes encode-side work and dispatch *count* grow, and both
are measurable. So the campaign measured them. The production one-token
decode graph reconciles to **760 profiling closures vs the legacy route's
564 — a +196 difference — reconciled exactly into 104 separated
norm-boundary groups + 39 SWA wrapped-ring staging groups + 52
KV-publication/attention splits + 1 last-row copy**
`[docs/decode-dispatch-gap-20260815.md §Corrected
closure-count diff]`. (Closures there are Rust profiling closures, not raw
Metal dispatches; the two counts are not interchangeable.)

Read that reconciliation and one term looks like free money. The 104
separated norm-boundary groups are the largest single line in the diff, and
they exist only because a boundary was drawn between two norms. So we built
the fix: a hybrid that fused across the boundary and deleted the separated
groups. The expectation was that the graph would shed the biggest slice of
its extra structure while computing exactly the same numbers, since fusing
adjacent arithmetic is supposed to be a rearrangement, not a change. It was
not. The hybrid changed bits — normalized-logprob max error 3.197e-4
against a 1e-4 contract, with the first divergence tracked down to one f16
ULP in layer-1 V — and we kept the postmortem instead of the patch
`[docs/decode-dispatch-gap-20260815.md §Rejected hybrid postmortem]`. (In
that sentence a *logprob* is the logarithm of the probability the model
assigns a token, and a *ULP* — unit in the last place — is the smallest
step a float format can take; the parity contract is measured in exactly
those units. [Ch 38](38-measuring-against-llama-cpp.md) owns it.)

The lesson is worth stating plainly, because the rest of the book keeps
running into it: dispatch structure is a lever, but exactness is the gate.
A restructuring that changes the output is not a faster engine, it is a
different engine, and it does not get to compete. What survived from the
same investigation was the one removal that was provably exact — a single
6,656-element copy — which bought −0.136 ms GPU (−0.34 %)
`[docs/decode-dispatch-gap-20260815.md §Landed and rejected
reductions]`. That is the honest size of the prize when you refuse to
trade bits for it.

**Concurrent vs serial encoding.** Concurrency buys overlap between
independent projections and sells you a new obligation in return: every
real dependency now needs its barrier, spelled out by hand, or the engine
computes garbage fast (§2.9). That is a bad trade whenever the overlap is
worth nothing — which is what the "prefill kernels are large enough
already" argument predicted, and what the measurement contradicted:
"serial prefill paid a launch tax on PP128"
`[crates/muser-engine/src/decode.rs:976-979]`, PP128 being the 128-token
prompt-processing benchmark cell. Concurrent is therefore the default, and
`MUSER_SERIAL_PREFILL_DISPATCH` survives for one purpose only: re-running
the comparison that settled it.

**Panicking on an unregistered pipeline.** The gentle-looking alternative
is to hand back nothing and carry on — return an `Option`, skip the
dispatch when the name is unknown, keep the server up. Follow that path to
its end, though, and a typo'd kernel name produces no error at all: the
output buffer keeps whatever was in it, the graph continues, and the model
goes on emitting plausible tokens computed from a step that never ran.
Muser takes the crash instead. `PsoCache::get` panics on a miss
`[crates/muser-engine/src/metal/pso_cache.rs:45-49]`, so the mistake
surfaces at the first dispatch, in development, as a programming error
with a name attached. Fail-closed, even here.

## 2.12 What's next

You now know every Metal concept needed to read any kernel in this book:
device, queue, command buffer, encoder, thread, threadgroup, SIMD group,
grid, buffer slots, the encode/commit/wait cycle, and Muser's
barrier-between-groups protocol. One big thing was deferred: §2.8's
`set_buffer(slot, buffer, offset)` hid the entire memory story — what a
`MTLBuffer` is on this machine, why one buffer can hold the whole
16.76 GB model, and how the weights get from disk into it without a single
copy. That is unified memory, and it is the next chapter.

---

## References

- `[crates/muser-engine/src/metal/context.rs:32-42]` — `MetalContext`
  (device, queue, and the three libraries); `:46-47` device/queue creation;
  `:149-211` the deadline-bounded `wait_for_completion`.
- `[crates/muser-engine/src/decode.rs:5432-5463]` — `forward_token`: the
  one-command-buffer-per-token cycle with the concurrent encoder.
- `[crates/muser-engine/src/decode.rs:954-984]` — `MetalShared`: one
  context/pipeline-set/arena per engine, and the
  `concurrent_prefill_dispatch` A/B comment.
- `[crates/muser-engine/src/decode.rs:1020-1030]` — `AcceleratorScheduler`,
  the one owner of the shared queue.
- `[crates/muser-engine/src/decode.rs:6124-6279]` — `EncodeTarget`,
  `GraphEncoder` (concurrent + `memoryBarrierWithScope`), `PhaseProfiler`,
  and the `dispatch` closure helper.
- `[crates/muser-engine/src/shaders/ferrite/rmsnorm_batch_tail.metal:1-33]`
  — `rms_norm_batch` (the three-units reduction kernel);
  `:147-201` the `32sg` dual-eps fused tail.
- `[crates/muser-engine/src/metal/encode/norm.rs:236-240, 270-277]` — the
  1×128 and rows×1024 dispatches and the 6,656-wide geometry comment.
- `[crates/muser-engine/src/shaders/ferrite/sigmoid_gate.metal:7-16]` and
  `[crates/muser-engine/src/metal/encode/gate.rs:7-23]` — the elementwise
  kernel and its dispatch.
- `[crates/muser-engine/src/metal/encode.rs:500-523]` —
  `encode_residual_add_batch` (explicit 1,024-wide geometry);
  `:1329-1343` — `set_value` and `dispatch_1d`;
  `:21-88` — the 66-name `PIPELINES` registry.
- `[crates/muser-engine/src/metal/pso_cache.rs:45-49]` — the
  panic-on-miss pipeline cache accessor.
- `[crates/muser-engine/src/metal/encode/multicol.rs:192]` — a real
  offset-view bind (forward pointer to [Ch 3](03-unified-memory-and-buffers.md)).
- `[docs/decode-dispatch-gap-20260815.md]` — the +196-closure
  reconciliation, the rejected hybrid (3.197e-4 vs the 1e-4 contract), and
  the one exact removal (−0.136 ms GPU).
- `[Metal-PG]` — Apple, *Metal Programming Guide*: "Command Buffers,"
  "Compute Processing," "Dispatching Threads."
- `[Metal-SS]` — Apple, *Metal Shading Language Specification*: "SIMD-group
  Functions," "Threadgroup Functions," address-space attributes.
- [glossary](../glossary.md) — terms introduced this chapter: Metal, MSL,
  kernel, thread, threadgroup, threadgroup memory, threadgroup barrier,
  SIMD group, grid, MTLDevice, MTLCommandQueue, command buffer, compute
  command encoder, dispatch.
- `[ferrite-book Ch 2]` — the ancestor's tape-recorder analogy and
  three-units pedagogy this chapter ports; its 215-shader, ~171-dispatch
  and encode-percentage figures are Ferrite-lineage.
