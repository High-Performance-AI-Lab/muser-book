# Chapter 3 — Unified memory and the buffer substrate
> **status:** polished  ·  **path:** Muse Glimmer, pinned Muser tree
>
> *Prerequisites: [Ch 2](02-metal-compute-model.md). You know what an
> `MTLDevice`, a command buffer, an encoder, and
> `set_buffer(slot, buffer, offset)` are. This chapter assumes you know
> nothing about memory architecture.*

---

## 3.1 The big picture: one DRAM, two brains

Chapter 2 closed on `set_buffer(slot, buffer, offset)` and deliberately
deferred the memory story — what an `MTLBuffer` is on this machine, and why
one buffer can hold the whole 16.76 GB model. That deferral leaves a
question standing, and it is the question this chapter answers: where do
the weights actually live while the GPU is reading them, and what does it
cost to put them there? On this machine the answer is short enough to be
surprising, and it starts from the single most important fact about Apple
Silicon for inference.

On a discrete GPU — an NVIDIA card in a desktop — the CPU and the GPU do
**not** share memory. The CPU sits next to one pool of memory chips
(CPU **[DRAM](../glossary.md#dram)**, what people call RAM), the GPU sits
next to a *separate* pool on the card (**[VRAM](../glossary.md#vram)**),
and the two pools are wired together by a bus (**[PCIe](../glossary.md#pcie)**).
Every time the CPU wants the GPU to see a weight tensor, it must `memcpy`
the bytes across that bus. A 16.76 GB model means shipping 16.76 GB over
PCIe before the first token can even be thought about.

Apple Silicon is built differently. The CPU cores and the GPU are on the
same piece of silicon (a system-on-chip, **[SoC](../glossary.md#soc)**),
and they both point at the **same physical pool of DRAM**. There is no
second pool and no bus to copy across. This single-pool design is called
**[unified memory](../glossary.md#unified-memory)**; Figure 3.1 draws the
contrast with the discrete world.

```text
   APPLE SILICON (unified memory)              DISCRETE GPU (e.g. NVIDIA)

   ┌─────────────────────────────┐             ┌───────────┐   PCIe   ┌───────────┐
   │            SoC              │             │    CPU    │◄════════►│    GPU    │
   │  ┌──────┐        ┌──────┐   │             │  ┌─────┐  │  bus +   │  ┌─────┐  │
   │  │ CPU  │        │ GPU  │   │             │  │ DRAM │  │ memcpy()│  │ VRAM │  │
   │  │cores │        │cores │   │             │  └─────┘  │          │  └─────┘  │
   │  └──┬───┘        └───┬──┘   │             └───────────┘          └───────────┘
   │     │                │      │               two physical DRAM pools;
   │     └───────┬────────┘      │               you memcpy weights
   │        ┌────┴─────┐         │               across the bus
   │        │ one DRAM │         │
   │        │  pool    │         │
   │        │ (96 GB)  │         │
   │        └──────────┘         │
   └─────────────────────────────┘
     one physical DRAM; CPU and GPU
     read/write the SAME bytes
```
*Figure 3.1: Unified memory (Apple Silicon) vs. the discrete split. On
Apple Silicon "uploading weights to the GPU" is a no-op — the bytes are
already there. (This diagram is also the seed of Part VI: the remote GB10
node *is* a discrete-memory machine across a 10GbE wire, which is why the
disaggregated lane moves KV tiles, not work-in-progress.)*

For inference the consequence is transformative: a model file on disk can
be mapped into the process's address space once, and — because the GPU
shares that address space — the GPU can read the weights directly from
those pages. There is no second copy. §3.6 shows the exact calls; first,
what Metal calls its buffers.

## 3.2 The three Metal storage modes, and the one Muser uses

Unified memory settles *where* bytes can live. It does not tell Metal what
you intend to do with them, and Metal insists on being told: every
allocation carries a declaration. So the question for this section is which
declaration Muser makes — for every buffer in the engine, without exception
— and why a faster-looking alternative was tried and then handed back.

When you ask the `MTLDevice` for a buffer, you must tell it a
**[storage mode](../glossary.md#storage-mode)** — where the bytes physically
live and who can see them. Metal defines three `[Metal-PG, "Resource
Objects: Storage Modes"]`:

- **`StorageModeShared`** — one copy of the bytes, in unified memory. CPU
  and GPU read and write the same physical pages; a CPU write is instantly
  visible to the GPU. **This is the Apple Silicon mode.**
- **`StorageModePrivate`** — bytes only the GPU can touch; the CPU must
  stage data through a Shared buffer and a GPU **blit** copy (a block copy
  run on the GPU through a dedicated blit encoder, as opposed to the
  compute encoder that runs kernels). This is how discrete VRAM works.
- **`StorageModeManaged`** — two caches, one per side, explicitly
  synchronized with `synchronize`/`didModifyRange` calls. It exists for
  older discrete-Mac setups.

On Apple Silicon, Private and Managed are simply irrelevant: there is no
separate VRAM to be private to and no two caches to manage. Muser uses
`StorageModeShared` for every buffer it creates, through one function:

```rust
// crates/muser-engine/src/metal/buffer.rs:7
fn shared_tracked() -> MTLResourceOptions {
    // Several accepted Muse paths still cross compute encoders (notably
    // target-hidden prefill/capture). Untracked resources are only valid when
    // every such dependency has an explicit fence/barrier. b9678d4 enabled
    // untracked mode globally before that contract existed and empirically
    // changed DFlash conditioning while leaving final greedy IDs unchanged.
    MTLResourceOptions::StorageModeShared
}
```

A function that returns one enum constant should not need a paragraph of
commentary above it. This one does, because the interesting part is what
the return value *omits*: it is just `StorageModeShared`, with no
`HazardTrackingModeUntracked` flag beside it. The comment is a scar, and
the story behind it is worth telling in full, because the failure it
records is one this book will keep meeting: a change that produces
correct-looking output and is wrong anyway.

> **Lineage — the untracked-hazards gamble, and why we declined it.**
> Metal can optionally mark buffers **untracked** (`HazardTrackingModeUntracked`),
> which tells the driver to skip its automatic dependency tracking between
> dispatches. Bookkeeping you skip is work you do not pay for, and we had
> reason to expect a win: the ancestor Ferrite engine ran untracked and made it safe
> with its own conflict-driven barrier planner, measured there at a real
> but modest win `[ferrite-book Ch 3]` (Ferrite-lineage numbers, ~2–4 % on
> that lab's A18-class hardware). So we took the fork. Commit `b9678d4`
> flipped the switch engine-wide — and, in the comment's own words, did it
> "before that contract existed": before every cross-encoder dependency in
> Muser had an explicit fence standing in for the tracker we had just
> turned off.
>
> What we expected was a small speedup and byte-identical output. What we
> got was identical output and *different internal state*. The build
> *empirically changed DFlash [speculative-draft] conditioning while leaving
> final greedy IDs unchanged* (`[crates/muser-engine/src/metal/buffer.rs:7-14]`).
> Read that twice. The tokens matched — so every greedy-output test we had
> would have gone green. What moved was the speculative draft's
> conditioning, which is watched on its own precisely so that a change like
> this has somewhere to show up.
>
> A silent conditioning change with correct-looking output is exactly the
> failure this engine exists to prevent, so the switch went back. Muser runs
> Metal's default **tracked** storage plus the explicit
> `memoryBarrierWithScope` groups you met in
> [Ch 2 §2.9](02-metal-compute-model.md). The general rule survives from the
> ancestor book, and it is worth stating in the abstract because it recurs
> for the rest of the engine: turning off a safety net is safe *if and only
> if* something else provably enforces the ordering. Ferrite had that
> something — a barrier planner. Muser, at that commit, did not, and the
> cheaper provable thing was to keep the net on.

## 3.3 The buffer substrate: three types, one view

Unified memory decides where bytes live; it says nothing about what they
*mean*. An engine that keeps f32 activations, f16 KV planes and immutable
quantized weights in a single address space needs some way of stopping
itself from confusing them — and it has to do that without a runtime type
tag on the hot path, because the hot path runs per token. That is the job
of the buffer module, and it is small enough to read in one sitting. Three
concrete buffer types and one view type carry the whole engine:

```rust
// crates/muser-engine/src/metal/buffer.rs:28
#[derive(Clone)]
pub struct GpuBuffer {
    inner: Buffer,
    len: usize,
}

/// Shared Metal storage whose logical elements are IEEE-754 binary16 bits.
///
/// Keeping this distinct from [`GpuBuffer`] prevents an F16 KV plane from
/// being accidentally fingerprinted or indexed as an F32 activation buffer.
#[derive(Clone)]
pub struct GpuHalfBuffer {
    inner: Buffer,
    len: usize,
}

#[derive(Clone)]
pub struct GpuBytes {
    inner: Buffer,
    len: usize,
    _mmap: Option<std::sync::Arc<memmap2::Mmap>>,
}

#[derive(Clone, Copy)]
pub struct GpuByteView<'a> {
    buffer: &'a GpuBytes,
    offset: usize,
    len: usize,
}
```

Walk them:

- **`GpuBuffer`** — f32 scratch: the residual stream, activations, logits.
  `len` counts floats. CPU access goes through checked `as_slice` /
  `as_mut_slice` views over the shared storage (`buffer.rs:216-232`).
- **`GpuHalfBuffer`** — f16 storage (binary16, the 16-bit "half" float;
  [Ch 5](05-quantization-from-scratch.md) covers its bit layout). The doc
  comment says why it
  is a separate type and not a flag: keeping it distinct "prevents an F16
  KV plane from being accidentally fingerprinted or indexed as an F32
  activation buffer" (`buffer.rs:34-37`). A KV plane is one layer's
  key/value cache buffer pair; every Metal KV plane is f16 on
  all lanes, and [Ch 15](15-kv-store-and-the-ring.md) explains the layout.
- **`GpuBytes`** — raw bytes, and the only type that can carry an mmap.
  `_mmap` holds the file mapping alive; the leading underscore suppresses
  a lint for a field that exists for its *lifetime*, not its value — drop
  the last `GpuBytes`, the `Arc` count hits zero, the mapping is released.
- **`GpuByteView`** — a borrowed slice of a `GpuBytes`: `(buffer, offset,
  len)`, created by the checked `view(offset, len)` method
  (`buffer.rs:153-160`, which refuses a view that would run past the end).
  This is the weight-tensor handle the kernels of Part IV receive.

If you know the ancestor Ferrite book: its `GpuBuffer` was a six-field
struct with a dtype, a view length, and an arena offset. Muser's substrate
is deliberately flatter — three typed wrappers instead of one typed field,
and views only where views are needed (byte storage). It can afford to be:
the engine serves one model, so the dispatch code already knows every
buffer's type statically, and a dtype field would only be describing at
runtime what the compiler could have enforced. That simplification was made
on purpose when the substrate was pulled across from the ancestor, and the
crate documentation records the "pull-and-simplify" provenance
`[crates/muser-engine/src/lib.rs:163-171]`.

## 3.4 Zero-init is policy, not luck

What is in a buffer at the moment you receive it, and who is allowed to
read it before anyone has written it? That sounds like a pedant's question
until you notice what the wrong answer looks like in an inference engine:
not a crash, but a plausible token. Garbage that happens to be zero is
indistinguishable from a real value, and a stale row in a KV cache is a
sentence the model half-remembers from someone else's conversation.

Metal's `new_buffer` does **not** zero the memory it hands you; contents
are undefined. Inference is full of buffers that are *partially* written
and then *fully* read (reductions with guard lanes; ring buffers whose
tail rows are not yet meaningful), so Muser makes initialization explicit:

- **`zeros`** allocates and CPU-memsets — the default for everything
  (`GpuBytes::zeros` at `buffer.rs:59-73`, `GpuBuffer::zeros` at
  `:182-196`, `GpuHalfBuffer::zeros` at `:238-244`). The cost is paid once
  at allocation, never on the hot path.
- **`uninitialized`** exists on `GpuHalfBuffer` only, and its doc comment
  is a contract, not an invitation:

```rust
// crates/muser-engine/src/metal/buffer.rs:246
    /// Allocate without CPU-touching the backing bytes.
    ///
    /// This is for multi-gigabyte KV planes ONLY. Their ring/logical
    /// metadata (`MetalKvPlane::origin_logical`/`origin_physical`/`len`)
    /// guarantees every row is written by a `store_kv_*` dispatch before any
    /// row within `[origin, origin + len)` is ever read back, so the CPU
    /// zero-fill `zeros()` performs is pure startup-time cost on an
    /// allocation that can be many gigabytes. Every other caller must keep
    /// using `zeros()` -- this path leaves stale bytes behind in release
    /// builds.
    ///
    /// In debug/test builds the bytes are poisoned (never plain zero)
    /// instead of left stale, so a bug that lets a read reach a
    /// not-yet-written row is conspicuous rather than silently reading
    /// zeros; see `kv_uninitialized_write_then_read_round_trips` below.
    pub fn uninitialized(context: &MetalContext, len: usize) -> Result<Self, MetalError> {
```

That last paragraph is the elegant part, and it is the answer to the
question this section opened with. In debug builds the unwritten rows are
filled with `0xDEAD` (`buffer.rs:270-275`), so a contract violation reads
back conspicuously poisoned values instead of plausible zeros. Put the
other way round: the fast path is allowed to leave stale bytes behind
precisely *because* the slow path refuses to leave believable ones. A bug
that lets a read reach a not-yet-written row cannot hide as a slightly odd
number — it is loud, in exactly the builds where somebody is looking.

Where each path is used, precisely:

- **Live session KV planes zero-fill by design.** The session constructor
  allocates through `MetalKvPlane::new` → `GpuHalfBuffer::zeros`
  (`[crates/muser-engine/src/decode.rs:229-244]`), with an explicit
  in-source justification at `decode.rs:1350-1352`: "Zero-filled on
  purpose: wrapped SWA rows must never expose uninitialized storage during
  a sequence boundary transition."
- **Detached remote-install generations use `uninitialized`.** When a
  kvpack tile arrives from the GB10 producer, a fresh plane is built with
  `MetalKvPlane::uninitialized` and every retained row is uploaded before
  the plane goes live (`[crates/muser-engine/src/decode.rs:1862-1888]`) —
  a bulk-write-then-publish pattern where the write-before-read guarantee
  is structural.

One correction while we are here, because we nearly walked into it
ourselves. Writing this section we started where you would start — from the
engineering doc — and `docs/memory-footprint.md` says flatly that "Metal KV
buffers are allocated without a CPU memset." Taken at face value that
inverts everything above: it makes the fast path the default and the
zero-fill the exception, and this section would have been a footnote.
The two constructors above say otherwise, and an audit had already caught
the same discrepancy. The doc sentence is **wrong for live planes** —
those zero-fill, as above — and right only about the detached
remote-install generations. We kept the receipt for the correction
`[docs/kvpack-merge-handoff.md §3 D2, the 2026-08-20 audit]`. The book
inherits the fix, not the error; the general lesson is that source outranks
prose about source, including prose written by the same people.

## 3.5 Page alignment: the 16 KB contract

Before the mmap story, one piece of arithmetic — the kind that works for
years and then doesn't. The question is narrow, and it is the one to ask
whenever you hand a driver a pointer to memory you did not allocate: what
exactly are you promising it about that memory? Apple Silicon uses a
**16 KB virtual-memory page** (not the 4 KB you may know from x86). The
Metal call that wraps external memory — `new_buffer_with_bytes_no_copy` —
requires a page-aligned pointer *and* a page-aligned length. A file's byte
length is not, in general, a multiple of 16,384.

Muser handles this by rounding the *Metal-facing* length up to the page
boundary while keeping the logical length exact — and the code is unusually
careful about *why* that is safe:

```rust
// crates/muser-engine/src/metal/buffer.rs:91
    pub fn from_mmap(
        context: &MetalContext,
        mmap: std::sync::Arc<memmap2::Mmap>,
    ) -> Result<Self, MetalError> {
        // Metal documents that `newBufferWithBytesNoCopy:length:options:
        // deallocator:` requires a page-aligned length (the pointer is
        // already page-aligned -- POSIX `mmap` always returns one). The raw
        // file length has "worked" here only because `mmap` itself reserves
        // whole pages under the hood and zero-fills the tail of the last
        // one -- that's an undocumented tolerance, not a guarantee, so round
        // the length Metal sees up to the page boundary explicitly. Bytes
        // between the real file length and that boundary are the kernel's
        // own zero-filled mmap tail, so this never reads outside the
        // mapping. `GpuBytes::len()` keeps reporting the exact, unrounded
        // file length; only the Metal-facing allocation grows.
        let mmap_len = mmap.len();
        let rounded_len = if mmap_len == 0 {
            0
        } else {
            let page = page_size();
            mmap_len
                .checked_add(page - 1)
                .map(|padded| padded / page * page)
                .ok_or(MetalError::Allocation(mmap_len))?
        };
        let inner = context.device.new_buffer_with_bytes_no_copy(
            mmap.as_ptr() as *const std::ffi::c_void,
            rounded_len as u64,
            shared_tracked(),
            None,
        );
        // … (allocation check elided: see buffer.rs:122-124) …
        Ok(Self {
            inner,
            len: mmap_len,
            _mmap: Some(mmap),
        })
    }
```

*(the allocation-size check at `buffer.rs:122-124` is elided; everything
else is verbatim.)*

That paragraph of comment is doing real epistemic work, and it is worth
slowing down for, because the bug it fixes had never once fired. Passing
the raw file length straight to Metal *worked*. It worked because `mmap`
reserves whole pages under the hood and zero-fills the tail of the last
one, so the bytes past the file's end were always mapped and always zero —
and no test we could write would tell that apart from being right. The comment
separates the two anyway: what Metal *guarantees* (a page-aligned length is
required) from what merely *happened to work* ("an undocumented tolerance,
not a guarantee"). Then the code goes and obeys the documented contract.
Nothing was failing when that change was made; the point is that the
alternative was a load path resting on an OS behaviour nobody ever promised
us, and a future macOS is under no obligation to keep providing it.

Two details finish the section. A unit test pins the rounding down
(`from_mmap_rounds_metal_length_up_to_the_page_boundary`,
`buffer.rs:347-376`): a 10-byte file yields `len() == 10` and a
Metal-facing length that is a page multiple — the logical length stays
exact, only the allocation grows. And the page size itself comes
from POSIX `getpagesize()` via a one-line `extern "C"` — avoiding a `libc`
dependency for one constant (`buffer.rs:16-26`).

Why does the pointer never need rounding here, where the ancestor engine
had to round tensor offsets down to page boundaries? Because of *what*
gets wrapped — the next section.

## 3.6 Zero-copy at 16.76 GB scale: mmap → one buffer → offset views

This is the payoff, end to end — the section where "the CPU and the GPU
share memory" stops being an architecture diagram and becomes a load path
that copies nothing. Hold the question a discrete-GPU engineer would ask
first while you read it: *when do the weights get uploaded?* The answer is
that there is no upload, and three code locations are enough to show why.

**1. The engine mmaps the whole GGUF once.** Loading does not read the
weights into RAM; it maps the file and records where each tensor lives:

```rust
// crates/muser-engine/src/weights.rs:171
    pub fn open(path: &Path, gguf: &GgufFile) -> Result<Self, MuseConfigError> {
        let file = File::open(path)
            .map_err(|e| MuseConfigError::Geometry(format!("open {}: {e}", path.display())))?;
        // SAFETY: the checkpoint is a read-only immutable input for the
        // lifetime of this process; we never write through the mapping.
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| MuseConfigError::Geometry(format!("mmap {}: {e}", path.display())))?;

        let mut index = HashMap::with_capacity(gguf.tensors.len());
        for t in &gguf.tensors {
            let start = (gguf.data_offset + t.offset) as usize;
            let n_elem: usize = t.shape.iter().product::<u64>() as usize;
            let be = t.dtype.block_elements();
            let len = n_elem.div_ceil(be) * t.dtype.block_size();
            if start + len > mmap.len() {
                return Err(MuseConfigError::Geometry(format!(
                    "tensor {} runs past end of file ({} + {} > {})",
                    t.name,
                    start,
                    len,
                    mmap.len()
                )));
            }
            index.insert(t.name.clone(), (start, len, t.dtype, t.shape.clone()));
        }
```

The `index` maps every tensor name to `(file_offset, byte_len, dtype,
shape)` — with a fail-closed bounds check per tensor. The CPU reference
path reads weights straight out of this mapping via `TensorView`
(`weights.rs:40-48`: raw bytes plus geometry, "weights are never
materialized as f32" per the module doc).

**2. The Metal driver wraps that mapping in one `MTLBuffer`.** The decode
path takes the same `Arc<Mmap>` and hands it to `GpuBytes::from_mmap` from
§3.5 — the entire 16.76 GB file becomes **one** Metal buffer, zero bytes
copied:

```rust
// crates/muser-engine/src/decode.rs:1199
let context = MetalContext::new()?;
let kernels = MetalKernels::new(&context)?;
let mapped_weights = GpuBytes::from_mmap(&context, weights.mapped_file())?;
let residency_set = crate::metal::residency::create_and_attach(
    &context.device,
    &context.queue,
    &[mapped_weights.metal()],
);
```

**3. Every weight tensor is a checked offset view of that one buffer.**
Each projection remembers only its GGUF layout (`TensorLayout`, parsed at
load, `weights.rs:350-373`); at dispatch time it asks the arena for a
slice:

```rust
// crates/muser-engine/src/decode.rs:148
    fn view<'a>(&self, mapped: &'a GpuBytes) -> GpuByteView<'a> {
        mapped
            .view(self.layout.file_offset, self.layout.byte_len)
            .unwrap_or_else(|| panic!("validated GGUF tensor {} left mapped file", self.name))
    }
```

And the bind — the line [Ch 2 §2.8](02-metal-compute-model.md) promised to
explain — hands the GPU "which buffer" and "where in it" in a single call:

```rust
// crates/muser-engine/src/metal/encode/multicol.rs:192
encoder.set_buffer(0, Some(weights.metal()), weights.offset() as u64);
```

Put the three together and the discrete-GPU question dissolves rather than
gets answered. There is no upload step because there is nowhere to upload
*to*: the file's pages, the Metal buffer, and the bytes a kernel dereferences
are one region of physical memory wearing three names. The figure below is
the entire load path — a mapping, a wrapper around it, and arithmetic.

```text
   GGUF file on disk (16,756,681,056 B)
   ┌──────────────────────────────────────────────────────────────────────┐
   │ [header] blk.0.attn_q  blk.0.attn_k … blk.51.ffn_down  output (LM)  │
   └───────────────────────────────┬──────────────────────────────────────┘
                                   │  mmap()  (one Arc<Mmap>, read-only)
                                   ▼
   unified memory: ONE MTLBuffer over the whole mapping (no copy, ever)
   ┌──────────────────────────────────────────────────────────────────────┐
   │ file start   +q_off        +k_off      …                 file end    │
   │ ├ q.weight ──┤ ├ k.weight ─┤ ………………………………………… ├ output.weight ──┤│
   └─▲──────────────▲──────────────────────────────────────────▲─────────┘
     │              │                                            │
  GpuByteView{   GpuByteView{                              GpuByteView{
    offset:q_off, offset:k_off,                          offset:out_off,
    len:q_len,    len:k_len,                             len:out_len,
    buffer:arena} buffer:arena}                          buffer:arena}
     └─── all views share the SAME MTLBuffer; set_buffer passes the offset ─┘
```
*Figure 3.2: One mmap → one MTLBuffer → one checked offset view per tensor
(offsets symbolic; each is the tensor's `TensorLayout.file_offset` from the
GGUF index). No weight byte is copied at load or at dispatch; pages are
demand-fetched by the OS the first time the GPU reads them.*

Why one giant buffer plus views, instead of one `MTLBuffer` per tensor
(hundreds of them)? Each distinct buffer costs Metal bookkeeping, a
virtual-memory mapping, and **TLB** pressure (the TLB —
translation-lookaside buffer — is the small cache of virtual→physical page
translations; each buffer consumes entries). With one arena there is
exactly one mapping, and the per-tensor "allocation" is a 24-byte struct.
The ancestor book demonstrated the same trade on its engine's arena
`[ferrite-book Ch 3]`; Muser's version is the same idea expressed through
`GpuByteView` over `GpuBytes`. We are not alone in the choice: llama.cpp's
Metal backend maps the whole file and slices it the same way. That
corroboration reaches us second-hand, through the ancestor book's audit of
`ggml_metal_buffer_map` `[ferrite-book Ch 3]` — this book has not re-read
llama.cpp's source, so we mark it lineage rather than verification.

### Keeping 16 GB resident: the residency set

One more substrate piece, and it exists only because of the scale — at a
few hundred megabytes nobody would bother. The mapped arena is
bound by *every* projection in *every* command buffer — per token. Rather
than let Metal redo residency bookkeeping for a 16+ GiB allocation each
time, Muser attaches it to an `MTLResidencySet` once at load:

```rust
// crates/muser-engine/src/metal/residency.rs:1
//! Minimal `MTLResidencySet` owner extracted from Ferrite's Metal substrate.
//!
//! The immutable GGUF arena is bound by every projection in every command
//! buffer. Attaching it once lets Metal skip repeating residency work for the
//! 16+ GiB allocation on every token. The Objective-C surface is public on
//! macOS 15+, and absence fails open to the ordinary Metal residency path.
```

`create_and_attach` (`residency.rs:65-108`) builds the set with raw
`objc::msg_send!` calls — the `metal` crate does not wrap this API — adds
the arena's buffer, commits, requests residency, and attaches the set to
the queue. On any macOS release without the API it returns `None` and the
engine proceeds on Metal's ordinary residency path. That distinction
matters more than it looks: this is an optimization that *fails open*.
Lose it and tokens still come out, correct, with Metal doing the residency
bookkeeping the long way — which is exactly the opposite of the untracked
gamble earlier in this chapter, where dropping the safety net changed
behaviour silently. Both are engine paths that may not be taken; only one
of them is permitted to be quiet about it.

## 3.7 What 96 GB buys, revisited from the buffer side

[Ch 1 §1.5](01-why-inference-is-a-memory-problem.md) budgeted the 96 GB by
artifact — how much each thing takes. Now that you know what a buffer
actually *is*, the same budget answers a sharper question: which of these
bytes does the engine own, and which is it merely borrowing from the
operating system? From the substrate's point of view the same budget
reads:

- **Weights are not "used" memory in the ordinary sense** — they are a
  read-only file mapping, page-cache backed
  `[docs/memory-footprint.md §Other material allocations]`. The OS can
  evict clean pages under pressure and re-fault them on next access. What
  the residency set adds is a request that the pages *stay* resident while
  serving.
- **Everything the engine allocates itself** — activations, logits,
  workspaces, the f16 KV planes, staging shadows — is a `StorageModeShared`
  buffer from §3.3. The KV planes dominate: 1.827 GB per slot at full
  131,072 context, ×4 slots = 7.306 GB `[docs/memory-footprint.md]`.
- **One arena serves all slots.** The `MetalShared` design
  (`[crates/muser-engine/src/decode.rs:954-957]`) exists *because* unified
  memory makes it legal: four serving sequences share one context, one
  pipeline set, one mapped arena. On a discrete-GPU machine the same
  sharing would be possible but the copy story would not — each GPU
  residency of the weights would be a distinct 16.76 GB VRAM occupation.

## 3.8 Tradeoffs

**Tracked storage vs the untracked gamble.** The story is in §3.2's lineage
box; here is what it costs. Muser pays for Metal's dependency tracker on
every encode, plus the explicit barrier groups, and gets back a safety net
that the one experiment we ran showed it needed: the engine-wide untracked
build (`b9678d4`) "empirically changed DFlash conditioning while leaving
final greedy IDs unchanged" `[crates/muser-engine/src/metal/buffer.rs:7-14]`.
Now notice what that sentence does *not* say. It does not say the untracked
path was slower, or faster — no Muser document isolates the
tracked-vs-untracked encode cost on this machine [unverified]. The ruling
was made on correctness alone, and it would stand unchanged if untracked
turned out to measure quicker.

**One arena + views vs one buffer per tensor.** Views are cheap (24 bytes,
one bounds check) and make the whole model one mapping — but they collapse
identity: Metal sees a single buffer, so its tracker cannot distinguish a
write to one tensor from a write to another. That is exactly why the
*weights* are views (immutable after load; a write would be a bug worth a
crash) while *activations and KV planes get their own typed buffers with
distinct identity*. The split is the design: share what is immutable,
individualize what is written.

**`uninitialized` KV planes: speed with a proof obligation.** Skipping the
CPU memset on multi-gigabyte planes saves real startup time on allocations
that can be many gigabytes (`buffer.rs:248-254`) — in exchange, the ring
metadata must guarantee write-before-read for every row in
`[origin, origin + len)`. Muser makes the obligation visible three ways:
the doc comment's "ONLY", the debug-build `0xDEAD` poison, and the test
that demonstrates it (`buffer.rs:328-344`). That is the shape of the whole
bargain, and it generalizes past this buffer: when you take a shortcut
whose safety lives in some other module's invariant, spend part of the
winnings making that invariant loud.

## 3.9 What's next

You now know the entire memory story: one DRAM pool, `StorageModeShared`
everywhere, three typed buffer types plus checked byte views, zero-copy
mmap of the 16.76 GB GGUF into one Metal buffer, a residency set to keep it
warm, and an initialization policy that decides byte-for-byte what may be
read before it is written. One object from [Ch 2](02-metal-compute-model.md)
is still a black box: `MetalContext.library` — "a bundle of compiled kernel
functions, addressable by name." How `.metal` source text becomes that
bundle, why Muser deliberately keeps **three** different kernel sources
(fast-math source, strict-f32 source, and a pinned llama.cpp metallib),
and how the engine makes a silent kernel fallback impossible — that is
[Ch 4](04-pso-and-three-kernel-sources.md).

---

## References

- `[crates/muser-engine/src/metal/buffer.rs:7-14]` — `shared_tracked()` and
  the `b9678d4` untracked-mode incident comment; `:28-56` the four substrate
  types; `:59-73, 182-196, 238-244` the zero-initializing allocators;
  `:246-277` `GpuHalfBuffer::uninitialized` and the `0xDEAD` poison;
  `:91-130` `from_mmap` with the page-rounding contract;
  `:328-376` the poison and page-rounding tests; `:16-26` `getpagesize`.
- `[crates/muser-engine/src/weights.rs:163-218]` — `MuseWeights::open`
  (whole-file mmap + per-tensor index with bounds checks);
  `:40-48` `TensorView`; `:350-373` `TensorLayout`;
  `:375-378` `mapped_file()`.
- `[crates/muser-engine/src/decode.rs:1195-1206]` — `load_shared`:
  from_mmap + residency set; `:148-161` `Projection::view` /
  `nvfp4_scale_view`; `:229-244, 246-261` the two `MetalKvPlane`
  constructors; `:1350-1352` the live-plane zero-fill rationale;
  `:1862-1888` the detached remote-install generation;
  `:954-984` `MetalShared`.
- `[crates/muser-engine/src/metal/residency.rs:1-107]` — the
  `MTLResidencySet` owner (raw `msg_send!`, fails open below macOS 15).
- `[crates/muser-engine/src/metal/encode/multicol.rs:192]` — the real
  offset-view `set_buffer` bind.
- `[docs/memory-footprint.md]` — artifact sizes, KV formula, the
  on-disk-vs-resident caveat.
- `[docs/kvpack-merge-handoff.md §3 D2]` — the 2026-08-20 audit correcting
  memory-footprint.md's memset claim for live planes.
- `[ferrite-book Ch 3]` — the ancestor's unified-memory chapter: the
  arena-view and zero-copy devices this chapter re-grounds; its
  untracked-hazards A/B (−2 to −4 % on A18-class hardware) and GpuHeap /
  packed-activations history are Ferrite-lineage.
- `[Metal-PG]` — Apple, *Metal Programming Guide*: "Resource Objects:
  Storage Modes," "Tracking Resource Dependencies."
- `[Metal-SS]` — Apple, *Metal Shading Language Specification*: "Address
  Spaces."
- [glossary](../glossary.md) — terms introduced this chapter: unified
  memory, SoC, DRAM, VRAM, PCIe, storage mode, blit, mmap, page fault,
  zero-copy, page alignment, TLB, MTLBuffer, GpuBuffer, GpuHalfBuffer,
  GpuBytes, GpuByteView, residency set.
