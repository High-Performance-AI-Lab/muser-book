# Glossary seeds — Part I (Chapters 1–4)

One line per term defined on first use in Part I, for folding into
`glossary.md`. Format: `### term — one-line definition (Ch N)`.
Anchors should match the lowercase-hyphen form used in chapter links
(e.g. `glossary.md#simd-group`).

## Chapter 1 — The problem

### parameter — one learned number inside the model, tuned during training; a 30B model holds ~30 billion (Ch 1)
### weights — the parameters collectively: the giant tables of learned numbers stored on disk in a GGUF (Ch 1)
### token — one unit of text, roughly a piece of a word; the model emits tokens one at a time (Ch 1)
### decode — generating tokens one by one after the prompt has been read; the bandwidth-bound regime (Ch 1)
### prefill — reading the prompt (many tokens at once, weight rows reused); the compute-friendly regime (Ch 1)
### matvec / GEMV — matrix × vector multiply, the one operation decode performs over and over; ~2 FLOPs per weight element (Ch 1)
### FLOP — one floating-point operation (a multiply or an add); the unit of arithmetic budget (Ch 1)
### GGUF — the on-disk model-file format Muser reads: a small header plus weight tensors packed end to end (Ch 1)
### quantization — storing each learned number in fewer bits than a full float (four-ish bits per weight on the kquant lane) (Ch 1)
### bandwidth — how fast memory can hand bytes to the GPU, measured GB/s; the budget that governs decode (Ch 1)
### arithmetic intensity — a workload's FLOPs per byte read; decode's is ~3.2, fixed by the model format (Ch 1)
### roofline — the compute-vs-memory crossover picture; a workload below the machine's balance point is memory-bound (Ch 1)
### effective read rate — bytes-per-token × measured tokens-per-second (~594 GB/s for kquant decode); derived from measured throughput, not a spec (Ch 1)
### teacher-forced — a decode benchmark cell that feeds known prior tokens (e.g. 32) rather than model-generated ones (Ch 1)

## Chapter 2 — The Metal compute model

### GPU — a processor built for ten thousand easy things at once; the unit of work is the thread (Ch 2)
### Metal — Apple's API for programming its GPUs: the MSL shading language, a host API, and a memory model (Ch 2)
### MSL — Metal Shading Language, the C++ dialect kernels are written in (Ch 2)
### kernel — the per-thread program: one function every thread in a launch runs once over its own data slice (Ch 2)
### MTLDevice — the handle to the (one, system-default) GPU; allocates memory and compiles shaders (Ch 2)
### MTLCommandQueue — the queue command buffers come from; created once at startup (Ch 2)
### command buffer — the "tape": a recorded sequence of GPU instructions handed to the GPU in one shot at commit (Ch 2)
### compute command encoder — the recorder that writes dispatches onto a command buffer (Ch 2)
### dispatch — the four-line unit of GPU work: bind kernel, bind buffers, set constants, launch N threadgroups (Ch 2)
### grid — the entire launch: how many threadgroups; always shaped like the output (Ch 2)
### threadgroup — a block of 32–1024 threads sharing on-chip memory and barriers; the unit of co-scheduling (Ch 2)
### threadgroup memory — on-chip shared memory visible to all threads of one threadgroup (`threadgroup(0)`) (Ch 2)
### threadgroup barrier — `threadgroup_barrier(...)`: every thread in a threadgroup waits until prior threadgroup-memory writes are visible (Ch 2)
### thread — one execution lane running the kernel once, identified by `thread_position_in_grid` (Ch 2)
### SIMD group — exactly 32 threads executing in lockstep on one SIMD ALU; the unit that actually matters on Apple Silicon (Ch 2)
### simd_sum — one-instruction 32-way reduction across a SIMD group's lanes (Ch 2)
### MetalContext — Muser's long-lived GPU state: device, queue, and the three kernel libraries (Ch 2)
### AcceleratorScheduler — the one Mutex+Condvar owner of the shared Metal queue; decode selected first, cyclic fairness (Ch 2)
### concurrent dispatch — an encoder (`MTLDispatchType::Concurrent`) that may overlap dispatches with no dependency between them (Ch 2)
### memoryBarrierWithScope — the explicit barrier Muser plants between dispatch groups on a concurrent encoder, delimiting real graph dependencies (Ch 2)

## Chapter 3 — Unified memory and the buffer substrate

### unified memory — Apple Silicon's single physical DRAM pool shared by CPU and GPU; no copy needed to move data between them (Ch 3)
### SoC — system-on-chip: CPU cores and GPU on one piece of silicon pointing at the same DRAM (Ch 3)
### DRAM — dynamic RAM, the machine's main system memory (96 GB on the decode Mac) (Ch 3)
### VRAM — a discrete GPU's own private memory pool, filled by copying across PCIe (Ch 3)
### PCIe — the bus connecting CPU and discrete GPU; the memcpy path unified memory eliminates (Ch 3)
### storage mode — Metal's declaration of where buffer bytes live: Shared, Private, or Managed (Ch 3)
### StorageModeShared — one copy of the bytes in unified memory, CPU- and GPU-visible; the only mode Muser uses (Ch 3)
### blit — a GPU block-copy executed via a blit command encoder rather than a compute kernel (Ch 3)
### hazard tracking — Metal's automatic dependency ordering between dispatches touching the same buffer; Muser keeps it on (tracked) plus explicit barriers (Ch 3)
### MTLBuffer — Metal's handle to a range of GPU-addressable memory (Ch 3)
### mmap — mapping a file's bytes directly into the process address space, lazily, page by page (Ch 3)
### page fault — the OS trap that pulls a chunk of a mapped file into memory on first touch (Ch 3)
### zero-copy — getting data to the GPU without moving bytes: the mmap'd GGUF becomes one MTLBuffer as-is (Ch 3)
### page alignment — `new_buffer_with_bytes_no_copy` requires page-aligned pointer and length; Apple Silicon pages are 16 KB (Ch 3)
### TLB — translation-lookaside buffer, the small cache of virtual→physical page translations; fewer buffers means less pressure (Ch 3)
### GpuBuffer — Muser's f32 activation buffer type: one shared MTLBuffer plus a length, with checked CPU slice access (Ch 3)
### GpuHalfBuffer — Muser's f16 (binary16) buffer type; kept distinct so an F16 KV plane can never be indexed as F32 (Ch 3)
### GpuBytes — Muser's raw-byte buffer type, the only one that can carry the GGUF mmap (`_mmap` keeps the mapping alive) (Ch 3)
### GpuByteView — a checked `(buffer, offset, len)` slice of a GpuBytes; the per-tensor weight handle kernels receive (Ch 3)
### residency set — an `MTLResidencySet` attaching the 16+ GiB weight arena once so Metal skips per-token residency work; fails open (Ch 3)
### KV plane — one layer's f16 key+value buffers with explicit ring metadata; live planes zero-fill by design (Ch 3)

## Chapter 4 — PSOs and the three kernel sources

### PSO — pipeline state object: one library function lowered all the way to machine code for this specific GPU; what `set_compute_pipeline_state` binds (Ch 4)
### MTLLibrary — a bundle of compiled kernel functions addressable by name; the middle compile stage (Ch 4)
### metallib — an `MTLLibrary` serialized to disk; loading it skips the frontend compiler entirely (Ch 4)
### JIT compilation — compiling shader source at runtime (`new_library_with_source`) rather than loading a prebuilt library (Ch 4)
### fast-math — a compiler contract allowing NaN/Inf assumptions and FP reordering for speed; ON for the serving library, OFF for cross-vendor parity (Ch 4)
### function constant — a per-PSO compile-time value (`[[ function_constant(N) ]]`) that specializes one source into many kernels (Ch 4)
### cross-vendor library — the strict-f32 recompile of muse_reference + nvfp4 whose arithmetic matches CUDA's scalar boundaries for remote-parity routes (Ch 4)
### PsoCache — Muser's in-process name→PSO registry; panics on an unregistered name rather than falling back (Ch 4)
### fingerprint — a record of what actually ran (e.g. the metallib's SHA-256), derived from resolved state, not an env-var echo (Ch 4)
### fail-closed — refusing to proceed when a required ingredient is missing (e.g. Q6_K without the metallib aborts load) rather than silently substituting (Ch 4)
### source receipt — a provenance JSON binding a built artifact to its exact source commit, per-file hashes, and toolchain (Ch 4)
### inert guard — the discipline (honest skips, ambiguity refusal, load-time aborts) that keeps an unset flag from masquerading as an enabled one (Ch 4)
