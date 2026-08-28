# Chapter 29 — CUDA versus Metal: the differences that mattered
> **status:** polished  ·  **path:** Muse Glimmer, pinned Muser tree
>
> *Prerequisites: [Ch 2](02-metal-compute-model.md) (the Metal compute
> model), [Ch 4](04-pso-and-three-kernel-sources.md) (the three kernel
> sources), [Ch 7](07-nvfp4-native-lane.md), [Ch 27](27-why-disaggregate.md),
> [Ch 28](28-the-gx10-and-vllm-nvfp4-prefill.md).*

---

## 29.1 Where we are

[Ch 28](28-the-gx10-and-vllm-nvfp4-prefill.md) left us with two machines
that must agree about bytes: a Mac that decodes with SIMD-group Metal
kernels out of unified memory, and a GB10 that prefills NVFP4 with CUDA
tensor cores inside vLLM. This chapter is about the divide between their
GPU programming models — but it is **not** a spec-sheet tour. The question
we kept asking, machine to machine, was never "which model is better." It
was narrower and more useful: *where does agreement actually have to
happen, and what does it cost us to get it?* Every contrast below earned
its place by forcing a decision you can read in the Muser tree.
Where a fact is about the vendors' models rather than Muser's code,
it carries a vendor tag (`[CUDA §…]`, `[Metal-SS §…]`) or `[unverified]`;
where it is about Muser, it carries a `file:line`.

The one-line summary, up front: **CUDA and Metal disagree most usefully at
the seams this lane crosses — batch matrix math, the host/device memory
boundary, execution-graph control, and floating-point compilation — and
Muser answered each disagreement by pinning an interface rather than
bridging a semantics gap in code.** Table 29.1 lists all six disagreements
and the decision each forced; the sections that follow take them in order.

## 29.2 The contrast table

Before the arguments, the map. Each row below is a place where the two
programming models genuinely disagree, paired with the decision that
disagreement forced on us. Read the right-hand column first if you read
nothing else: in every row, without exception, the decision was to pin an
interface rather than to write code that bridges one model to the other.

| # | CUDA (GB10 producer) | Metal (Mac consumer) | The Muser decision it forced |
|---|---|---|---|
| 1 | **Warp**: 32 threads in lockstep; branch divergence serializes `[CUDA §thread-hierarchy]` | **[SIMD group](../glossary.md#simd-group)**: 32 lanes; `simd_*` group ops `[Metal-SS §simd-group-functions]` | Ferrite-lineage kernels ported byte-for-byte; both models center a 32-lane lockstep unit (§29.3) |
| 2 | **Tensor-core FP4 batch matmul** (W4A4, producer prefill) `[docs/disaggregated-prefill-sealing-plan-20260818.md §4]` | **SIMD-group matvec** (decode) + an M16 batch route used only at 16-row shapes | Ask each silicon for what it is shaped to do — the [Ch 27](27-why-disaggregate.md) roofline, enforced by lane (§29.4) |
| 3 | **Host/device boundary**: explicit D2H copy to pinned host memory (`connector.py`) | **One shared address space** (`StorageModeShared`, [Ch 3](03-unified-memory-and-buffers.md)) | The wire becomes the memory bus: pacing, streaming, and two chapters of wire discipline (§29.5) |
| 4 | **Streams + events**: side streams, async copies, event fences (`connector.py`) | **One queue, record-then-commit** command buffers [Ch 2](02-metal-compute-model.md) | One accelerator owner (`AcceleratorScheduler`); concurrent-vs-serial dispatch is an explicit flag (§29.6) |
| 5 | (ggml kernels compiled for both) CUDA build of llama.cpp is the comparator's home | **Pinned llama.cpp metallib** as Muser's third kernel source [Ch 4](04-pso-and-three-kernel-sources.md) | Bit-parity with the comparator beats re-expressing ggml kernels in native Metal (§29.7) |
| 6 | **Determinism knobs** (e.g. `VLLM_BATCH_INVARIANT=1`, producer self-consistency only) `[docs/disaggregated-prefill-sealing-plan-20260818.md §4]` | **Two compiled libraries**: fast-math serving + strict-f32 cross-vendor | Logit parity made Muser run *both* Metal build modes and pin the seam arithmetic (§29.8) |

*Table 29.1: Six contrasts that survived contact with this codebase.*

## 29.3 Contrast 1 — warps and SIMD groups: the shape that ported

Start with the contrast that turned out not to be much of a contrast at
all. If the two shader languages were as alien to each other as their
toolchains are, nothing in this codebase could have crossed between them
without a rewrite. So ask the question the port had to answer first: what
is the unit a kernel is *organized around*, and do the two vendors agree
about it?

First definitions, vendor-side, once. A CUDA **[warp](../glossary.md#warp)**
is the unit of 32 consecutive threads that an NVIDIA SM executes in
lockstep; when threads of a warp take different branches, the hardware
serializes the paths — the divergence penalty `[CUDA §thread-hierarchy]`.
A Metal **[SIMD group](../glossary.md#simd-group)** is Apple's counterpart:
32 threads of a threadgroup acting in lockstep, with `simd_sum`,
`simd_shuffle` and friends as explicit group-wide operations
`[Metal-SS §simd-group-functions]` — [Ch 2](02-metal-compute-model.md)
introduced it as "the unit that actually matters on Apple Silicon." The
microarchitectures beneath differ in ways this book does not need
[unverified]; what matters is that both models expose the *same abstraction
shape*: a 32-lane lockstep unit with cheap group reductions.

That shape is why Muser's ferrite-lineage kernels ported cleanly. Look at
the Q4_K matvec — four rows per threadgroup, two rows per SIMD group, one
output element per lane:

```metal
// crates/muser-engine/src/shaders/muse_reference.metal:741-743
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
```

and the closing reduction — one `simd_sum` per row, lane 0 writes:

```metal
// crates/muser-engine/src/shaders/muse_reference.metal:780-786
    for (uint row_index = 0; row_index < active_rows; ++row_index) {
        accumulator[row_index] = simd_sum(accumulator[row_index]);
    }
    if (lane == 0) {
        for (uint row_index = 0; row_index < active_rows; ++row_index) {
            output[base_row + row_index] = accumulator[row_index];
        }
    }
```

Nothing in this structure is Apple-specific *as a structure*: "one 32-lane
group owns one output row, reduce with a group op, lane 0 writes back" is
exactly how a warp-organized matvec is written on CUDA (`__shfl_down_sync`
reductions are the counterpart idiom `[CUDA §thread-hierarchy]`). The
provenance record makes the same point at file granularity: fifteen shader
files were pulled byte-for-byte from the Ferrite tree at `a85048a90`, and
three more with small adaptations `[docs/extraction-manifest.md §Stage 2]`
— a port this mechanical is only possible because the *organizing unit*
matched. The Mac's own batch kernels go one step further and use the
SIMD-group *matrix* types — `simdgroup_float8x8 mc[2]` accumulating
matrix-multiply tiles in `m16_q4k_n32`
(`[crates/muser-engine/src/shaders/ferrite/batch_m16_n32.metal:86-88]`) —
hardware matrix operations inside a SIMD-group programming model, which
foreshadows contrast 2.

## 29.4 Contrast 2 — tensor cores versus SIMD groups: the right silicon per regime

The producer's prefill arithmetic is **W4A4 dense matmul on tensor cores** —
vLLM's FP4 path (FlashInfer/CUTLASS kernels of the pinned stack)
`[docs/disaggregated-prefill-sealing-plan-20260818.md §4]`. The Mac's
decode arithmetic is a **SIMD-group matvec** — `muser_nvfp4_matvec_c{1..16}`
for the native lane (`[crates/muser-engine/src/shaders/nvfp4.metal:226]`
onward), the pinned ggml matvecs for kquant ([Ch 13](13-the-qkv-gate-matvec-family.md)).
Same numeric format on both sides (NVFP4, [Ch 7](07-nvfp4-native-lane.md));
radically different machines and purposes.

This is [Ch 27](27-why-disaggregate.md)'s roofline enforced as a *lane
policy* rather than as a benchmark observation — and the honest way to tell
it is to admit that we tried the other thing first.

Here is the fork. The Mac is not helpless at batch work: it *has* a batch
route, the `m16_q4k_n32`/`m16_n32`-family kernels, dispatched when a
16-token batch, the NVFP4 lane, and a 64-aligned input width all coincide.
For 16-row speculative-verify shapes that route is exactly the right tool,
and its existence is what made the tempting question tempting. If the Mac
can already multiply a matrix by a small block of rows, why ship prefill to
a second machine at all? Drop the wire, drop the box, keep the whole
inference in one address space. We expected to pay a penalty for that
convenience. We did not expect the shape of the bill.

The measurement closed the question. Native NVFP4 speculative decode — the
lane that lives on batched W4A4 target execution, and so the closest thing
to Mac-native batch GEMM we could put on a stopwatch — ran at **6.805
tok/s** against the 107.9 tok/s kquant bar. Worse than the headline ratio
is where the time sat: the verify step alone consumed 35.915 s of a
37.619 s decode span. The batch work was not merely slower than the rest;
it *was* the clock. We kept the run that proved it
`[docs/nvfp4-fast-lane-evidence-20260817.md]` `[ledger F-series
remediation]`.

The lesson is narrower than "Metal is bad at GEMM," which would be an
intuition rather than a finding. What the run actually says is that a
matvec-shaped machine, asked to do prefill-shaped arithmetic, spends nearly
all of its time in precisely the part of the workload the disaggregated
lane was invented to move elsewhere. So the engine does not try to make the
Mac a tensor-core machine. It sends batch work to the machine whose silicon
*is* that shape, and keeps the Mac on the matvec workload its wide SIMD
groups and unified memory are built for. The disaggregated lane is that
policy, wired — and the policy is code, not advice: the batch entry point
is `encode_nvfp4_w4a4_prequant_m16`
(`[crates/muser-engine/src/metal/encode/qkv.rs:13]`), and the predicate
that decides when the Mac is allowed to take that route lives at
`[crates/muser-engine/src/decode.rs:5946-5980]`.

## 29.5 Contrast 3 — the memory boundary: where the wire starts

The deepest difference between the two models is not in the kernels at all.
It is in what the word *pointer* is allowed to mean on each side, and if
you follow that single question far enough you arrive at the physical
picture the rest of this Part is built on. On the Mac, CPU and
GPU share one physical memory and one pointer space —
[Ch 3](03-unified-memory-and-buffers.md); a `StorageModeShared` buffer is
visible to both without copies. CUDA's programming model, whatever the
underlying packaging (the GB10's own memory topology is not recorded in any
Muser document this book quotes [unverified]), retains an explicit
host/device boundary — and the producer's connector code lives on that
boundary. Every KV layer the GPU computes must be *gathered to pinned host
memory* before TLS can ship it:

```python
# scripts/gx10/vllm/muser_vllm/connector.py:249-261 (excerpt)
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream(device=pair.device)
        current = torch.cuda.current_stream(device=pair.device)
        self._copy_stream.wait_stream(current)
        with torch.cuda.stream(self._copy_stream):
            canonical_pair = pair.contiguous()
            host_pair = torch.empty(
                canonical_pair.shape, dtype=torch.float16, device="cpu", pin_memory=True
            )
            copied_ns = time.perf_counter_ns()
            host_pair.copy_(canonical_pair, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(self._copy_stream)
```

That is the CUDA idiom in one quote: a dedicated **stream** for the copy, a
pinned-memory host tensor, a non-blocking device-to-host DMA, and an
**event** to fence it — plus the allocator subtlety documented right below
(`canonical_pair.record_stream(self._copy_stream)` keeps the device
allocation alive until the DMA has finished reading it
`[scripts/gx10/vllm/muser_vllm/connector.py:262-266]`).

On the Mac side, the mirror image: there is no gather, because there is no
boundary — the receiver's KV planes are Metal buffers in shared memory
([Ch 15](15-kv-store-and-the-ring.md)), and the sealing plan's receive-side
design wraps the network destination in a Metal buffer directly
(`makeBuffer(bytesNoCopy:…, .storageModeShared)` — zero copies, zero GPU
work) `[docs/disaggregated-prefill-sealing-plan-20260818.md §4
"Apple-side install"]`.

Put the two together and you get this Part's central physical picture:

> **Between a CUDA address space and a Metal address space, the network is
> the memory bus.**

Everything that follows from that — the pacing pin, the streaming schedule
that overlaps CUDA prefill with TLS sends, the EEE blackouts, the fsync tail
— is the cost of doing a "memory copy" across a wire. That is why
[Ch 31](31-the-wire-discipline.md) exists, and why the connector's
streaming seam (enqueue each intent whose layers exist, mid-prefill;
`[scripts/gx10/vllm/muser_vllm/connector.py:274-281]`) is designed exactly
like a DMA engine hiding behind compute.

## 29.6 Contrast 4 — streams and graphs versus one queue and one owner

Both models have to answer the same question — how do you keep an
accelerator busy when the work has dependencies? — and they answer it at
very different altitudes. Knowing which altitude you are standing on
decides who owns the ordering, and therefore who is to blame when the
ordering is wrong.

CUDA exposes concurrency as a first-class graph: multiple **streams**, each
an ordered queue; **events** to cross-synchronize; and CUDA graphs to
capture and replay whole dependency DAGs `[CUDA §streams]`. The connector
uses the small version of this — a forward thread, a sender thread, a copy
stream, and events (§29.5) — because the producer must overlap three
resources at once: tensor cores, PCIe-class D2H, and the NIC.

Metal's model is sparser and stricter: you *record* work into command
buffers and *commit* them; one `MTLCommandQueue` serializes
([Ch 2](02-metal-compute-model.md)). Muser's answer to "how do I get
concurrency?" is not more queues — it is **one scheduler owning one
accelerator**:

```rust
// crates/muser-engine/src/decode.rs:1020-1026
/// One owner for the shared Metal queue. Decode work is selected first and
/// resident sequence IDs rotate in ascending cyclic order, preventing a hot
/// slot from repeatedly reacquiring the accelerator ahead of its peers.
struct AcceleratorScheduler {
    state: Mutex<AcceleratorSchedulerState>,
    ready: Condvar,
}
```

Concurrency *inside* a submitted graph is expressed with a concurrent
compute encoder plus explicit resource barriers — the packed decode group
encodes 1..=4 sequences with "one concurrent encoder + one commit + one
wait" (`[crates/muser-engine/src/decode.rs:4920-4937]`, walk in
[Ch 34](34-scheduler-and-slots.md)) — and the *encoder type itself* is the
one place Muser kept an escape hatch back toward CUDA-style serial
ordering: `MUSER_SERIAL_PREFILL_DISPATCH` selects a serial (non-concurrent)
prefill encoder (`concurrent_prefill_dispatch` defaults to true;
`[crates/muser-engine/src/decode.rs:1331-1333]`).

Why no CUDA-graph analogue on the Mac? Because the problem CUDA graphs solve
— cheap re-submission of a fixed DAG — is solved differently here: the
52-layer graph is *encoded directly* by hand-written Rust (`encode_token`,
`encode_decode_group`) rather than captured, and ordering hazards are
managed by the single-queue owner plus tracked buffers and targeted barriers
(the full hazard story is [Ch 35](35-ordering-hazards-and-the-dispatch-gap.md)).
Worth one detour, because it is the kind of absence a reader can mistake
for an omission: there is a third design in this space, the ancestor
Ferrite book's compiled-VM replay, and Muser kept neither it nor graph
capture. That matters for the handoff story because it means the consumer's
execution order is authored in plain Rust that anyone auditing the seam can
read top to bottom — no captured graph, no replayed bytecode standing
between the source and what the GPU does. The divergence is documented
lineage, not an oversight
`[ferrite-book Ch 21]` (KEEP-AS-LINEAGE in the port audit).

## 29.7 Contrast 5 — when NOT to re-express a kernel: the pinned metallib

When is the right amount of kernel code to write *none*? Here, and the
reasoning is worth slowing down for, because it inverts the instinct that
owning the source of everything you dispatch is always the safer
engineering.

The lane needs the Mac's Q/K/V/gate/o projections to match what the
comparator (llama.cpp) computes — and later, what the producer computed
([Ch 32](32-precision-across-the-handoff.md)). Muser *could* have
re-expressed llama.cpp's ggml Metal kernels natively. It did the opposite:
it **pins llama.cpp's own prebuilt metallib** as the engine's third kernel
source ([Ch 4](04-pso-and-three-kernel-sources.md)):

```rust
// crates/muser-engine/src/metal/context.rs:122-131 (excerpt)
        let ggml_library_path = std::env::var_os("MUSER_GGML_METALLIB").map(PathBuf::from);
        let ggml_library = match ggml_library_path.as_ref() {
            Some(path) => Some(device.new_library_with_file(path).map_err(|message| {
                MetalError::GgmlLibrary { path: path.clone(), message }
            })?),
            None => None,
        };
```

The bet: **bit-parity with the comparator beats any performance you could
buy by rewriting.** The serving graph dispatches the pinned
`kernel_mul_mv_q*_K_f32` matvecs and the pinned `flash_attn_ext` family
(the route tables in [Ch 13](13-the-qkv-gate-matvec-family.md) and
[Ch 16](16-attention-decode-kernels.md)); a native re-expression would
re-litigate every accumulation order in those kernels for a gain the
dispatch-gap accounting says is not there ([Ch 35](35-ordering-hazards-and-the-dispatch-gap.md)).

A pin you cannot verify at runtime is not a pin; it is a wish. The
discipline that makes this one honest is **fingerprinting**, and we
inherited it as a scar rather than as a principle: in the ancestor project,
a metallib that simply failed to load fell back silently, and a benchmark
spent an afternoon carefully timing the wrong kernel family
(`[ferrite-book Ch 23 §23.7]`, lineage). Nothing errored. The numbers
looked plausible. That is the failure mode a pin invites — it moves the
identity of your kernels out of your build and into your environment, where
a missing file becomes a quiet substitution instead of a crash. So Muser
bakes the check into every route identity: `route_identity` reads the
metallib bytes and hashes them into the record —

```rust
// crates/muser-bench/src/main.rs:329-333, 341-346 (excerpt)
    let bytes = std::fs::read(&path).map_err(|error| {
        format!(
            "cannot fingerprint GGML metallib {}: {error}",
            path.display()
        )
    })?;
    let digest = Sha256::digest(bytes);
    // …
        matvec_route: "llama-ggml-metallib",
        ggml_metallib_sha256: Some(format!("sha256:{digest:x}")),
```

— so no measurement can silently report a kernel family it did not run
(compare the same discipline for the server-side receipt at
`MUSER_GGML_METALLIB_RECEIPT`,
`[crates/muser-server/src/node/mod.rs:83]`). The same thinking governs the
producer side of the lane: pinned vLLM commit, digest-pinned image
([Ch 28 §28.3](28-the-gx10-and-vllm-nvfp4-prefill.md)). When the two
vendors' stacks must agree, you do not *bridge* them; you **pin both ends
and hash the bridge**.

## 29.8 Contrast 6 — compiler discipline: fast math, strict f32, and running BOTH

The last contrast is the quietest and the most expensive. CUDA compilers
offer determinism knobs (and vLLM a `VLLM_BATCH_INVARIANT=1` mode) that
give the *producer self-consistency only* — "cross-vendor parity remains
our own pinned-op-order route plus calibrated drift bands"
`[docs/disaggregated-prefill-sealing-plan-20260818.md §4]`. Metal's
compiler has its own switch that matters just as much: **fast math**.

Muser compiles its kernels **twice, from the same sources, under different
flags** ([Ch 4](04-pso-and-three-kernel-sources.md)):

```rust
// crates/muser-engine/src/metal/context.rs:36-39
    /// Strict-f32 copy of the standalone Muse kernels.  The cross-vendor
    /// Q8 projection and integer NVFP4 routes must match CUDA's explicit
    /// scalar boundaries, while the ordinary serving kernels retain fast math.
    pub cross_vendor_library: Library,
```

The main library compiles with `set_fast_math_enabled(true)`, and the tree
records the reason in place: disabling it "materially slows attention, FFN,
and the norm/tiny-op stack without changing the imported GGML PSOs"
(`[crates/muser-engine/src/metal/context.rs:49-53]`). The strict-f32
`cross_vendor_library` then recompiles exactly the same two shader files,
`muse_reference.metal` and `nvfp4.metal`, with fast math **off**. Say that
again slowly, because it is the whole trick: the source does not change at
all. What changes is the license the compiler has to reassociate
floating-point arithmetic — and that license is the difference between a
result that matches CUDA and one that merely rounds to it.

A single flag, `MUSER_CROSS_VENDOR_QK`, routes QK-norm and attention
through the strict build, so that the Mac "must derive Q/K exactly the way
the producer did, or the KV is foreign by construction"
`[docs/disaggregated-prefill.md §What you need]`. The flag is not advisory:
serving refuses to start on the remote lane without it. We kept the trail —
the strict recompile at
`[crates/muser-engine/src/metal/context.rs:111-121]`, the routing sites at
`[crates/muser-engine/src/decode.rs:5645]` and
`[crates/muser-engine/src/metal/encode/norm.rs:115-116]`, and the
onboarding rule at
`[docs/one-button-onboarding.md §Starting the production consumer]`.

Why pay for two builds of the same source? Because **logit parity across
the handoff demanded it** — and we learned that over the campaign we came
to call the wizard's arithmetic-ABI chase. It is worth telling slowly,
because it is the story that turned a compiler flag into a protocol.

The setup: the consumer has to derive Q and K exactly as the producer did,
or the KV planes arriving over the wire describe a subtly different model
than the one the Mac is decoding. Our plan was the obvious one — compile
the seam kernels strictly, check the logits, declare victory. Attempts
10–30 of the combined-lane onboarding said otherwise. They kept failing on
single-bit seam divergences: not garbage, not a crash, one bit in one
element. That is the most exhausting failure mode there is, because
everything *looks* right.

So we stopped guessing and built a ladder. Comparing element by element
from layer-0 forward, the first mismatch appeared at `attn_norm-0` element
4 and propagated into K RoPE element 256 before surfacing downstream at
`attn_out-0` element 4,096. Two **arithmetic-ABI splits** fell out of that
trace: "CUDA's serial 128-dim attention reduction vs Metal's 32-lane tree,
and F32 vs F16 residual materialization" `[ledger
§2b, 2026-08-24]`. Neither is a bug in anybody's compiler. Both are legal,
defensible choices that two teams made independently — which is exactly why
no tolerance band would have found them for us; a tolerance would have
hidden them.

The fix was to stop treating the arithmetic as an implementation detail and
write it down as an interface: a versioned cross-vendor arithmetic ABI,
commits `27b5790`/`80f294f`, at a cost of 4–7 accelerator-hours of rework
`[ledger §2b, 2026-08-24]`. Attempt 31 then passed 7/7 with **exact
full logits** and payload rates 9.812/8.887/8.690 Gbps `[claims #9]`.

The lesson deserves saying twice, in two different registers. Put
mechanically: the compiler is part of the ABI, so a build flag that
reorders a reduction is as much a protocol change as renaming a field on
the wire. Put another way: two machines do not agree because they were
handed the same source. They agree because somebody pinned the order in
which the source is permitted to add things up, and then refused to paper
over the remainder with a tolerance.

Notice what this contrast did *not* become: a search for bitwise CUDA↔Metal
equality everywhere. Nobody achieves that — "that matches the state of
practice (nobody achieves bitwise CUDA↔Metal; llama.cpp uses tolerance-based
backend diffs)" `[docs/disaggregated-prefill-sealing-plan-20260818.md §4]`.
The discipline is surgical: pin the *seam* arithmetic exactly (strict f32,
pinned op order), bound the *interior* drift with calibrated bands, and let
[Ch 32](32-precision-across-the-handoff.md) carry the boundary between
them.

## 29.9 Tradeoffs

Three roads we did not take. Each was a genuine option rather than a straw
man, and none of them died against taste — each died against something
measured, and the measurement is what the reader should walk away with.

- **Re-express ggml kernels natively (rejected).** The case for it was
  respectable: owning the source of every kernel you dispatch is a sane
  default, and a native rewrite could in principle be tuned to Muser's own
  dispatch shapes rather than to llama.cpp's. What stopped us was not a
  benchmark but a correctness precedent from inside our own tree. When
  Muser fused a pair of adjacent norm ops — its *own* kernels, its own
  accumulation order, a change we were confident about — the fused path
  breached the 1e-4 logprob contract, coming in at 3.197e-4, and was
  rejected `[docs/decode-dispatch-gap-20260815.md
  §Rejected hybrid postmortem]`. Generalize from that, and the argument
  makes itself: if rearranging kernels *we* wrote can move the last digits,
  re-expressing kernels a foreign project wrote multiplies exactly that
  risk class, once per kernel, across the whole projection stack. So the
  pin stayed, and the measured consequence was a good one — llama's own
  bytes became the parity gate (the J0 anchor flip,
  [Ch 38](38-measuring-against-llama-cpp.md)), and when we wanted
  llama.cpp's one-reduction attention DAG (`flash_attn_ext_vec`) we adopted
  it *as a pinned kernel* rather than imitating it
  `[ledger Stage A close-out]`.
- **One strict seam library vs strict everywhere (chosen: seam only).**
  The tidy answer to the chase above would be to switch fast math off across
  the whole engine and never think about associativity again. The tree
  records why we did not: strict-f32 everywhere means materially slower
  attention/FFN/norm "without changing the imported GGML PSOs"
  `[crates/muser-engine/src/metal/context.rs:49-53]` — you would pay over
  the entire serving graph and buy nothing on the imported kernels that
  dominate it. The win is confined to the routes that must match CUDA
  scalar boundaries, so that is the only place strictness lives
  (`context.rs:36-39`). Drawing that line has a price when you draw it
  wrong, and we have the invoice: the wizard chase (§29.8) is what one
  f32-vs-f16 materialization on the wrong side of the seam cost.
- **Mac-native batch GEMM for prefill-scale work (rejected).** Told as a
  story earlier in the chapter; here is the receipt on its own line. Native
  spec decode reached 6.805 tok/s against the 107.9 bar, with 35.915 s of
  verify inside a 37.619 s span
  `[docs/nvfp4-fast-lane-evidence-20260817.md]`. That is the measured no-go
  that keeps batch work on the producer and matvec work on the Mac — a
  roofline argument that ended up encoded in a dispatch predicate rather
  than in a paragraph of advice.

## 29.10 What comes next

Strip the vendors away and this chapter was about *software contracts*:
pinned kernels with hashed identities, pinned commits, pinned op order at
the seam, two compiled libraries where one language would do if parity did
not matter. But contracts need a channel that cannot be forged or replayed.
The two machines in this lane talk to each other over a network, and
everything you just read about pinning identity is only as strong as the
transport that carries it. The next chapter is Handoff V2: mutually
authenticated TLS, an HMAC-sealed manifest, a durable replay ledger, and
the one-button wizard that proves a stranger's GX10 in three handoffs.

---

## References

- `[crates/muser-engine/src/shaders/muse_reference.metal:735-788]` —
  `muser_matvec_q4k_4r2s`: the SIMD-group matvec with `simd_sum` reduction.
- `[crates/muser-engine/src/shaders/ferrite/batch_m16_n32.metal:59-88]` —
  `m16_q4k_n32`: SIMD-group matrix types (`simdgroup_float8x8`) in the
  16-row batch kernel.
- `[crates/muser-engine/src/shaders/nvfp4.metal:226+]` — the
  `muser_nvfp4_matvec_c*` native-lane decode family.
- `[crates/muser-engine/src/metal/encode/qkv.rs:13]`,
  `[crates/muser-engine/src/decode.rs:5946-5980]` — the M16 NVFP4 batch
  route and its dispatch predicate.
- `[crates/muser-engine/src/metal/context.rs:36-39, 49-53, 111-131]` — the
  two compiled libraries, the fast-math justification comment, the pinned
  metallib load.
- `[crates/muser-engine/src/decode.rs:1020-1026, 1331-1333, 4920-4937,
  5645]` — the one-queue owner, the serial-dispatch flag, the packed decode
  group's single commit, the cross-vendor route.
- `[crates/muser-engine/src/metal/encode/norm.rs:115-116]`,
  `[crates/muser-server/src/node/mod.rs:83]` — cross-vendor norm routing;
  the metallib provenance receipt.
- `[crates/muser-bench/src/main.rs:305-346]` — `route_identity`: the
  metallib SHA-256 fingerprint baked into every route identity.
- `[scripts/gx10/vllm/muser_vllm/connector.py:184-285]` — the CUDA side of
  contrasts 3–4: sender thread (214), copy stream and pinned-host D2H
  (249–261), `record_stream` (262–266), streaming seam (274–281).
- `[docs/disaggregated-prefill-sealing-plan-20260818.md]` — §4 (W4A4
  FlashInfer/CUTLASS, `VLLM_BATCH_INVARIANT` limits, "nobody achieves
  bitwise CUDA↔Metal", Apple-side install design), §2 (connector streaming
  status).
- `[docs/disaggregated-prefill.md §What you need]` — the
  `MUSER_CROSS_VENDOR_QK=1` requirement in plain words.
- `[docs/one-button-onboarding.md §Starting the production consumer]` —
  serving refuses the remote lane without the cross-vendor flag.
- `[docs/nvfp4-fast-lane-evidence-20260817.md]` — the 6.805 tok/s native
  spec no-go.
- `[docs/decode-dispatch-gap-20260815.md §Rejected hybrid postmortem]` —
  the 3.197e-4 breach that prices re-expression risk.
- `[docs/launch-claims.md]` — #9 (attempt-31 exactness and wire rates).
- `[ledger §2b, 2026-08-24]` — the arithmetic-ABI chase, attempts 10–31,
  fixes `27b5790`/`80f294f`.
- `[docs/extraction-manifest.md §Stage 2]` — the fifteen byte-for-byte
  shader pulls behind contrast 1.
- `[CUDA §thread-hierarchy]`, `[CUDA §streams]` — NVIDIA CUDA C++
  Programming Guide (SIMT/warps; streams and events), vendor context only.
- `[Metal-SS §simd-group-functions]` — Metal Shading Language Specification
  (simd-group operations), vendor context only.
- `[ferrite-book Ch 21]`, `[ferrite-book Ch 23 §23.7]` — lineage: the
  compiled-VM replay Muser did not keep; the silent-metallib-fallback
  lesson behind fingerprinting (ancestor context).
- [glossary](../glossary.md) — terms introduced this chapter: warp,
  divergence penalty, tensor core (cross-ref [Ch 7](07-nvfp4-native-lane.md)),
  metallib pin, fast-math library, strict-f32 cross-vendor library,
  arithmetic ABI, D2H gather, record-then-commit.
