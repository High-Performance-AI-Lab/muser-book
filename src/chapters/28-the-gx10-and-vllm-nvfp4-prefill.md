# Chapter 28 — The GX10 node and vLLM NVFP4 prefill
> **status:** polished  ·  **path:** Muse Glimmer, pinned Muser tree
>
> *Prerequisites: [Ch 7](07-nvfp4-native-lane.md) (the NVFP4 format),
> [Ch 27](27-why-disaggregate.md) (why a prefill machine at all). This
> chapter is about the machine at the other end of the wire: what it is,
> what runs on it, and how it fails.*

---

## 28.1 Where we are

[Ch 27](27-why-disaggregate.md) ended on the supply-side question: the
payoff band is real, the KV payload is small, but only if you actually have
a prefill machine whose arithmetic is up to a 130k-token prompt. This
chapter is that machine. It is one node, it runs one pinned producer
process inside one docker container, and it has a stricter operational
culture than most servers you have owned — because
[Ch 27 §27.6](27-why-disaggregate.md) made it a TTFT single point of
failure, and because everything it computes crosses a trust boundary
([Ch 30](30-handoff-v2-transport.md), [Ch 32](32-precision-across-the-handoff.md)).

Three questions organise everything that follows. What is the machine, and
what wire does it hang on? What exactly runs on it, and how do we know it is
still the same thing tomorrow? And — the long one — what does it do when
something goes wrong?

## 28.2 The node and the wire it hangs on

Start with the physical facts, because two of them decide whether any number
in this chapter is admissible at all: what the machine is, and what path a
measurement travels over. Get the second one wrong and every throughput
figure you collect is fiction, however honestly you collected it.

The producer is **one ASUS GX10** — a DGX Spark-class machine built around
the **NVIDIA GB10** package, with an aarch64 host and an NVIDIA driver
`[docs/disaggregated-prefill.md §What you need]`
`[docs/one-button-onboarding.md §What "Add node" requires up front]`. The
canonical SSH alias is `producer-1`. Its exact GPU microarchitecture details
beyond "GB10, Blackwell-generation FP4 tensor cores" are not recorded in
any Muser document this book quotes [unverified] — and the book does not
need them; every claim that matters is a *measured* property of the
producer process, not a spec-sheet property of the chip.

The topology, which the operator cheat sheet fixes exactly and Figure 28.1
draws (`[AGENTS.md §The GX10 lane]`, mirrored in
`[docs/gx10-return-runbook-2026-08.md §Constants]`):

```text
   Mac (decode)                                   GX10 (prefill)
   ┌───────────────────────────┐                  ┌───────────────────────────┐
   │ en0 Ethernet              │   wired MikroTik  │ enp1s0f0np0               │
   │ 192.0.2.10                │◄────10GbE───────►│ 192.0.2.20                │
   │ 10Gbase-T full duplex     │   switched fabric │ (200 GbE port; link runs  │
   │                           │                  │  at the fabric's 10GbE)   │
   │ en1 Wi-Fi — NEVER a       │                  │ ssh alias: producer-1      │
   │ measurement path          │                  │ docker: resident producer │
   └───────────────────────────┘                  └───────────────────────────┘
```
*Figure 28.1: The enrolled transfer path since the 2026-08-23 topology
migration `[AGENTS.md §The GX10 lane]` `[docs/gx10-return-runbook-2026-08.md
§Correction]`. Historical (pre-migration) numbers used a retired direct
`retired /30` link `[docs/disaggregated-prefill-sealing-plan-20260818.md
§Operational topology amendment]`.*

Two disciplines are welded to this picture, and they will recur all through
[Ch 31](31-the-wire-discipline.md):

- **Wi-Fi is not a measurement path.** Mac Wi-Fi is `en1`; "a measurement is
  invalid if it routes there" `[scripts/gx10/README.md]`. Verify direct
  same-subnet routes in both directions before probing anything.
- **Re-prove the raw ceiling before trusting it.** A topology that was
  characterised yesterday is not a measured ceiling today.

The second rule earned itself during the migration. Before it, the direct
link had been characterised at ~9.4 Gbps single-stream each way
`[ledger T0]`, and we expected the switched fabric to land just under that
in both directions — a switch is nominally direction-agnostic, so a
symmetric result was the boring prediction. It was not what came back. The
product direction measured 9.256 Gbps, close enough to the old reference to
be unremarkable; the reverse direction measured 6.161 Gbps. We had no root
cause, and the two honest options were to keep hunting or to write it down,
so we wrote it down: the asymmetry is retained as a deviation, not promoted
to a pass `[ledger GX10 return 2026-08-23, attempts 3–4 + readiness entries]`
`[docs/gx10-return-runbook-2026-08.md §Execution annotation, attempt 4]`.
The lesson outlives this particular link. A fabric change invalidates the
ceiling in *both* directions until both directions have been re-measured,
which is why the discipline is written as a rule and not as a number.

The node also holds the lab's accelerator lease file — the *same* path the
Mac side uses, `/tmp/ferrite.gpu.lock`
(`NODE_GPU_LOCK = "/tmp/ferrite.gpu.lock"`,
`[scripts/gx10/vllm/resident_producer.py:43]`). The producer takes this
flock for its lifetime `[scripts/gx10/vllm/resident_producer.py:39-57]`,
which is how a lab full of GPU-hungry processes avoids stampeding one
Spark. When you see "the resident holds the lease" in a health receipt,
this is the lock being described
`[docs/gx10-return-runbook-2026-08.md §1.3]`.

## 28.3 The resident producer: one pinned identity chain

The question here sounds boring — what, exactly, is running over there? —
and it is the question the whole lane rests on. The Mac will later refuse
KV bytes that arrive from an identity it does not recognise, and for that
refusal to mean anything, "the producer" has to name something far more
precise than a hostname.

What runs on the GX10 is the **resident pinned Muse Glimmer NVFP4 prefill
producer** — a Python process living inside a docker container, running
vLLM with the NVFP4 checkpoint `[scripts/gx10/vllm/resident_producer.py:2]`.
"Pinned" is doing real work in that sentence. The producer refuses to start
against anything but the qualified vLLM commit:

```python
# scripts/gx10/vllm/resident_producer.py:21
PINNED_VLLM_COMMIT = "6adad08767583f52eb4d2122111af0bf638ed5e6"
```

```python
# scripts/gx10/vllm/resident_producer.py:68-69
    if config.get("vllm_commit") != PINNED_VLLM_COMMIT:
        raise ValueError("producer config does not pin the qualified vLLM commit")
```

The full identity chain is frozen in the onboarding identity document
`[scripts/gx10/vllm/native_onboarding_identity_v1.json]`: the NVFP4
checkpoint (`RedHatAI/Muse-Glimmer-30B-NVFP4`, revision `d5109a1…`,
23,409,256,035 bytes, per-file SHA-256 map), the producer image
(`muser/gx10-vllm-native:593b96a`, image id
`sha256:578888b2…`), and the vLLM overlay adapter digest. The enrolled
resident container at the pinned tree is `muser-redhat-native-f1-593b96a`
with host work directory `/home/<user>/.muser/lane/gx10/work/<deployment>`
`[docs/gx10-return-runbook-2026-08.md §Constants and evidence discipline]`.

One fact in that chain ties both ends of the lane together, and it is the
one worth carrying with you. The checkpoint's `chat_template.jinja` is
exactly 7,167 bytes, with SHA-256 `114f55eb…07965e`. That is the *same*
template hash the Mac-side release test asserts against the kquant GGUF:
two artifacts, produced by two vendors' toolchains through two unrelated
quantisation pipelines, agreeing on one template identity down to the byte.
Both halves of that assertion live in the tree, and we kept them there
deliberately — the Mac-side test at
`[crates/muser-server/src/chat_template.rs:237-261]`, the frozen producer
identity at `[scripts/gx10/vllm/native_onboarding_identity_v1.json]`. The
receiver will not accept a handoff whose identities do not match to the
byte ([Ch 30](30-handoff-v2-transport.md)).

The process is just as unwilling to be vague about the model it serves. It
asserts the shape from its own constants — 39 RoPE modules, head size 128,
context length 131,072 `[scripts/gx10/vllm/resident_producer.py:25-27]` —
so a checkpoint that quietly changed geometry fails at startup rather than
at handoff time. Every generate call runs under a default watchdog of
900 s `[scripts/gx10/vllm/resident_producer.py:24]`, and its vLLM KV-cache
allocation is bounded to 1–8 GiB by argument validation
(`1 << 30 <= args.kv_cache_memory_bytes <= 8 << 30`,
`[scripts/gx10/vllm/resident_producer.py:582]`).

Hold on to that last bound, because it is the easiest number in the lane to
misread. The next time you meet a "~7 GB payload" figure for the deep cell,
remember what it is describing: 7–8 GiB is the *producer's KV-cache
allocation* on its own GPU, not the wire payload. What actually crosses the
wire for that cell is 1.82 GB, and
[Ch 27 §27.4](27-why-disaggregate.md) derived it.

One more process belongs in the answer to "what runs over there," and the
line between the two is drawn on purpose. A small host-side daemon,
`muser_native_prefilld.py`, owns lifecycle and the authenticated control
channel, while "the vLLM image owns the GPU and Handoff V2 data plane" —
and the daemon's own docstring states the boundary as a prohibition: "No
cache bytes cross the control connection"
`[scripts/gx10/vllm/muser_native_prefilld.py:2-9]`. Control authority in one
process, cache bytes in the other, and no path by which a control message
can quietly become a data path.

## 28.4 NVFP4 prefill on tensor cores

[Ch 7](07-nvfp4-native-lane.md) introduced NVFP4 as a *weight format*: E2M1
values, one E4M3FN scale per 16, a per-tensor f32 `scale2`. On the Mac it
feeds SIMD-group matvecs — memory-bound decode. The same numeric format on
the GB10 plays a different role: **it is the working precision of batch
prefill on tensor cores**, the matrix-multiply units that consume FP4
operands directly. The producer runs vLLM W4A4 (4-bit weights, 4-bit
activations) via the FlashInfer/CUTLASS kernels of the pinned stack
`[docs/disaggregated-prefill-sealing-plan-20260818.md §4]`.

That is the [Ch 27](27-why-disaggregate.md) roofline made silicon: the
512-chunk's ~1,619 FLOPs/byte intensity is unreachable in FP32 on a Mac,
and trivially fed by dense FP4 matrix units. The measured consequence is
the entire payoff band of Table 27.2: the deep cell's producer compute
finishes far inside the 137.405 s remote median `[claims #6]`. If you want
a feel for *how* far inside, one integrated cell at 2,048 tokens put native
producer compute at 1.87 s — a dated packet and a single cell, and it is
scoped exactly that narrowly where it lives
`[docs/nvfp4-fast-lane-evidence-20260817.md §Measured product
numbers]`.

One hardware honesty note the sealing plan carries, because [Ch 32](32-precision-across-the-handoff.md)
needs it: on this GB10 generation the FP4 *conversion* path is emulated in
software (E2M1 round-to-nearest-even in code, not a hardware convert
instruction) — "a systematic producer-side rounding our drift band must
absorb" `[docs/disaggregated-prefill-sealing-plan-20260818.md §4]`. The
same format, on two machines, is not the same arithmetic. That is why the
lane's trust chapter exists.

## 28.5 Exact and native: one process, two modes

Why would one process need two personalities? Because the fast lane and the
trustworthy lane are not the same lane, and the engine needs both: one to
ship the product numbers, one to check the shipping lane against something
that cannot drift. The producer therefore has exactly two modes, selected by
an environment variable at start:

```python
# scripts/gx10/vllm/resident_producer.py:31-36
def producer_mode() -> str:
    """Return the closed producer lane selected for this process."""
    value = os.environ.get("MUSER_NVFP4_EXACT", "0")
    if value not in {"0", "1"}:
        raise RuntimeError("MUSER_NVFP4_EXACT must be exactly 0 or 1")
    return "exact" if value == "1" else "native"
```

- **`native`** (the default, `MUSER_NVFP4_EXACT=0`): the tensor-core NVFP4
  producer of §28.4 — the fast product lane.
- **`exact`** (`MUSER_NVFP4_EXACT=1`): the integer-dot verification
  producer, built on the llamacpp runtime (`spark_kv_export`
  `[scripts/gx10/llamacpp/spark_kv_export.cpp:1-3]`), which computes in
  integer/scalar arithmetic pinned to the CUDA compatibility graph. It is
  the deterministic anchor the native lane is *checked against* — [Ch 32](32-precision-across-the-handoff.md)
  is that story.

A subtlety the code map insists on and this book will not let you misread:
**`MUSER_NVFP4_EXACT` does not exist in Rust.** It is a producer-side
Python environment variable. What exists on the Mac is the *receiver-side
record* of which mode produced the KV — the enum in the cluster config:

```rust
// crates/muser-cluster/src/config.rs:13-18
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Nvfp4ProducerMode {
    Exact,
    Native,
}
```

It would be convenient if the two modes were interchangeable — take
whichever producer happens to be warm, take its KV, decode. They are not,
and the receiver is built to know it. Exact and native producers **derive
different target-cache identities**, so a cache filled by one is not even
addressable by the other
`[docs/muser-architecture.md §Durable and remote KV]`; neither mode may mix
KV with a differently-modeled decode lane; and a native-mode enrollment
cannot carry a DFlash identity at all. That last one
is not a warning in a doc, it is a hard config-validation error —
`native producer mode cannot enroll DFlash context geometry`
`[crates/muser-cluster/src/config.rs:128-132]`.

The split polices itself on the producer side too. Run the *native*
benchmark with the exact flag set and it refuses to start, on the grounds
that setting it "would invalidate the native-path claim"
`[scripts/gx10/vllm/benchmark_native_prefill.py:99-102]`. That is a
benchmark guarding the meaning of its own result rather than the
convenience of whoever is running it — the same instinct as the fail-closed
exit in the next section, one layer up.

## 28.6 Fail-closed by construction: exit 75

Now the culture. The producer **never serves degraded state**. Any
engine-touched error — including something as mundane as the receiver not
listening yet — kills the process with exit status 75
`[scripts/gx10/restart_resident_producer.py:5-8]`. There are exactly three
death sites in the request loop:

```python
# scripts/gx10/vllm/resident_producer.py:747-749
                if worker.is_alive():
                    print("[muser-nvfp4-producer] watchdog fired", flush=True)
                    os._exit(75)
```

The first is the watchdog: the 900 s ceiling elapsed on a generate that
never returned. The second is where the interesting decision lives:

```python
# scripts/gx10/vllm/resident_producer.py:786-795
            try:
                connection.sendall((_canonical(response) + b"\n"))
            finally:
                # A connector/send failure can leave vLLM's synchronous V1
                # engine request registered even though generate() raised.
                # Reusing that engine produced a host-side busy loop with no
                # GPU work. Fail closed after returning the error so an
                # orchestrator can restart from the persistent compile cache.
                if engine_touched and response["status"] != "ok":
                    os._exit(75)
```

That second site is the one with a story behind it, and the comment is the
post-mortem. The instinct when a send fails is to log it, drop the request,
and keep serving — the process is alive, the model is loaded, why throw
away a warmup that took minutes? Because the engine does not come back
clean. A connector or send failure can leave vLLM's synchronous V1 engine
request registered even though `generate()` raised, and reusing that engine
produced "a host-side busy loop with no GPU work": a producer that looked
healthy from the outside, held its port, burned a core, and computed
nothing. That is the worst possible state for a machine whose whole job is
to be trusted. The resolution is not a cleverer retry — it is to hand the
error back to the caller and then die, so an orchestrator can restart from
the persistent compile cache.

The third death site is quieter than the other two: three consecutive
non-engine errors also trip the exit
`[scripts/gx10/vllm/resident_producer.py:796-797]`.

Why 75? It is the EX_TEMPFAIL convention — "temporary failure, distinguish
me from a normal exit" — so a supervisor can tell "this producer died on
purpose" apart from other process deaths. The design bet: **a dead producer
is a known, recoverable state; a live producer serving wrong bytes is not.**
The cost of the bet is that restarts are routine; the next two sections are
the machinery that pays that cost.

## 28.7 Why `docker restart` is not enough — the ritual

Because the producer dies by design, bringing it back is a *ritual*, not a
command. The tool's own docstring is the specification:

```python
# scripts/gx10/restart_resident_producer.py:2-12 (module docstring, excerpt)
"""Restart a resident GX10 vLLM producer and wait for real readiness.

The resident producer exits fail-closed (status 75) after an engine-touched
error — including a receiver that is not listening yet — and a bare
`docker restart` is NOT enough to bring it back: the startup receipt and the
RoPE cache are created with O_EXCL, and a stale producer.sock blocks the new
bind. This script performs the full restart ritual and then waits for the
fresh startup receipt, which only appears after the model is loaded and the
warmup has run.
"""
```

Three pieces of stale state, three different failure modes if you skip
them: an `O_EXCL`-created startup receipt that the new process cannot
recreate over the old one; an `O_EXCL` RoPE cache with the same problem; a
stale Unix socket file blocking the bind. The ritual, as code:

```python
# scripts/gx10/restart_resident_producer.py:101-106
    rows: list[tuple[str, Path, Path | None]] = [
        ("move-aside", rope_cache, rope_cache.with_name(f"{rope_cache.name}.stale-{stamp}")),
        ("move-aside", receipt, receipt.with_name(f"{receipt.name}.stale-{stamp}")),
        ("remove", sock, None),
    ]
```

Note what it does *not* do: nothing is deleted — stale artifacts are moved
aside with a timestamp (evidence discipline even in recovery). And before
any of it runs, the tool checks the accelerator lease and refuses to
restart a producer whose lease is still held, printing a `fuser` holder
hint rather than weakening the refusal
`[scripts/gx10/restart_resident_producer.py:154-162]`. After `docker
restart`, it polls for the *fresh startup receipt* — "real readiness, not
process liveness" — because a running container that has not finished model
load (~2–3 min) is not a producer `[scripts/gx10/restart_resident_producer.py:21-23]`.

The runbook makes this the only legal path: "For an exited resident with a
free lease, use the node copy of `restart_resident_producer.py`; never use
bare `docker restart`" `[docs/gx10-return-runbook-2026-08.md §1.3]`. The
operator cheat sheet agrees (`[AGENTS.md §The GX10 lane]`), and adds a
second discipline that comes from the same pinned-identity thinking — this
one learned the expensive way. Patching a file inside the container looks
like a solved problem: the repo has the corrected version, `docker cp` it
in, restart, done. We did exactly that with `resident_producer.py`,
expecting the container to behave like a checkout of the tree. It does not.
The image was built from an older commit, the `muser_vllm` package inside it
had drifted from HEAD in the meantime, and the copied file failed on
`muser_vllm.native_capture`. The lesson is the one this chapter keeps
teaching from new angles: a container here is not a working copy, it is a
pinned identity that happens to contain files, and editing it as if it were
a checkout silently unpins it. Hence the rule:
**extract the file from the container, modify, verify, `docker cp`
back — never copy a file from repo HEAD wholesale into a container built
from an older commit** — and both the rule and the drift that taught it to
us are written down where an operator will meet them
`[AGENTS.md §The GX10 lane]` `[docs/disaggregated-prefill-sealing-plan-20260818.md §W1 finding 4]`.

## 28.8 The supervisor: recovery without flapping

A ritual needs a caller at 3 a.m. `supervise_resident_producer.py` is that
caller — "Keep it back: restart ritual + readiness wait in a loop, with a
failure latch" `[scripts/gx10/README.md]`:

```python
# scripts/gx10/vllm/supervise_resident_producer.py:65-67
def decide(consecutive_failures: int, max_failures: int) -> str:
    """Latch off at the failure ceiling; otherwise restart."""
    return "latch" if consecutive_failures >= max_failures else "restart"
```

The default ceiling is three consecutive failed starts
(`--max-consecutive-failures N` default 3,
`[scripts/gx10/vllm/supervise_resident_producer.py:124]`), with backoff
doubling per failure (`time.sleep(backoff * (1 << (consecutive - 1)))`,
`[scripts/gx10/vllm/supervise_resident_producer.py:113]`). The latch is the
interesting part: a supervisor that restarts forever converts "producer is
broken" into "producer is flapping," which is strictly worse — you get
half-loaded models racing watchdogs instead of one clear down state. When
the latch trips, the supervisor prints the container's last log lines and
exits 1, leaving the failure legible to an operator
`[scripts/gx10/vllm/supervise_resident_producer.py:110-112]`. A successful
restart resets the counter `[scripts/gx10/vllm/supervise_resident_producer.py:98-101]`.

What the operator sees when this machinery trips is precisely specified in
the runbook's health checklist: the container `Up`, `producer.sock` present
as a Unix socket, `LEASE HELD` (exit 1 from the probe is *expected* — it
means the resident owns the lease), and exactly the intended supervisor
active and not latched `[docs/gx10-return-runbook-2026-08.md §1.3]`.

## 28.9 What the evidence says about recovery

So the producer dies on purpose and a supervisor brings it back. Does that
actually work — and how far can the lane be pushed before it stops working?
The claims register keeps both answers, with their boundaries drawn
`[claims #13]`.

Take the cheapest test first: kill the producer outright and watch. It "was
detected and recovered with no operator action in testing, resuming
bit-identical payloads" `[docs/disaggregated-prefill.md §Operating it]` —
recovery that restores not merely service but the *same bytes*, which is the
only kind of recovery this lane can use. A harder case came out of the
kvpack ladder, whose stage 4 swapped the resident producer under the lane
and then restored a healthy supervised resident; we kept the node state
captured after that stage
`[receipt kvpack-ladder-20260820/attempt-13-…-stage4-naive/node-state-after-stage4.log]`.

Then we went after duration, because a lane that survives one kill can still
be terrible at staying up. The bounded deep soak ran eight consecutive
130,815-token handoffs, back to back. It passed on every axis we had
pre-registered: zero producer deaths, deterministic output, payload
throughput drifting 6.87→3.47 Gbps with every rep still at or above the
3.0 floor `[ledger "eight-handoff deep soak",
  2026-08-23]`
`[receipt final-campaign-20260823/attempt-4/soak/run-attempt-3/SOAK_VERDICT.json]`.
Reading that verdict, the temptation is obvious — the machinery works, call
sustained load solved.

Then, during the EEE-off sequence, a producer died on the *ninth*
consecutive deep handoff.

We do not know what the ninth handoff hit. What we do know is that eight was
not the ceiling we had quietly credited ourselves with: the passing soak and
the death sit on either side of one boundary, and only the death told us the
boundary was there. So the register
says "Sustained-deep-load stability remains open" `[claims #13]` and forbids
claiming otherwise, and the answer is not to re-run the eight-handoff soak
until it looks better — it is to design a run that would have caught this
one. The return runbook pre-registers exactly that: a bounded soak at N=12,
chosen to cover the observed ninth-handoff boundary
`[docs/gx10-return-runbook-2026-08.md §4]`.

That pairing is the fail-closed culture in miniature: a pass at N=8 and a
death at N=9 are reported *together*, and neither is smoothed into "stable"
or "unstable."

## 28.10 Tradeoffs

Three decisions in this chapter had a live alternative on the table, and in
each case the alternative was the more comfortable engineering choice. Here
is what taking it would have cost.

- **Fail-closed exit 75 vs serve-degraded.** The alternative — catch the
  error, keep the process, keep the port open — would hide the
  half-registered-engine busy loop the comment at
  `[scripts/gx10/vllm/resident_producer.py:789-794]` documents. Measured
  consequence: one tooling mistake (a refused receiver connection) is
  enough to kill the producer `[docs/disaggregated-prefill-sealing-plan-20260818.md §W1 finding 3]`
  — the lane accepted that fragility on the death side and bought back
  recovery with the supervisor. The economics work only because the
  producer is a TTFT SPOF, never a correctness SPOF ([Ch 27 §27.6](27-why-disaggregate.md)).
- **Supervised restart with a latch vs no supervision.** The sealing plan
  originally listed "no supervisor" as producer gap G3
  `[docs/disaggregated-prefill-sealing-plan-20260818.md §3]`; the latch
  design answers its other half (flapping). The kill/recovery evidence
  above is the measured outcome.
- **A second producer as failover?** Not in v0.1: one producer at a time is
  a receiver-side admission property
  (`[crates/muser-cluster/src/lib.rs:9-12]`), and the claims register bars
  multi-node failover wording `[claims #13]`. The local-prefill fallback is
  the qualified answer to producer outage.

## 28.11 What comes next

You now have both machines: a Mac that decodes from unified memory with
SIMD-group Metal kernels, and a GB10 that prefills NVFP4 on CUDA tensor
cores and dies loudly on exit 75. They must agree, bit for bit at the
seams, on what the KV means — and they were never designed to agree. The
next chapter is the CUDA-versus-Metal divide: the differences that actually
mattered to this engine, each one tied to a decision you can read in the
Muser tree.

---

## References

- `[AGENTS.md §The GX10 lane]` — the operator cheat sheet: `producer-1`
  (`192.0.2.20`), Mac `en0` `192.0.2.10`, Wi-Fi `en1` never a
  measurement path, status-75 fail-closed, restart tooling, container-file
  edit discipline.
- `[docs/gx10-return-runbook-2026-08.md]` — enrolled lane constants
  (container, work dir, socket, lease), preflight §1.2–1.3, the bounded
  soak §4, and the 2026-08-23 topology correction.
- `[scripts/gx10/README.md]` — the five diagnostic tools and the
  bottom-up diagnostic flow.
- `[scripts/gx10/restart_resident_producer.py]` — the restart ritual:
  docstring (lines 2–38), the plan rows (101–106), the lease guard
  (154–162), readiness wait (170–182).
- `[scripts/gx10/vllm/supervise_resident_producer.py]` — the supervisor:
  docstring (2–32), `decide` latch (65–67), backoff (113), defaults (124).
- `[scripts/gx10/vllm/resident_producer.py]` — pinned vLLM commit (21),
  model expectations (25–27), lease (39–57), `producer_mode()` (31–36),
  config pin check (68–69), KV-cache bounds (582), the three exit-75 sites
  (747–749, 786–795, 796–797).
- `[scripts/gx10/vllm/muser_native_prefilld.py:2-9]` — control-plane
  daemon scope ("no cache bytes cross the control connection").
- `[scripts/gx10/vllm/benchmark_native_prefill.py:99-102]` — the native
  benchmark's refusal of `MUSER_NVFP4_EXACT=1`.
- `[scripts/gx10/llamacpp/spark_kv_export.cpp:1-30]` — the integer-exact
  llama.cpp KV export producer.
- `[crates/muser-cluster/src/config.rs:13-18, 128-132]` —
  `Nvfp4ProducerMode` and the native-mode/DFlash enrollment refusal.
- `[scripts/gx10/vllm/native_onboarding_identity_v1.json]` — the frozen
  producer identity chain (checkpoint `d5109a1…`, image `593b96a`, the
  shared 7,167-byte chat template).
- `[crates/muser-server/src/chat_template.rs:237-261]` — the Mac-side
  assertion of the same template identity.
- `[docs/disaggregated-prefill-sealing-plan-20260818.md]` — §2 producer
  split, §3 gap G3, §4 (W4A4/CUTLASS, sm_121 software E2M1 RNE), §W1
  findings 3–4.
- `[docs/disaggregated-prefill.md]` — operating characteristics and the
  kill/recovery statement.
- `[docs/launch-claims.md]` — #8 (topology), #13 (producer self-recovery
  and its boundaries).
- `[ledger …]` — "eight-handoff deep soak" (2026-08-23); attempt-4
  readiness entries (post-rebuild TCP asymmetry).
- `[receipt final-campaign-20260823/attempt-4/soak/run-attempt-3/SOAK_VERDICT.json]`
  and `[receipt kvpack-ladder-20260820/attempt-13-…-stage4-naive/node-state-after-stage4.log]`
  — the soak and producer-swap evidence.
- [glossary](../glossary.md) — terms introduced this chapter: GB10/GX10,
  resident producer, producer mode (exact/native), exit 75, restart
  ritual, supervisor latch, O_EXCL startup receipt, accelerator lease.
