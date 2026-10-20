# engine-lab

**A 125B-parameter model at 24 tokens/second on one 16 GB graphics card from 2019.**

This repo is the patch set, launcher, test harness and raw lab notebook behind getting
Qwen3.8-Flash-Next (a 125B-class mixture-of-experts model) to run at interactive speed on a single
Radeon VII, with the weights sitting mostly in ordinary system RAM. Nothing here is a new
inference engine. It is a set of changes to [llama.cpp](https://github.com/ggml-org/llama.cpp) that
fix what turned out to be a *plumbing* problem, not a compute problem.

| One Radeon VII, 64 GB DDR4 host, IQ3_XXS quant | Stock offload | This repo |
|---|---|---|
| Decode, short prompt | 8.5 t/s | **24 t/s** (15–17 before the fused ops below) |
| Decode with 31K tokens already in context | 7.5 t/s | **14.7–15.3 t/s** |
| Prefill, 31.5K-token prompt | ~40 t/s (13 min) | **138 t/s (~4 min)** |
| Prefill, small prompts | 45–52 t/s | **56–68 t/s** |
| Context that fits in VRAM | spills at 64K | **64K** (q8_0 KV cache) |
| Needle-in-a-haystack at ~29K | n/a | 6/6 |

"Stock" is the Unsloth llama.cpp build b11541 with experts computed on the CPU, which is how you
would normally run a model this size on this machine.

## The idea in one minute

A mixture-of-experts model is mostly *dormant* weight. Flash-Next has 48 layers with 512 experts
each, but only 10 experts fire per token, so each token touches about 6B parameters even though the
file is 47 GB. That is why a model this big can run on modest hardware at all, and why it does not
fit on this card: 16 GB of VRAM against 47 GB of weights.

The usual workaround keeps the experts in system RAM and does the expert math on the CPU. On this
machine (a 14-core Xeon from 2016, AVX2 only) that tops out at 8.5 t/s. The CPU is the bottleneck.

llama.cpp already has a better idea in `--moe-cache-mib`: keep a **cache of recently used experts
in VRAM**, let the GPU do all the math, and copy only the missed experts up from RAM. Routing
has strong temporal locality (consecutive tokens share about a third of their experts), so a cache
holding roughly a quarter of all experts hits about 75% of the time.

On paper that should have been fast. Measured, it was 9–11 t/s. The patch in this repo is the story
of why, and of removing every reason one at a time.

## What was actually slow

Everything below was found with timers inside the process, since `perf` is locked down on the box.

1. **Uploads were stalling the queue.** Vulkan copies from ordinary (pageable) memory cost a
   full submit-and-wait per call. Staging the experts in a pinned buffer removes the stall.
2. **One thread could not copy fast enough.** Decode moves a couple of dozen megabytes of missed
   experts per layer, and a single `memcpy` thread out of a memory-mapped file ran at a fraction
   of the DDR4's ~65 GB/s. A small pool of persistent worker threads copies the misses into the
   pinned buffer in parallel. This one change took decode from **10.5 to 17 t/s**.
3. **The weights had never actually been read.** The model file is memory-mapped, so the first copy
   of every expert paid a minor page fault, which capped copies at ~2 GB/s and made prefill crawl.
   Prefaulting all host-resident expert weights at startup with `MADV_POPULATE_READ` (8 threads)
   made prefill **2–3× faster**.
4. **Large prefill uploads were chunked badly.** Big batches now go through two 96 MiB
   double-buffered halves of the pinned buffer, packed back to back, syncing only when the buffer
   being filled is still in flight.
5. **The worker pool burned a core while idle.** Idle workers spin briefly and then sleep on a
   futex, so the pool costs nothing between layers.

There is also a placement bug on this particular card that is easy to hit and expensive. The Radeon
VII has no resizable BAR, and RADV sometimes backs a ~150 MiB buffer with system RAM instead of
VRAM, which makes the recurrent-state ops about 30× slower. Setting
`GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` avoids it. The launcher sets it for you.

### Where the time goes now

The first version of this section said ~1.3 ms per layer. Two later changes cut that to about 1.0 ms:

- **A spinning memcpy pool with 64 KB chunks** (the expert upload goes through pinned staging memory):
  upload per layer fell from ~350 µs to ~65 µs. That alone was about +20% decode.
- **A fused `HC_MIX` op** (`ggml_hc_mix`, Vulkan only, single-token decode). Qwen3.8's hyper-connection
  mix is a chain of ~12 small ops per half layer (grouped rmsnorm, bf16 down projection, silu, bf16 up
  projection, sigmoid gate, stream mean, bf16 inject projection). Dispatch latency, not bandwidth,
  dominates them: ~75 µs per half layer on the GPU. The fused op is two dispatches and ~31 µs. The first
  stage streams each bf16 weight row exactly once with `x * gamma` staged in shared memory, the second
  finishes the norm and bottleneck in a prologue and does the gate dot products. `ENGINE_LAB_NO_HC_MIX=1`
  turns it off for A/B tests.

Result on Flash-Next, one Radeon VII, 12 copy threads, cache 8500 MiB: 200-token decode 19.4–19.9 t/s
(was 18.1–18.4 with the fused op disabled in the same session), per-layer gap 913 µs (was ~1170 µs
in that A/B). What is left is mostly GPU launch latency (~59 dispatches per layer) and the per-layer sync.

## Second pass: the GPU is the bottleneck (2026-10-19, 20 → 24.5 t/s)

Once uploads were out of the way, the engine turned out to be **GPU-bound**: kernel time per token
equals wall time per token. A decode step is ~40 ms = 48 MoE layers x ~0.84 ms, of which only a few
hundred microseconds per layer is real bandwidth. The rest is dependent small dispatches. Measured
floors on this card (see `kernlab/`):

- A chain of dependent dispatches with a compute-only barrier costs **4.7-5.5 µs per link**; in the real
  engine the floor is ~7 µs. A persistent kernel with an atomic grid barrier does the same in
  **1.9-2.3 µs** (240 workgroups, all co-resident), so a megakernel layer could save ~10-15%, but it is a
  very large project. Narrowing the pipeline barrier flags is worth 1.9 µs in kernlab and **zero** in
  the engine (tried, reverted).
- A matvec at the measured roofline (`kernlab/mv_*.comp`) reads ~130-160 GB/s for the 4 MB expert ops;
  they are latency-bound (ramp-up, tail, load -> LDS lookup -> reduce), not bandwidth- or ALU-bound.
- `bench/opbench.cpp` has ~27 µs of harness overhead per op, so isolated timings flatter nothing.

What went in, all in the Vulkan backend and `src/models/qwen4exp.cpp`, all **on by default** with `=0` opt-out:

| Knob | What it does | Effect |
|---|---|---|
| `LAB_GDN_PRE` | `gdn_pre.comp`: the conv/norm/gate prologue of the delta-net layer as one dispatch | ~9% of the GDN layer |
| `LAB_GDN_CACHE` | delta-net state read through the cache-friendly access mode | +1.3% |
| `LAB_MOE_COMBINE` | `ggml_moe_combine`: weighted expert sum + shared expert + residual in one dispatch | ~5% per layer with the next |
| `LAB_GATED_NORM` | `ggml_gated_norm`: rmsnorm * silu gate in one dispatch | (see above) |
| `LAB_MMVQ_GRID` | integer-dot MMVQ kernels for the grid quants (iq2_xxs/xs/s, iq3_xxs/s, iq4_nl). Default 1 = only iq3_s and iq4_nl, where they win (dense iq3_s -14.5 µs per block). The expert-side iq2/iq3_xxs kernels were 2-5 µs *slower* than the float path, so 0 = all float, 2 = all MMVQ | -0.3 to -0.5% of block time |
| `LAB_REPACK=1` (opt-in) | IQ4_XS weights repacked structure-of-arrays at load | ~3% GPU time on IQ4_XS ops only |

Plumbing that is in the tree but is a no-op unless asked for: an expert-id mailbox (`moe_mailbox.comp`),
deferred submit (`ENGINE_LAB_DEFER`, no gain), scheduler timers (`ENGINE_LAB_SCHT`), per-op timers
(`ENGINE_LAB_VKT`), graph dump (`ENGINE_LAB_DUMP_GRAPH`, `ENGINE_LAB_DUMP_FUSED`).

Precision: every new op matches its unfused chain in `test-backend-ops` (`MOE_COMBINE`, `GATED_NORM`,
`HC_MIX`, `GATED_DELTA_NET`, and the MMVQ types for MUL_MAT / MUL_MAT_ID). The fused ops reorder float
sums, so they differ at the 1e-6 level like the earlier `HC_MIX`; the integer-dot kernels quantize the
activation to q8_1, the same thing the CPU path and the stock MMVQ kernels already do. `LAB_MMVQ_GRID=0`
and the `=0` opt-outs restore the old numerics exactly.

Measured, same machine, 200-token decode: **24.3-24.6 t/s** with the default knobs (per-layer gap
~382 µs, readback+sync ~414 µs, upload ~80 µs). With the perf logger on, decode drops to ~9.6 t/s from
host overhead, so wall-clock comparisons are done interleaved and order-balanced (the first run of a
pair is systematically slower).

**Multi-slot throughput** (`-np 4`, decode-only aggregate): 14.8 t/s at one stream, 17.2 at two, 22.4 at
four. A two-token step costs ~1.7x a one-token step because the union of selected experts doubles, so
batched or speculative-verify decode is not a cheap win without a small-n expert matvec.

More dead ends: more threads alone, a transfer queue, eviction policy, cheap co-occurrence prefetch,
rows-per-workgroup knobs, forcing MMVQ everywhere, removing the scheduler's pre-callback sync alone,
command-pool reset guard, deferring the sync, an IQ4_XS LDS lookup table in the stock kernel, and
skip-op studies (invalid for whole-graph timing).

Still on the list: q6_K and q4_K/q5_K (including the 436 MB output matrix), the q2_0 expert down
projection (the single biggest op at ~62 µs per layer), expert gate+up+down+combine as one chain,
`HC_MIX` (still ~12% of block time), and the second card.

## Does it still give the same answers?

The cache only changes *where* an expert's weights come from, not what is computed, and the
measurements agree:

| Perplexity check (3×512 tokens, project text, single-token decode path) | PPL |
|---|---|
| Cache 8000 MiB (rerun, bit-identical) | 6.1891 |
| Cache 400 MiB (very different hit pattern) | 6.1863 |

A 0.05% spread is noise. The larger differences you will see between *kernel paths* (6.33 with
CPU-side experts, 6.47 with the GPU batch upload path) are numerics, not cache bugs. The CPU path
quantizes activations to q8_K and the GPU path to q8_1, and batched versus sequential delta-net
differ slightly. Those gaps are 2–5% perplexity.

Fused `HC_MIX` check (2026-10-19): on real activations the fused op matches the unfused chain to
~1e-7 relative at every layer (2208 layer/token pairs, `llama-hccheck` in `examples/eval-callback`, run
with `ENGINE_LAB_HCMIX_CHECK=1`). End-to-end the 1e-7 noise gets rounded differently further on in
the network (logits differ by up to ~0.06 on the first token), which on the 3-chunk text moves
perplexity 6.19 to 6.32. On 16 chunks the gap shrinks to 5.915 (unfused) vs 5.945 (fused), +0.5%,
well inside the ±0.21 statistical error. That is numerics noise of the same kind as the kernel-path
differences above, not a change in what is computed.

Behaviour checks on the full setup: needle retrieval 6/6 at ~29K, and a
tool-calling gate of 17/18 correct with no malformed JSON (the dense 27B model scored 18/18).

Honest caveat: this compares against the same quantized model run through other paths, not against
a full-precision reference. The quant itself (IQ3_XXS) is the largest precision cost, and this repo
does not change it.

## Things that did not work

Negative results are in the history too, because they are the useful part.

- **CPU compute of cache misses in parallel with the upload** (the `ENGINE_LAB_HYBRID` path, still in
  the patch but dead). The CPU expert math is real: 370 µs per layer at 14 threads. It was slower.
- **N-gram speculative decoding** on prose. Almost no drafts were accepted.
- **MTP speculative decoding.** It does work, with the standalone head from
  `unsloth/Qwen3.8-Flash-Next-GGUF` (the `-shared-` variants do not load on upstream). Acceptance
  was 64–95%. But the head costs ~2.6 GB of VRAM, which is a third of the expert cache. At equal
  VRAM it is break-even to worse on prose. It may win on code.
- **Tuning Vulkan submit granularity.** One submit per layer: 10 t/s. Finer splits: no change.
  The default was already near optimal.
- **A bigger cache.** Past ~9000 MiB at 64K context, buffers spill into system RAM and the gain
  halves. Keep it at or below 9000.

## Reproduce it

You need a Vulkan-capable AMD GPU with 16 GB, about 64 GB of RAM, and the GGUF
(`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, IQ3_XXS, ~47 GB plus a 28.8 GB n-gram file).

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 547208d
git apply /path/to/engine-lab/patches/lab-547208d-moe-cache-pinned-parallel-hcmix.patch
cmake -B build -DGGML_VULKAN=ON && cmake --build build -j --target llama-server
```

Then edit the model path in `run-flash-next.sh` and run it. The settings it uses, and why:

| Flag / env | Value | Why |
|---|---|---|
| `--n-cpu-moe 48` | all layers | experts live in host RAM |
| `--moe-cache-mib` | 8500 | the VRAM expert cache; stay ≤ 9000 at 64K |
| `-fa on -ctk q8_0 -ctv q8_0` | | flash attention with a quantized KV cache |
| `-c 65536 -ub 512` | | 64K context |
| `ENGINE_LAB_CPY_THREADS` | 12 | copy-pool size |
| `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM` | 1 | avoids the placement bug above |

Useful switches while debugging: `ENGINE_LAB_NO_PREFAULT`, `ENGINE_LAB_NO_BIGUP`,
`ENGINE_LAB_NO_STAGING`, and the timing probes (`ENGINE_LAB_VKT`, the cache stats printed on
shutdown).

## Trying a different GPU

Nothing in the cache path is tied to the Radeon VII, but the numbers above were tuned on it, and the
defaults reflect one specific balance of VRAM, PCIe bandwidth and host RAM. If you try another card,
these are the things to experiment with, roughly in order of how much they matter.

**Sizing (command-line flags)**

| Knob | Where | What to try on another GPU |
|---|---|---|
| `--moe-cache-mib` | flag | The big one. Set it to what is left after weights, KV cache and compute buffers. Hit rate scales with it, so a 24 GB card can hold a much larger cache and a 12 GB card a much smaller one. Watch VRAM use: if it overruns, buffers spill into system RAM and the gain collapses (we saw this above 9000 MiB at 64K context on 16 GB). |
| `-c`, `-ctk/-ctv` | flags | Context and KV quantization compete with the cache for VRAM. Trade context for cache or the reverse. |
| `-ub` / `-b` | flags | Prefill batch size. 512 fit at 64K context here, and 1024 was a little faster for prefill at 32K. Larger batches need larger compute buffers. |
| `-t` | flag | CPU threads for the non-expert work. 14 matched the physical cores of the test CPU. |

**Host-side copy path (`src/llama-moe-cache.cpp`)**

These depend on your PCIe generation and host memory bandwidth, not on the GPU vendor.

| Knob | Default | Notes |
|---|---|---|
| `ENGINE_LAB_CPY_THREADS` | 12 | Size of the memcpy pool. More threads help until you saturate DDR bandwidth. A dual-channel desktop will want fewer than a quad-channel workstation. |
| `big_half` (constant) | 96 MiB | Size of each half of the double-buffered prefill upload buffer. |
| `chunk` (constant) | 128 KiB | Per-worker copy chunk. Copies of 256 KiB or less skip the pool. |
| idle spin (`idle < 20000`) | about 20k pauses | How long workers spin before sleeping on the futex. Raise it for lower latency, lower it to save CPU. |
| prefault threads (the `8` in `stg_init`) | 8 | Threads used to `MADV_POPULATE_READ` the model at startup. |
| `ENGINE_LAB_NO_PREFAULT`, `_NO_BIGUP`, `_NO_STAGING` | off | Disable each optimization to measure what it is worth on your hardware. |

**Vulkan backend (`ggml/src/ggml-vulkan/ggml-vulkan.cpp`)**

| Knob | Notes |
|---|---|
| The `AMD_GCN` branch that sets `rm_kq_int = 4` | The one kernel default I changed. It is already limited to AMD GCN (Vega and Radeon VII). On RDNA, NVIDIA or Intel this change has no effect, but it is the first place to look if you want to tune matvec row counts for your architecture. |
| `ENGINE_LAB_RM_STDQ_INT`, `ENGINE_LAB_RM_KQ_INT`, `ENGINE_LAB_WG` | Override rows-per-workgroup and workgroup size from the environment so you can sweep them without rebuilding. |
| `ENGINE_LAB_FLOPS_DIV`, `GGML_VK_MAX_NODES_PER_SUBMIT` | Submit granularity. The default was best on the Radeon VII, but drivers differ. Try it once. |
| `ENGINE_LAB_VKT=1` | Prints CPU record-and-submit time per graph, which tells you how much of your per-layer cost is driver overhead. |

**Environment and platform**

- `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` works around a memory-placement problem on cards without
  resizable BAR. If your card has resizable BAR it is probably unnecessary, but it should be harmless.
  If decode speed is very uneven from run to run, check for it first.
- The copy pool uses Linux futexes and `madvise(MADV_POPULATE_READ)`, so it is Linux-only today.
  Windows and macOS need equivalents (`WaitOnAddress`, a prefault loop).
- Only Vulkan on AMD has been tested. The cache logic uses generic backend calls and should work
  with CUDA or HIP builds too, but that is untested. Please report results either way.

**How to tell what limits you**

Start the server with the cache stats on (they print at shutdown) and compare:

- Low hit rate: give the cache more VRAM, or shrink context and batch size.
- High hit rate but slow decode: the per-layer fixed cost dominates. Check `ENGINE_LAB_VKT` and
  the "decode(1 tok)" line for gap time versus upload time.
- Slow prefill: check the "big uploads" memcpy time. If the prefault step reports many seconds,
  your model is probably on a slow disk or not in the page cache yet.

## What is in the repo

| Path | Contents |
|---|---|
| `patches/` | the llama.cpp diff against 547208d (`git apply`); moe-cache, HC_MIX and the second-pass fused ops |
| `run-flash-next.sh` | launcher with the settings above |
| `scripts/` | GPU window harness, determinism, needle and perplexity probes |
| `results/` | per-run logs, with `results/runs/` holding each run's console summary |
| `trace/` | expert-routing trace tooling and workload generator |
| `kernlab/` | stand-alone Vulkan micro-benchmarks: matvec roofline, dispatch chains, persistent-kernel grid barrier |
| `bench/` | `opbench` (per-op timing), `chainbench`, `gdnpre_check` |

The numbers were taken on an HP Z640 (Xeon E5-2680 v4, 64 GB quad-channel DDR4, Radeon VII on
PCIe 3.0 x16, RADV). The scripts take a lease on a GPU broker that is specific to my setup, so
treat them as examples to adapt rather than something to run unchanged.

## Third pass: host gaps, DMA, and what did not work

- **GPU timeline.** A decode token spans ~49 ms: ~35 ms GPU busy, ~14 ms idle in ~65 gaps of ~210 us. Each
  layer's host round trip is memcpy (~35-54 us) + command recording (~79 us) + submit (~22 us) + launch (~12 us).
  About 72 expert misses per token (83% hit rate), ~1.65 MiB each, so ~119 MiB per token over PCIe at ~9.7 GB/s.
- **DEFER + coarse submits** (pre-record the next split, run the callback, then submit uploads and compute; one
  submission per chunk) is worth ~3-4.5% on decode only. Either alone is 0 or negative. It is on by default for
  single-token decode graphs (`ENGINE_LAB_DECODE_FAST=0` opts out); prefill is unaffected and output is identical.
- **In-process A/B** (`ENGINE_LAB_ABT=N`, `ENGINE_LAB_ABK=...`) flips knobs every N evaluations and reports
  ms/token per mode with a standard error. Separate runs of the same config vary by +-8%, so this is the only way
  to see 3% effects.
- **Dead ends:** GPU clock pinning (no wall gain, and it needs root), zero-copy host-pointer import of experts
  (the model is 76 GB, RAM 64 GB, GTT 31 GB), a transfer queue, eviction policies, co-occurrence prefetch,
  hybrid CPU-miss compute, deeper HC_MIX rewrites (it is within ~2% of its floor), MTP/n-gram speculation.
  An earlier per-token logger overstated some gaps; the numbers above are from the corrected instrumentation.
- **dmalab** (`kernlab/dmalab.cpp`): pinned to device copy 9.7 GB/s (queue family 0) and 9.63 (family 1), imported
  host pointer 8.5-8.8, shader read from host memory 12.2, device stream read 452. Compute plus copy overlap
  17 ms vs 21 ms serial.
- **Next-layer predictor** (`LAB_PRED=1`): applying layer L+1's router to layer L's FFN input predicts 64% of the
  experts a layer uses and 55% of the ones that miss the cache. That caps async prefetch at roughly 3-5% after
  overlap loss and wasted uploads, so it is parked behind other work.

## What is next

A second identical card. With two GPUs the interesting questions are pooling the expert cache
across both and splitting the per-layer overhead, which is now the dominant cost.

## Credits and license

The expert cache itself is upstream llama.cpp (`--moe-cache-mib`); this repo is the work of making
it fast on this hardware. The quantized model is by ISTA-DASLab. MIT licensed, like llama.cpp.
