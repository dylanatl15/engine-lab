# engine-lab

**A 125B-parameter model at 15–17 tokens/second on one 16 GB graphics card from 2019.**

This repo is the patch set, launcher, test harness and raw lab notebook behind getting
Qwen3.8-Flash-Next (a 125B-class mixture-of-experts model) to run at interactive speed on a single
Radeon VII, with the weights sitting mostly in ordinary system RAM. Nothing here is a new
inference engine. It is a set of changes to [llama.cpp](https://github.com/ggml-org/llama.cpp) that
fix what turned out to be a *plumbing* problem, not a compute problem.

| One Radeon VII, 64 GB DDR4 host, IQ3_XXS quant | Stock offload | This repo |
|---|---|---|
| Decode, short prompt | 8.5 t/s | **15–17 t/s** |
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

About 1.3 ms per layer, of which roughly 450 µs is the CPU recording and submitting the Vulkan graph
(~145 nodes per layer), about 110 µs is expert upload, and the rest is GPU compute and a per-layer
sync. The cache is no longer the limiting factor. Fixed per-layer overhead is.

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
git apply /path/to/engine-lab/patches/lab-547208d-moe-cache-pinned-parallel.patch
cmake -B build -DGGML_VULKAN=ON && cmake --build build -j --target llama-server
```

Then edit the model path in `run-flash-next.sh` and run it. The settings it uses, and why:

| Flag / env | Value | Why |
|---|---|---|
| `--n-cpu-moe 48` | all layers | experts live in host RAM |
| `--moe-cache-mib` | 8500 | the VRAM expert cache; stay ≤ 9000 at 64K |
| `-fa on -ctk q8_0 -ctv q8_0` | | flash attention with a quantized KV cache |
| `-c 65536 -ub 512` | | 64K context |
| `ENGINE_LAB_CPY_THREADS` | 6 | copy-pool size |
| `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM` | 1 | avoids the placement bug above |

Useful switches while debugging: `ENGINE_LAB_NO_PREFAULT`, `ENGINE_LAB_NO_BIGUP`,
`ENGINE_LAB_NO_STAGING`, and the timing probes (`ENGINE_LAB_VKT`, the cache stats printed on
shutdown).

## What is in the repo

| Path | Contents |
|---|---|
| `patches/` | the llama.cpp diff against 547208d (applies cleanly) |
| `run-flash-next.sh` | launcher with the settings above |
| `scripts/` | GPU window harness, determinism, needle and perplexity probes |
| `results/` | per-run logs, with `results/runs/` holding each run's console summary |
| `trace/` | expert-routing trace tooling and workload generator |

The numbers were taken on an HP Z640 (Xeon E5-2680 v4, 64 GB quad-channel DDR4, Radeon VII on
PCIe 3.0 x16, RADV). The scripts take a lease on a GPU broker that is specific to my setup, so
treat them as examples to adapt rather than something to run unchanged.

## What is next

A second identical card. With two GPUs the interesting questions are pooling the expert cache
across both and splitting the per-layer overhead, which is now the dominant cost.

## Credits and license

The expert cache itself is upstream llama.cpp (`--moe-cache-mib`); this repo is the work of making
it fast on this hardware. The quantized model is by ISTA-DASLab. MIT licensed, like llama.cpp.
