# engine-lab

Experiments to run Qwen3.8-Flash-Next (125B-class MoE, 6B active, IQ3_XXS GSQ-RCO) on a single
16 GB Radeon VII (Vulkan/RADV) with experts in host RAM, on top of upstream llama.cpp 547208d.

Result (one card, 64 GB DDR4 host): decode ~15-17 t/s (stock build 8.5), 14.7-15.3 t/s at 31K
context, prefill 56-138 t/s (stock ~40-50), needle recall 6/6 at ~29K, tool-calling gate 17/18.
Precision: the LRU expert cache is exact to within noise (PPL 6.1891 vs 6.1863).

## Layout
- `patches/lab-547208d-moe-cache-pinned-parallel.patch` - the llama.cpp changes (apply with `git apply` on 547208d):
  pinned staging + persistent futex-idling memcpy pool for `--moe-cache-mib` uploads, big-batch
  upload packing, MADV_POPULATE_READ prefault of host expert weights, timing probes, and
  experimental knobs (`ENGINE_LAB_*`; the hybrid CPU-miss path is dead code, it was slower).
- `run-flash-next.sh` - launcher (64K ctx, q8_0 KV, ub512, cache 8500 MiB, 6 copy threads).
- `scripts/` - GPU window harness (broker lease, run, restore), determinism/needle/PPL probes.
- `results/` - per-window logs; `results/runs/` - console output of each window.
- `trace/` - expert-routing trace tooling and workload.

## Findings
- Biggest wins: parallel memcpy into pinned staging (decode 10.5 -> 17 t/s) and prefaulting the mmap'd
  weights (prefill x2-3).
- Keep cache <= 9000 MiB at 64K ctx (more spills to GTT); `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` is required.
- Not wins: n-gram speculation on prose, MTP (head costs 2.6 GB of cache), CPU compute of misses,
  changing Vulkan submit granularity.
- Remaining cost is ~1.3 ms/layer of fixed overhead (graph record/submit, per-layer sync).
- Next: a second identical card (pooled / expert-parallel cache).

The toolchain (`tools/`) and the llama.cpp checkout (`src/`) are not tracked; see the patch header.
