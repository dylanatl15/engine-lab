#!/usr/bin/env bash
# Flash-Next on one Radeon VII with the lab MoE cache (measured 2026-10-18: decode ~17 t/s, 15 t/s at 31K ctx, prefill 56-138 t/s)
# 64K ctx: cache must stay <=9000 MiB (more spills to GTT). Optional MTP: add -md ~/AI/llm-models/unsloth-FlashNext/MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf --spec-type draft-mtp --spec-draft-n-max 3 and lower cache by ~2.6 GB (not a net win on prose)
# GPU must be free (take the broker lease first). Port 8098.
B=/home/dylan/AI/engine-lab/src/build/bin
cd $B && GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1 ENGINE_LAB_CPY_THREADS=12 LD_LIBRARY_PATH=. exec ./llama-server \
  -m /home/dylan/AI/llm-models/flash-next/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  -ngl 99 --n-cpu-moe 48 --moe-cache-mib 8500 -fa on -ctk q8_0 -ctv q8_0 -c 65536 -ub 512 -b 512 -t 14 --port 8098 "$@"
