#!/bin/bash
S=/tmp/claude-1000/-home-dylan/94a5f9dc-0848-4663-aa9c-2948d59ce2d7/scratchpad
cd /home/dylan/AI/engine-lab/trace; MAXTOK=100 python3 prompts.py 2 2>&1 | tail -3
for f in mem_info_vram_used mem_info_gtt_used; do echo "$f $(( $(cat /sys/class/drm/card1/device/$f)/1048576 )) MiB"; done
