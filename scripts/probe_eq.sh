#!/bin/bash
S=/tmp/claude-1000/-home-dylan/94a5f9dc-0848-4663-aa9c-2948d59ce2d7/scratchpad
DT=300 python3 $S/det.py
sleep 10
echo "idle cpu: $(top -b -n2 -d3 -p $(pgrep -x llama-server | head -1) | tail -1 | awk '{print $9}')%"
