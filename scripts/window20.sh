#!/usr/bin/env bash
# One GPU window: take broker lease, bench, always release. Token is read in-shell and never printed.
set -u
OUT=/home/dylan/AI/engine-lab/results/window20; mkdir -p "$OUT"
TOKEN=$(grep -m1 '^MEDIA_WORKER_TOKEN=' /home/dylan/my-server/agent-stack/.env | cut -d= -f2- | tr -d '"'"'")
B=http://127.0.0.1:8100
H=(-H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json")
VRAM=/sys/class/drm/card1/device/mem_info_vram_used
M=/home/dylan/AI/llm-models/unsloth-27B/Qwen3.8-27B-UD-IQ4_XS.gguf
NEW=/home/dylan/AI/engine-lab/src/build/bin/llama-bench
OLD=/home/dylan/AI/llm-bench/bin-unsloth
LEASE=""; RENEW_PID=""
release() {
  echo "gtt at end: $(( $(cat /sys/class/drm/card1/device/mem_info_gtt_used) / 1048576 ))" >> "$OUT/log"
  [ -n "$RENEW_PID" ] && kill "$RENEW_PID" 2>/dev/null
  pkill -x llama-bench 2>/dev/null; sleep 2
  echo "vram at release: $(( $(cat $VRAM) / 1048576 )) MiB" >> "$OUT/log"
  if [ -n "$LEASE" ]; then
    CODE=$(curl -s -m 10 -o "$OUT/release.body" -w '%{http_code}' "${H[@]}" -X POST $B/lease/release -d "{\"leaseId\":$LEASE}")
    echo "release http=$CODE body=$(cat "$OUT/release.body")" | tee -a "$OUT/log"
  fi
}
trap release EXIT

echo "start $(date -Is)" | tee "$OUT/log"
curl -s -m 5 $B/status | head -c 400 | tee -a "$OUT/log"; echo
curl -s -m 5 http://127.0.0.1:8190/health | head -c 300 | tee -a "$OUT/log"; echo

R=$(curl -s -m 600 "${H[@]}" -X POST $B/lease/acquire -d '{"label":"llama-bench window 20"}')
LEASE=$(echo "$R" | python3 -c 'import sys,json;print(json.load(sys.stdin)["leaseId"])') || { echo "acquire failed: $R"; exit 1; }
echo "lease $LEASE granted $(date -Is)" | tee -a "$OUT/log"
( while true; do curl -s -m 5 "${H[@]}" -X POST $B/lease/renew -d "{\"leaseId\":$LEASE}" >/dev/null; sleep 15; done ) &
RENEW_PID=$!

for i in $(seq 1 60); do
  U=$(( $(cat $VRAM) / 1048576 )); [ "$U" -lt 600 ] && break; sleep 2
done
echo "vram free check: ${U} MiB used" | tee -a "$OUT/log"
[ "${U:-99999}" -ge 600 ] && { echo "VRAM not freed, aborting"; exit 1; }



export GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1
run() { n=$1; shift; timeout 1000 $NEW -m "$M" -ngl 99 -r 1 -ub 256 -o md "$@" > "$OUT/$n.md" 2> "$OUT/$n.err"; echo "== $n: $*"; grep -E 'pp|tg' "$OUT/$n.md" | awk -F'|' '{print $(NF-2),$(NF-1)}' | tee -a "$OUT/log"; }
run pp_fa1_f16 -fa 1 -ctk f16 -ctv f16 -p 512 -n 0 -d 0,16384
run pp_fa0_f16 -fa 0 -ctk f16 -ctv f16 -p 512 -n 0 -d 0,16384
run pp_fa1_q8 -fa 1 -ctk q8_0 -ctv q8_0 -p 512 -n 0 -d 0,16384
run big_fa1_q8 -fa 1 -ctk q8_0 -ctv q8_0 -p 0 -n 64 -d 63488
echo "vram peak check: $(( $(cat $VRAM) / 1048576 )) MiB" | tee -a "$OUT/log"
echo "done $(date -Is)" | tee -a "$OUT/log"
