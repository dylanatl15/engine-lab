#!/usr/bin/env bash
# GPU window: Flash-Next with upstream --moe-cache-mib. CACHE_MIB and NJOBS from env.
set -u
OUT=/home/dylan/AI/engine-lab/results/window${TAG:-22}${TAG:-}; mkdir -p "$OUT"
TOKEN=$(grep -m1 '^MEDIA_WORKER_TOKEN=' /home/dylan/my-server/agent-stack/.env | cut -d= -f2- | tr -d '"'"'")
B=http://127.0.0.1:8100
H=(-H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json")
VRAM=/sys/class/drm/card1/device/mem_info_vram_used
M=/home/dylan/AI/llm-models/flash-next/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
SRV=/home/dylan/AI/engine-lab/src/build/bin/llama-server
LEASE=""; RENEW_PID=""; SPID=""
release() {
  [ -n "$SPID" ] && kill -INT "$SPID" 2>/dev/null && for _ in $(seq 1 30); do kill -0 $SPID 2>/dev/null || break; sleep 1; done; [ -n "$SPID" ] && kill -9 "$SPID" 2>/dev/null
  [ -n "$RENEW_PID" ] && kill "$RENEW_PID" 2>/dev/null
  sleep 3
  echo "vram at release: $(( $(cat $VRAM) / 1048576 )) MiB" | tee -a "$OUT/log"
  if [ -n "$LEASE" ]; then curl -s -m 10 -w ' http=%{http_code}' "${H[@]}" -X POST $B/lease/release -d "{\"leaseId\":$LEASE}" | tee -a "$OUT/log"; echo; fi
}
trap release EXIT
echo "start $(date -Is) cache=${CACHE_MIB:-9000} jobs=${NJOBS:-6}" | tee "$OUT/log"
R=$(curl -s -m 600 "${H[@]}" -X POST $B/lease/acquire -d '{"label":"flash-next moe-cache window 21"}')
LEASE=$(echo "$R" | python3 -c 'import sys,json;print(json.load(sys.stdin)["leaseId"])') || { echo "acquire failed: $R"; exit 1; }
echo "lease $LEASE $(date -Is)" | tee -a "$OUT/log"
( while true; do curl -s -m 5 "${H[@]}" -X POST $B/lease/renew -d "{\"leaseId\":$LEASE}" >/dev/null; sleep 15; done ) &
RENEW_PID=$!
for i in $(seq 1 60); do U=$(( $(cat $VRAM) / 1048576 )); [ "$U" -lt 600 ] && break; sleep 2; done
echo "vram free check: ${U} MiB" | tee -a "$OUT/log"
[ "${U:-99999}" -ge 600 ] && { echo "VRAM not freed"; exit 1; }
run_mode() {
  MODE=$1; shift
  env GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1 ENGINE_LAB_HYBRID=$MODE "$@" $SRV -m "$M" -ngl 99 --n-cpu-moe 48 --moe-cache-mib ${CACHE_MIB:-9000} -fa on -c ${CTX:-8192} -ub ${UB:-256} -b ${UB:-256} -t ${THREADS:-14} ${EXTRA2:-} --port 8098 -lv 4 > "$OUT/server_$MODE.log" 2>&1 &
  SPID=$!
  for i in $(seq 1 300); do
    kill -0 $SPID 2>/dev/null || { echo "server died (mode $MODE)"; tail -20 "$OUT/server_$MODE.log" | cut -c1-250 | tee -a "$OUT/log"; return 1; }
    curl -s -m 2 http://127.0.0.1:8098/health | grep -q ok && break; sleep 2
  done
  echo "mode $MODE up $(date -Is)" | tee -a "$OUT/log"
  python3 /tmp/claude-1000/-home-dylan/94a5f9dc-0848-4663-aa9c-2948d59ce2d7/scratchpad/det.py 2>&1 | tee "$OUT/det_$MODE.txt" | cut -c1-400 | tee -a "$OUT/log"
  python3 /tmp/claude-1000/-home-dylan/94a5f9dc-0848-4663-aa9c-2948d59ce2d7/scratchpad/det.py 2>&1 | cut -c1-60 | tee -a "$OUT/log"
  if [ -n "${PROBE:-}" ]; then LLAMA_URL=http://127.0.0.1:8098 timeout 3000 $PROBE 2>&1 | cut -c1-300 | tee -a "$OUT/log"; else ( cd /home/dylan/AI/engine-lab/trace; LLAMA_URL=http://127.0.0.1:8098 MAXTOK=${MAXTOK:-150} timeout 1200 python3 prompts.py ${NJOBS:-3} 2>&1 | tee -a "$OUT/log" ); fi
  kill -INT $SPID; for _ in $(seq 1 40); do kill -0 $SPID 2>/dev/null || break; sleep 1; done; kill -9 $SPID 2>/dev/null; SPID=""
  grep llama_moe_cache "$OUT/server_$MODE.log" | cut -c1-300 | tee -a "$OUT/log"
}
for m in ${MODES:-0 1}; do run_mode $m; done
echo "done $(date -Is)" | tee -a "$OUT/log"
