#!/usr/bin/env bash
# Decode benchmark: starts llama-server, generates N tokens a few times, prints decode t/s from the server's own timings.
# usage: MODEL=/path/first-shard.gguf [BIN=dir-with-llama-server] [CACHE_MIB=8500] [NCMOE=48] [N=200] [RUNS=3] scripts/bench.sh
# CACHE_MIB=0 turns the expert cache off (plain --n-cpu-moe), for a before/after comparison.
set -u
: "${MODEL:?set MODEL to the first gguf shard}"
BIN=${BIN:-$(dirname "$0")/../src/build/bin}; N=${N:-200}; RUNS=${RUNS:-3}; PORT=${PORT:-8098}
CACHE_MIB=${CACHE_MIB:-8500}; NCMOE=${NCMOE:-48}
CACHE_ARGS=(); [ "$CACHE_MIB" != 0 ] && CACHE_ARGS=(--moe-cache-mib "$CACHE_MIB")
LD_LIBRARY_PATH=$BIN GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1 ENGINE_LAB_CPY_THREADS=${CPY_THREADS:-12} \
  "$BIN/llama-server" -m "$MODEL" -ngl 99 --n-cpu-moe "$NCMOE" "${CACHE_ARGS[@]}" -fa on -c 8192 -ub 512 -b 512 \
  -t ${THREADS:-14} --port "$PORT" > "${LOG:-/tmp/bench-server.log}" 2>&1 &
SPID=$!; trap 'kill -INT $SPID 2>/dev/null; wait $SPID 2>/dev/null' EXIT
for _ in $(seq 1 300); do
  kill -0 $SPID 2>/dev/null || { echo "server died, see ${LOG:-/tmp/bench-server.log}"; exit 1; }
  curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q ok && break; sleep 2
done
req() { curl -s -m 600 http://127.0.0.1:$PORT/completion -d "{\"prompt\":\"Write a long story about a lighthouse keeper who finds a door in the sea.\",\"n_predict\":$N,\"temperature\":0,\"cache_prompt\":false}"; }
req > /dev/null   # warm-up: fills the expert cache
for i in $(seq 1 "$RUNS"); do
  req | python3 -c 'import sys,json;t=json.load(sys.stdin)["timings"];print("run %d: decode %.2f t/s (%d tokens), prefill %.1f t/s"%('"$i"',t["predicted_per_second"],t["predicted_n"],t["prompt_per_second"]))'
done
