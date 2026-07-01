#!/usr/bin/env bash
# infra/serve_and_run.sh — rent-friendly: launch an engine, wait, run a lab, tear down.
# GPU on only for the prepared sweep (the course's metered-GPU discipline).
#
# !!! Validated for shell syntax only (no GPU here). Check on your first rental. !!!
#
# Usage:
#   infra/serve_and_run.sh --model <hf-id> --lab labs/m05_paging/run_lab.py -- [lab args...]
#   infra/serve_and_run.sh --model <hf-id> --lab labs/m03_quantization/run_lab.py \
#       --serve-args "--quantization awq" -- --label int4_awq
# Everything after `--` is passed through to the lab's run_lab.py.
set -euo pipefail

ENGINE=vllm
MODEL=""
LAB=""
PORT=8000
KEEP=0
SERVE_ARGS=""
TIMEOUT=600

usage() { sed -n '2,13p' "$0"; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --engine)     ENGINE="$2"; shift 2 ;;
    --model)      MODEL="$2"; shift 2 ;;
    --lab)        LAB="$2"; shift 2 ;;
    --port)       PORT="$2"; shift 2 ;;
    --serve-args) SERVE_ARGS="$2"; shift 2 ;;   # extra engine flags, quoted
    --timeout)    TIMEOUT="$2"; shift 2 ;;       # startup wait (s)
    --keep)       KEEP=1; shift ;;
    --)           shift; break ;;
    -h|--help)    usage ;;
    *) echo "unknown arg: $1"; usage ;;
  esac
done
LAB_ARGS=("$@")  # passthrough to the lab

[[ -n "$MODEL" && -n "$LAB" ]] || { echo "error: need --model and --lab"; usage; }

LOG="serve_${PORT}.log"
echo ">> launching $ENGINE for $MODEL on :$PORT  (log: $LOG)"
# shellcheck disable=SC2086  # SERVE_ARGS is intentionally word-split into flags
if [[ "$ENGINE" == "vllm" ]]; then
  vllm serve "$MODEL" --port "$PORT" $SERVE_ARGS >"$LOG" 2>&1 &
elif [[ "$ENGINE" == "sglang" ]]; then
  python -m sglang.launch_server --model-path "$MODEL" --port "$PORT" $SERVE_ARGS >"$LOG" 2>&1 &
else
  echo "error: unknown --engine '$ENGINE' (use vllm|sglang)"; exit 1
fi
SERVER_PID=$!

teardown() {
  if [[ "$KEEP" == "0" ]]; then
    echo ">> tearing down engine (pid $SERVER_PID)"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  else
    echo ">> --keep set; engine (pid $SERVER_PID) left running on :$PORT"
  fi
}
trap teardown EXIT INT TERM

echo ">> waiting up to ${TIMEOUT}s for the server to be ready..."
for ((i = 1; i <= TIMEOUT; i++)); do
  if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    echo ">> ready after ${i}s"; break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "!! engine exited during startup. Tail of $LOG:"; tail -n 25 "$LOG"; exit 1
  fi
  sleep 1
done

echo ">> running: python $LAB --endpoint http://localhost:${PORT} --model $MODEL ${LAB_ARGS[*]}"
python "$LAB" --endpoint "http://localhost:${PORT}" --model "$MODEL" "${LAB_ARGS[@]}"
echo ">> lab finished."
