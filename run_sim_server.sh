#!/usr/bin/env bash
set -euo pipefail

# Trace Simulator Server for vLLM (SimulatorExecutor)

# Usage:
#   TRACE=/abs/path/to/trace.jsonl MODEL=/abs/path/to/stub_model ./run_sim_server.sh
# Optional env:
#   HOST=127.0.0.1 PORT=8000 PY=.venv/bin/python
#   SIM_PREFILL_MS_PER_TOK=0 SIM_DECODE_MS_BASE=0 SIM_DECODE_MS_PER_SEQ=0

: "${TRACE:?Set TRACE to the path of your trace.jsonl}"
: "${MODEL:?Set MODEL to the path of your stub model directory}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
PY="${PY:-python}"

SIM_PREFILL_MS_PER_TOK="${SIM_PREFILL_MS_PER_TOK:-0}"
SIM_DECODE_MS_BASE="${SIM_DECODE_MS_BASE:-0}"
SIM_DECODE_MS_PER_SEQ="${SIM_DECODE_MS_PER_SEQ:-0}"

# Basic checks
[[ -f "$TRACE" ]] || { echo "ERROR: TRACE not found: $TRACE"; exit 1; }
[[ -d "$MODEL" ]] || { echo "ERROR: MODEL dir not found: $MODEL"; exit 1; }
[[ -f "$MODEL/config.json" ]] || { echo "ERROR: Missing $MODEL/config.json"; exit 1; }

export VLLM_NO_USAGE_STATS=1

exec "$PY" -m vllm.entrypoints.openai.api_server \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL" \
  --served-model-name trace-sim \
  --device cpu \
  --dtype float16 \
  --simulator \
  --sim-trace-path "$TRACE" \
  --sim-prefill-ms-per-tok "$SIM_PREFILL_MS_PER_TOK" \
  --sim-decode-ms-base "$SIM_DECODE_MS_BASE" \
  --sim-decode-ms-per-seq "$SIM_DECODE_MS_PER_SEQ" \
  --skip-tokenizer-init \
  --max-seq-len 128 \
  --disable-log-requests