#!/usr/bin/env bash

python -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port 8000 \
  --model facebook/opt-125m \
  --served-model-name trace-sim \
  --device cpu \
  --dtype float32 \
  --distributed-executor-backend sim \
  --sim-trace-path traces/sample.jsonl \
  --sim-prefill-ms-per-tok 0 \
  --sim-decode-ms-base 0 \
  --sim-decode-ms-per-seq 0 \
  --max-model-len 1024 \
  --enable-prefix-caching \
  --enforce-eager \
  --disable-log-requests \
  --skip-tokenizer-init \
  --disable-frontend-multiprocessing \
  --num-gpu-blocks-override 64 \
  --eviction-policy workload_aware