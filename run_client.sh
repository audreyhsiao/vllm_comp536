#!/usr/bin/env bash

python sim/client_simulator.py \
  --max-concurrent-requests 500 \
  --poisson-lambda 50 \
  --dataset-file sim/clean1000000.json \
  --server-url http://127.0.0.1:8000 \
  --model-name facebook/opt-125m \
  --backend-model-name trace-sim \
  --mode single \
  --no-use-timestamps

