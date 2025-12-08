#!/usr/bin/env bash

json_file=$1
mode=$2

python sim/client_simulator.py \
  --max-concurrent-requests 500 \
  --poisson-lambda 50 \
  --dataset-file $json_file \
  --server-url http://127.0.0.1:8000 \
  --model-name facebook/opt-125m \
  --backend-model-name trace-sim \
  --mode $mode \
  --no-use-timestamps

