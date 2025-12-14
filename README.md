# Prefix-Sharing Cache Simulation on vLLM

This project extends **vLLM** with a trace-driven simulator and multiple cache-eviction policies (`lru`, `lfu`, `fifo`, `workload_aware`).  
The workflow includes:

1. starting a simulator-enabled vLLM server  
2. replaying conversation traces  
3. collecting prefix-sharing statistics  
4. generating analysis plots  

---

## 1. Installation

Install dependencies and ensure vLLM is correctly set up:

## 2. Start the Simulator Server

Use:

```bash
./run_sim_server.sh
```

Inside the script, you may modify key parameters:

- `--eviction-policy` - supports: lru, lfu, fifo, workload_aware
- `--num-gpu-blocks-override`
- `--block-size`
- any additional simulation flags

The server will run at: http://127.0.0.1:8000

## 3. Run the Client Simulator

Replay a dataset using:

```bash
./run_client.sh <dataset_json_path> <multi|single>
```

Arguments:

- <dataset_json_path> – path to the processed dataset JSON
- <multi> – simulate multi-turn conversations
- <single> – simulate only the first turn of each conversation

The client will send requests to the running vLLM server.

## 4. Dump Prefix-Sharing Statistics

After the client finishes, export prefix cache stats from the server:

```bash
curl http://127.0.0.1:8000/dump_prefix_stats -o prefix_stats.json
```

This generates a JSON report that includes:

- prefix hit tokens
- block hit counts
- block reuse intervals
- per-request prefix hit ratios
- eviction patterns

5. Generate Analysis Plots
Run:

```bash
python analyze_prefix_stats.py \
    --stats "TITLE" <prefix_stats.json> \
    --stats "TITLE2" <prefix_stats_2.json> \
    --stats "TITLE3" <prefix_stats_3.json> \
    --out_dir <output_dir>
```

This will produce four plots:

- fraction of requests with prefix hits
- CDF of block hit counts
- CDF of prefix hit ratio per request
- CDF of block reuse intervals (log scale)

Results are saved into <output_dir>.