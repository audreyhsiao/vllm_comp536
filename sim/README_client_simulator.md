# Client Simulator for ShareGPT Dataset Replay

This module implements a client simulator that replays collected prompts from ShareGPT dataset for evaluating prefix sharing in chatbot workloads.

## Features

- **ShareGPT Dataset Loading**: Supports loading from HuggingFace datasets or local JSON files
- **Timing Simulation**: Uses timestamps from dataset if available, otherwise uses Poisson distribution
- **Chat Template Formatting**: Automatically formats conversations using model-specific chat templates
- **Prefix Sharing Support**: Generates prefix keys for prefix sharing optimization

## Installation

```bash
# Install required dependencies
pip install transformers datasets numpy
```

## Usage

### Basic Usage

```bash
# Load from HuggingFace dataset
python sim/client_simulator.py \
    --dataset-name "anon8231489123/ShareGPT_Vicuna_unfiltered" \
    --max-samples 100 \
    --model-name "meta-llama/Llama-3.2-1B-Instruct" \
    --max-tokens 512

# Load from local JSON file
python sim/client_simulator.py \
    --dataset-file /path/to/sharegpt.json \
    --model-name "meta-llama/Llama-3.2-1B-Instruct" \
    --max-tokens 512
```

### With Custom Chat Template

```bash
python sim/client_simulator.py \
    --dataset-name "anon8231489123/ShareGPT_Vicuna_unfiltered" \
    --model-name "meta-llama/Llama-3.2-1B-Instruct" \
    --chat-template /path/to/custom_template.jinja \
    --max-tokens 512
```

### Timing Options

```bash
# Use timestamps from dataset (default)
python sim/client_simulator.py \
    --dataset-file sharegpt.json \
    --use-timestamps \
    --model-name "meta-llama/Llama-3.2-1B-Instruct"

# Use Poisson distribution for timing
python sim/client_simulator.py \
    --dataset-file sharegpt.json \
    --no-use-timestamps \
    --poisson-lambda 2.0 \
    --model-name "meta-llama/Llama-3.2-1B-Instruct"
```

### Prefix Sharing Options

```bash
# Enable prefix sharing (default)
python sim/client_simulator.py \
    --dataset-file sharegpt.json \
    --prefix-sharing \
    --model-name "meta-llama/Llama-3.2-1B-Instruct"

# Use full conversation as prefix key
python sim/client_simulator.py \
    --dataset-file sharegpt.json \
    --prefix-sharing \
    --use-full-conversation-prefix \
    --model-name "meta-llama/Llama-3.2-1B-Instruct"
```

## Command Line Arguments

### Dataset Options
- `--dataset-file`: Path to local ShareGPT JSON file
- `--dataset-name`: HuggingFace dataset name (default: "anon8231489123/ShareGPT_Vicuna_unfiltered")
- `--max-samples`: Maximum number of samples to load

### Model and Template Options
- `--model-name`: HuggingFace model name for chat template
- `--tokenizer-path`: Path to tokenizer (if different from model-name)
- `--chat-template`: Custom chat template string or file path

### Timing Options
- `--use-timestamps`: Use timestamps from dataset if available (default: True)
- `--no-use-timestamps`: Disable using timestamps, use Poisson instead
- `--poisson-lambda`: Lambda parameter for Poisson distribution (requests per second, default: 1.0)

### Simulator Backend Options
- `--block-size`: KV cache block size (default: 16)
- `--num-blocks`: Number of KV cache blocks (default: 200000)
- `--max-prefill-tokens`: Maximum prefill tokens per batch (default: 8192)
- `--max-decode-batch`: Maximum decode batch size (default: 32)

### Request Options
- `--max-tokens`: Maximum tokens to generate per request (default: 512)
- `--prefix-sharing`: Enable prefix sharing (default: True)
- `--no-prefix-sharing`: Disable prefix sharing
- `--use-full-conversation-prefix`: Use full conversation as prefix key

## ShareGPT Dataset Format

The expected format for ShareGPT dataset is:

```json
[
  {
    "id": "conversation_id",
    "conversations": [
      {"from": "human", "value": "User message"},
      {"from": "gpt", "value": "Assistant response"},
      {"from": "human", "value": "Another user message"},
      {"from": "gpt", "value": "Another assistant response"}
    ],
    "t": 1234567890.123  // Optional timestamp
  }
]
```

## Chat Template Formatting

The client simulator automatically detects and uses the chat template from the tokenizer. For example, Llama-3.2-1B-Instruct uses a specific template that formats conversations with special tokens.

You can also provide a custom chat template:
- As a file path: `--chat-template /path/to/template.jinja`
- As a string: `--chat-template "{{ messages | apply_chat_template }}"`

## Output

The simulator prints statistics at the end:
- Total time
- Total/completed/failed requests
- Success rate
- Average latency
- Backend metrics (KV evictions, templates, etc.)

## Example Output

```
Loading ShareGPT dataset...
Loaded 100 conversations
Starting simulation...
============================================================
Simulation Statistics
============================================================
Total time: 45.23 seconds
Total requests: 100
Completed requests: 98
Failed requests: 2
Success rate: 98.00%
Average latency: 234.56 ms

Backend report:
  Finished requests: 98
  KV evictions: 5
  KV templates: {'abc123def456': 3, 'xyz789uvw012': 2}
============================================================
```

