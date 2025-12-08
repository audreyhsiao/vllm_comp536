# Milestone 2 Task 2 Summary: ShareGPT Trace Replay & Prefix Sharing Evaluation

## Overview

Implemented prefix sharing statistics collection system and analysis tools to evaluate prefix caching effectiveness by replaying ShareGPT traces in single-turn and multi-turn conversation modes.

## New Files Created

- **`vllm/prefix_stats_collector.py`**: Global statistics collector for prefix sharing metrics (~195 lines)
- **`analyze_prefix_stats.py`**: Analysis and visualization script for comparing single vs multi-turn results (~250 lines)

## Files Modified

- **`vllm/core/block_manager.py`**: Added `record_hit()` call when prefix cache blocks are reused
- **`vllm/engine/llm_engine.py`**: Added `record_prompt_length()` calls when processing requests
- **`vllm/entrypoints/openai/api_server.py`**: Added `/dump_prefix_stats` HTTP GET endpoint

## Key Features

### Prefix Statistics Collection
- Per-request tracking: prompt length, prefix hit tokens, hit ratio, hit block IDs
- Per-block tracking: hit count, reuse interval statistics (avg, min, max)
- Thread-safe collection with lock-based synchronization
- **Implemented by**: `PrefixStatsCollector`

### Statistics Integration with vLLM
- Records prompt length when requests are processed
- Records block hits when prefix cache blocks are reused
- Tracks reuse intervals with timestamps
- **Implemented by**: Integration in `block_manager.py`, `llm_engine.py`

### Statistics Export API
- HTTP endpoint `/dump_prefix_stats` for retrieving statistics
- JSON format output with per-request and per-block metrics
- **Implemented by**: `api_server.py` endpoint

### Data Analysis & Visualization
- CDF plots: prefix hit ratio, block hit count, reuse interval (log-scale)
- Bar chart: fraction of requests benefiting from prefix sharing
- Summary statistics: averages, medians, counts
- **Implemented by**: `analyze_prefix_stats.py`

### Experiment Modes
- Single-turn mode: Replay only first user turn per conversation
- Multi-turn mode: Replay all user turns in each conversation
- **Implemented by**: `ClientSimulator` with `--mode` parameter

## Statistics Collected

### Per-Request Metrics
- `total_prompt_tokens`: Total tokens in prompt
- `prefix_hit_tokens`: Number of tokens that hit prefix cache (clamped to total)
- `prefix_hit_ratio`: Fraction of prompt tokens that hit cache
- `hit_block_ids`: List of block IDs that were reused

### Per-Block Metrics
- `hit_count`: Total number of times block was reused
- `reuse_interval_avg`: Average time between reuses (seconds)
- `reuse_interval_min`: Minimum reuse interval
- `reuse_interval_max`: Maximum reuse interval
- `reuse_interval_count`: Number of reuse events tracked

## Visualizations Generated

### Prefix Hit Ratio CDF
- Cumulative distribution of per-request prefix hit ratios
- Compares single-turn vs multi-turn modes
- **Output**: `prefix_hit_ratio_cdf.png`

### Benefit Fraction Bar Chart
- Fraction of requests with prefix hits > 0
- Shows percentage of requests benefiting from prefix sharing
- **Output**: `benefit_fraction_bar.png`

### Block Hit Count CDF
- Distribution of how many times each block was reused
- Filters to blocks with hit_count >= 1
- **Output**: `block_hit_count_cdf.png`

### Reuse Interval CDF
- Distribution of time gaps between block reuses (log10 scale)
- Shows temporal patterns of cache reuse
- **Output**: `reuse_interval_cdf.png`

## Code Modifications

### vllm/prefix_stats_collector.py (New)
- `PrefixStatsCollector` class with thread-safe statistics tracking
- `RequestStats` and `BlockStats` dataclasses
- `record_prompt_length()`, `record_hit()`, `snapshot()`, `dump()` methods
- Global instance: `global_prefix_collector`

### vllm/core/block_manager.py
- Added `global_prefix_collector.record_hit()` call when prefix cache blocks are reused
- Tracks which blocks are hit for each request with request_id

### vllm/engine/llm_engine.py
- Added `global_prefix_collector.record_prompt_length()` calls when processing requests
- Records prompt token count for each request (from token_ids)
- Wrapped in try-except for error safety

### vllm/entrypoints/openai/api_server.py
- Added `/dump_prefix_stats` HTTP GET endpoint
- Calls `global_prefix_collector.dump()` to save statistics to JSON file
- Returns JSON file path and status

## Analysis Script Features

### Command-Line Interface
- `--single`: Path to single-turn statistics JSON file
- `--multi`: Path to multi-turn statistics JSON file
- `--out_dir`: Output directory for figures (default: "figs")

### Summary Statistics Output
- Number of requests and blocks
- Average and median prefix hit ratios
- Fraction of requests with hits
- Average and maximum block hit counts

## Dependencies Added

- `matplotlib`: For generating visualization plots
- `numpy`: For statistical calculations and CDF computation

## Integration

- **vLLM integration**: Statistics collection integrated into core vLLM components
- **Standalone analysis**: Analysis script works independently with exported JSON files
- **API access**: Statistics can be retrieved via HTTP endpoint during runtime

