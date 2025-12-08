# Prefix Cache 正確性驗證指南

本文件說明如何使用測試 JSON 檔案來驗證 prefix cache 的正確性。

## 測試檔案說明

### 1. `test_prefix_sharing_simple.json`
**描述**: 兩個完全相同的 user message，應該有 100% prefix hit rate。

**預期結果**:
- **Request 1 (conv_1)**: 
  - prefix_hit_tokens: 0 (第一個請求，沒有 cache)
  - prefix_hit_ratio: 0.0
  - hit_block_ids: []
  
- **Request 2 (conv_2)**:
  - prefix_hit_tokens: 應該等於第一個 user message 的 token 數量（約 15-20 tokens，取決於 tokenizer）
  - prefix_hit_ratio: 接近 1.0 (100%)，因為兩個 user message 完全相同
  - hit_block_ids: 應該包含多個 block IDs（取決於 block_size，通常是 16 tokens/block）

**驗證步驟**:
```bash
python sim/client_simulator.py \
  --input-file sim/test_prefix_sharing_simple.json \
  --server-url http://127.0.0.1:8000 \
  --model-name meta-llama/Llama-3.2-1B-Instruct \
  --prefix-sharing \
  --mode single
```

### 2. `test_prefix_sharing_partial.json`
**描述**: 兩個 user message，第二個是第一個的擴展（包含額外的問題）。

**預期結果**:
- **Request 1 (conv_1)**:
  - prefix_hit_tokens: 0
  - prefix_hit_ratio: 0.0
  - hit_block_ids: []
  
- **Request 2 (conv_2)**:
  - prefix_hit_tokens: 應該等於第一個 user message 的 token 數量（約 25-30 tokens）
  - prefix_hit_ratio: 約 0.7-0.8 (70-80%)，因為第二個 message 更長
  - hit_block_ids: 應該包含多個 block IDs

**驗證步驟**:
```bash
python sim/client_simulator.py \
  --input-file sim/test_prefix_sharing_partial.json \
  --server-url http://127.0.0.1:8000 \
  --model-name meta-llama/Llama-3.2-1B-Instruct \
  --prefix-sharing \
  --mode single
```

### 3. `test_prefix_sharing_multi_turn.json`
**描述**: 多輪對話，兩個對話的第一個 user message 相同。

**預期結果**:
- **Request 1 (conv_1, first turn)**:
  - prefix_hit_tokens: 0
  - prefix_hit_ratio: 0.0
  
- **Request 2 (conv_2, first turn)**:
  - prefix_hit_tokens: 應該等於第一個 user message 的 token 數量（約 5-8 tokens）
  - prefix_hit_ratio: 接近 1.0 (100%)
  
- **Request 3 (conv_1, second turn)**:
  - prefix_hit_tokens: 0（因為第二個 user message 不同）
  - prefix_hit_ratio: 0.0
  
- **Request 4 (conv_2, second turn)**:
  - prefix_hit_tokens: 0（因為第二個 user message 不同）
  - prefix_hit_ratio: 0.0

**驗證步驟**:
```bash
python sim/client_simulator.py \
  --input-file sim/test_prefix_sharing_multi_turn.json \
  --server-url http://127.0.0.1:8000 \
  --model-name meta-llama/Llama-3.2-1B-Instruct \
  --prefix-sharing \
  --mode multi
```

### 4. `test_prefix_sharing_long.json`
**描述**: 兩個完全相同的長 user message，用於測試長 prefix 的 cache hit。

**預期結果**:
- **Request 1 (conv_1)**:
  - prefix_hit_tokens: 0
  - prefix_hit_ratio: 0.0
  
- **Request 2 (conv_2)**:
  - prefix_hit_tokens: 應該等於第一個 user message 的 token 數量（約 60-80 tokens）
  - prefix_hit_ratio: 接近 1.0 (100%)
  - hit_block_ids: 應該包含多個 block IDs（約 4-5 blocks，假設 block_size=16）

**驗證步驟**:
```bash
python sim/client_simulator.py \
  --input-file sim/test_prefix_sharing_long.json \
  --server-url http://127.0.0.1:8000 \
  --model-name meta-llama/Llama-3.2-1B-Instruct \
  --prefix-sharing \
  --mode single
```

### 5. `test_prefix_sharing_from_trace.json`
**描述**: 基於真實 trace 格式的測試案例，兩個相同的 user message。

**預期結果**:
- **Request 1 (trace_conv_1)**:
  - prefix_hit_tokens: 0
  - prefix_hit_ratio: 0.0
  
- **Request 2 (trace_conv_2)**:
  - prefix_hit_tokens: 應該等於第一個 user message 的 token 數量（約 20-25 tokens）
  - prefix_hit_ratio: 接近 1.0 (100%)
  - hit_block_ids: 應該包含多個 block IDs

**驗證步驟**:
```bash
python sim/client_simulator.py \
  --input-file sim/test_prefix_sharing_from_trace.json \
  --server-url http://127.0.0.1:8000 \
  --model-name meta-llama/Llama-3.2-1B-Instruct \
  --prefix-sharing \
  --mode single
```

## 如何驗證結果

### 1. 執行測試並獲取統計資訊

執行測試後，可以通過以下方式獲取 prefix cache 統計資訊：

```bash
# 方法 1: 使用 API endpoint
curl http://127.0.0.1:8000/dump_prefix_stats > prefix_stats.json

# 方法 2: 檢查 client_simulator 的輸出
```

### 2. 檢查統計資訊

打開 `prefix_stats.json`，應該會看到類似以下的結構：

```json
{
  "block_size": 16,
  "requests": {
    "request_1": {
      "total_prompt_tokens": 20,
      "prefix_hit_tokens": 0,
      "prefix_hit_ratio": 0.0,
      "hit_block_ids": []
    },
    "request_2": {
      "total_prompt_tokens": 20,
      "prefix_hit_tokens": 20,
      "prefix_hit_ratio": 1.0,
      "hit_block_ids": [0, 1]
    }
  },
  "blocks": {
    "0": {
      "hit_count": 1,
      "reuse_interval_count": 0,
      "reuse_interval_avg": 0.0,
      "reuse_interval_min": 0.0,
      "reuse_interval_max": 0.0
    },
    "1": {
      "hit_count": 1,
      "reuse_interval_count": 0,
      "reuse_interval_avg": 0.0,
      "reuse_interval_min": 0.0,
      "reuse_interval_max": 0.0
    }
  }
}
```

### 3. 驗證要點

對於每個測試案例，檢查：

1. **第一個請求**:
   - `prefix_hit_tokens` 應該是 0
   - `prefix_hit_ratio` 應該是 0.0
   - `hit_block_ids` 應該是空陣列

2. **第二個請求（有 prefix sharing）**:
   - `prefix_hit_tokens` 應該 > 0
   - `prefix_hit_ratio` 應該接近預期值（見各測試案例說明）
   - `hit_block_ids` 應該包含至少一個 block ID

3. **Block 統計**:
   - 被 hit 的 blocks 應該在 `blocks` 物件中有對應的統計資訊
   - `hit_count` 應該至少為 1

## 注意事項

1. **Token 數量可能因 tokenizer 而異**: 不同的 tokenizer 可能會將相同的文字 tokenize 成不同數量的 tokens，因此實際的 `prefix_hit_tokens` 可能會與預期略有不同。

2. **Block Size**: 預設的 block size 通常是 16 tokens。如果使用不同的 block size，block 數量會相應調整。

3. **Prefix Key 計算**: 預設情況下，prefix key 只使用第一個 user message。如果使用 `--use-full-conversation-prefix`，則會使用整個對話作為 prefix key。

4. **執行順序**: 確保兩個請求是按順序執行的（一個接一個），這樣第二個請求才能 hit 第一個請求建立的 cache。

## 自動化驗證腳本

可以創建一個簡單的驗證腳本來檢查結果：

```python
import json

def verify_prefix_stats(stats_file: str, expected_hit_ratio: float, tolerance: float = 0.1):
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    requests = stats.get('requests', {})
    request_ids = sorted(requests.keys())
    
    # 第一個請求應該沒有 hit
    first_req = requests[request_ids[0]]
    assert first_req['prefix_hit_ratio'] == 0.0, f"First request should have 0 hit ratio, got {first_req['prefix_hit_ratio']}"
    
    # 第二個請求應該有 hit
    if len(request_ids) > 1:
        second_req = requests[request_ids[1]]
        actual_ratio = second_req['prefix_hit_ratio']
        assert actual_ratio >= expected_hit_ratio - tolerance, \
            f"Second request should have hit ratio >= {expected_hit_ratio - tolerance}, got {actual_ratio}"
        assert len(second_req['hit_block_ids']) > 0, \
            f"Second request should have hit blocks, got {second_req['hit_block_ids']}"
        print(f"✓ Verification passed: hit ratio = {actual_ratio:.2f}, expected >= {expected_hit_ratio - tolerance:.2f}")
```

