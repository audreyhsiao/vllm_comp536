# Prefix Cache 測試檔案摘要

## 測試檔案列表

| 檔案名稱 | 描述 | 預期 Hit Ratio | 說明 |
|---------|------|---------------|------|
| `test_prefix_sharing_simple.json` | 兩個完全相同的 user message | **~100%** | 最簡單的測試案例，第二個請求應該完全 hit 第一個請求的 prefix |
| `test_prefix_sharing_partial.json` | 第二個 message 是第一個的擴展 | **~70-80%** | 測試部分 prefix sharing，第二個 message 更長 |
| `test_prefix_sharing_multi_turn.json` | 多輪對話，第一個 turn 相同 | **~100%** (第一個 turn) | 測試多輪對話中的 prefix sharing |
| `test_prefix_sharing_long.json` | 兩個完全相同的長 user message | **~100%** | 測試長 prefix 的 cache hit |
| `test_prefix_sharing_from_trace.json` | 基於程式設計問題的測試 | **~100%** | 模擬真實使用場景 |
| `test_prefix_sharing_real_trace.json` | 基於真實 trace 的對話 | **~100%** | 使用真實 trace 中的對話格式 |

## 快速測試指令

### 1. 簡單測試（部分 hit rate 預期，因為 prompt 可能不完全相同）
```bash
# 執行測試
python sim/client_simulator.py \
  --dataset-file sim/test_prefix_sharing_simple.json \
  --server-url http://127.0.0.1:8000 \
  --model-name meta-llama/Llama-3.2-1B-Instruct \
  --prefix-sharing \
  --mode single

# 獲取統計資訊
curl http://127.0.0.1:8000/dump_prefix_stats > prefix_stats.json

# 驗證結果（降低預期值，因為可能只有部分 blocks 匹配）
python sim/verify_prefix_cache.py prefix_stats.json 0.6 0.2
```

**注意**: 由於 prefix cache 是基於 block hash 的，如果兩個請求的 prompt 不完全相同（例如 chat template 格式化差異），可能只有部分 blocks 會被 hit。這是正常的行為。

### 2. 部分共享測試（70-80% hit rate 預期）
```bash
python sim/client_simulator.py \
  --input-file sim/test_prefix_sharing_partial.json \
  --server-url http://127.0.0.1:8000 \
  --model-name meta-llama/Llama-3.2-1B-Instruct \
  --prefix-sharing \
  --mode single

curl http://127.0.0.1:8000/dump_prefix_stats > prefix_stats.json
python sim/verify_prefix_cache.py prefix_stats.json 0.7 0.15
```

### 3. 多輪對話測試
```bash
python sim/client_simulator.py \
  --input-file sim/test_prefix_sharing_multi_turn.json \
  --server-url http://127.0.0.1:8000 \
  --model-name meta-llama/Llama-3.2-1B-Instruct \
  --prefix-sharing \
  --mode multi

curl http://127.0.0.1:8000/dump_prefix_stats > prefix_stats.json
python sim/verify_prefix_cache.py prefix_stats.json 0.9 0.1
```

## 預期結果詳細說明

### test_prefix_sharing_simple.json
- **Request 1**: 
  - `prefix_hit_tokens`: 0
  - `prefix_hit_ratio`: 0.0
  - `hit_block_ids`: []
  
- **Request 2**: 
  - `prefix_hit_tokens`: ~16 tokens（1 個 block，block_size=16）
  - `prefix_hit_ratio`: ~66-100%（取決於 prompt 是否完全相同）
  - `hit_block_ids`: [0] 或更多（取決於匹配的 blocks）
  
**注意**: 如果兩個請求的 prompt 不完全相同（例如 chat template 格式化差異），可能只有部分 blocks 會被 hit。這是正常的，因為 prefix cache 是基於 block hash 的。

### test_prefix_sharing_partial.json
- **Request 1**: 
  - `prefix_hit_tokens`: 0
  - `prefix_hit_ratio`: 0.0
  
- **Request 2**: 
  - `prefix_hit_tokens`: ~25-30 tokens
  - `prefix_hit_ratio`: ~0.7-0.8 (70-80%)
  - 說明: 第二個 message 更長，所以 hit ratio 不是 100%

### test_prefix_sharing_multi_turn.json
- **Request 1 (conv_1, turn 1)**: hit_ratio = 0.0
- **Request 2 (conv_2, turn 1)**: hit_ratio = ~1.0 (100%)
- **Request 3 (conv_1, turn 2)**: hit_ratio = 0.0（不同的 user message）
- **Request 4 (conv_2, turn 2)**: hit_ratio = 0.0（不同的 user message）

### test_prefix_sharing_long.json
- **Request 1**: hit_ratio = 0.0
- **Request 2**: 
  - `prefix_hit_tokens`: ~60-80 tokens
  - `prefix_hit_ratio`: ~1.0 (100%)
  - `hit_block_ids`: [0, 1, 2, 3, 4] 或更多（約 4-5 blocks）

## 驗證要點檢查清單

對於每個測試案例，確認：

- [ ] 第一個請求的 `prefix_hit_ratio` = 0.0
- [ ] 第一個請求的 `prefix_hit_tokens` = 0
- [ ] 第一個請求的 `hit_block_ids` = []
- [ ] 第二個請求的 `prefix_hit_tokens` > 0
- [ ] 第二個請求的 `prefix_hit_ratio` 在預期範圍內
- [ ] 第二個請求的 `hit_block_ids` 不為空
- [ ] 被 hit 的 blocks 在 `blocks` 統計中有對應的 `hit_count` >= 1

## 常見問題

### Q: Hit ratio 不是 100%，是否正常？
A: 如果兩個 user message 完全相同，hit ratio 應該接近 100%。如果不是：
- 檢查兩個 user message 是否真的完全相同（包括空格、標點符號）
- 檢查 tokenizer 是否正確處理了文字
- 檢查 prefix key 計算是否正確

### Q: 如何知道實際的 token 數量？
A: 可以查看 `total_prompt_tokens` 欄位，或者使用 tokenizer 手動計算：
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokens = tokenizer.encode("your text here")
print(len(tokens))
```

### Q: Block size 如何影響結果？
A: Block size 決定了每個 block 包含多少 tokens。預設通常是 16 tokens/block。
- 如果 prompt 有 20 tokens，會使用 2 個 blocks（16 + 4）
- 如果 prompt 有 30 tokens，會使用 2 個 blocks（16 + 14，但可能因為對齊而使用 3 個 blocks）

## 從真實 Trace 選擇對話

如果要從真實 trace 中選擇對話進行測試：

1. 從 `ShareGPT_V3_unfiltered_cleaned_split.json` 中選擇一個對話
2. 複製該對話，創建兩個相同的對話（只有 id 不同）
3. 確保兩個對話的第一個 user message 完全相同
4. 執行測試並驗證 hit ratio

範例：
```bash
# 從 trace 中提取前 2 個對話並創建測試檔案
python -c "
import json
with open('sim/ShareGPT_V3_unfiltered_cleaned_split.json', 'r') as f:
    data = json.load(f)
    # 選擇第一個對話
    conv1 = data[0]
    # 創建第二個相同的對話（只有 id 不同）
    conv2 = {**conv1, 'id': conv1['id'] + '_copy'}
    test_data = [conv1, conv2]
    with open('sim/test_from_trace.json', 'w') as out:
        json.dump(test_data, out, indent=2)
"
```

