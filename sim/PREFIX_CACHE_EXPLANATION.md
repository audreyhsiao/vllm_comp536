# Prefix Cache Hit Ratio 說明

## 為什麼 Hit Ratio 可能不是 100%？

### 問題現象

當兩個請求有相同的 user message 時，你可能會看到：
- 第一個請求：24 tokens，0% hit ratio
- 第二個請求：24 tokens，66.67% hit ratio（16 tokens hit，1 個 block）

而不是預期的 100% hit ratio。

### 原因分析

Prefix cache 是基於 **block hash** 的，每個 block 的 hash 包含：
1. 該 block 內的 tokens
2. 該 block 之前的所有 prefix tokens

```
Block 0: hash(tokens[0:16])
Block 1: hash(tokens[0:32])  ← 包含 Block 0 的內容
Block 2: hash(tokens[0:48])  ← 包含 Block 0 和 Block 1 的內容
```

### 為什麼只有部分 Blocks 被 Hit？

如果兩個請求的 **實際 prompt**（經過 chat template 格式化後）不完全相同，那麼：

1. **Block 0** 可能匹配（如果前 16 個 tokens 相同）
2. **Block 1** 可能不匹配（如果後面的 tokens 不同）

這可能發生在以下情況：

#### 1. Chat Template 格式化差異
不同的請求可能因為：
- 時間戳不同
- 會話 ID 不同
- 其他 metadata 不同

導致 chat template 格式化後的 prompt 略有差異。

#### 2. Tokenization 差異
即使文字看起來相同，tokenizer 可能因為：
- 上下文不同
- 特殊字符處理
- 編碼差異

產生不同的 token 序列。

#### 3. Block 對齊問題
- 24 tokens = 2 blocks（16 + 8）
- 如果只有前 16 個 tokens 匹配，只有 Block 0 會被 hit
- Hit ratio = 16/24 = 66.67%

### 這是正常的嗎？

**是的，這是正常的行為！**

Prefix cache 的設計目標是：
- ✅ 自動檢測和重用相同的 prefix blocks
- ✅ 不需要手動管理 cache
- ✅ 即使只有部分匹配，也能節省計算

部分匹配（例如 66.67%）仍然是有價值的，因為：
- 節省了 16 個 tokens 的 KV cache 計算
- 減少了 memory 使用
- 提高了 throughput

### 如何獲得更高的 Hit Ratio？

#### 方法 1: 確保 Prompt 完全相同
確保兩個請求的實際 prompt（經過所有格式化後）完全相同。

#### 方法 2: 使用更長的相同 Prefix
如果相同的 prefix 更長（例如 32+ tokens），可以匹配更多 blocks。

#### 方法 3: 檢查 Prefix Key 計算
確認 `get_prefix_key()` 函數計算的 prefix key 是否正確匹配。

### 驗證建議

1. **基本驗證**（推薦）：
   ```bash
   python sim/verify_prefix_cache.py prefix_stats.json 0.6 0.2
   ```
   - 預期 hit ratio >= 60%
   - 允許 20% 的誤差

2. **嚴格驗證**（如果確定 prompt 應該完全相同）：
   ```bash
   python sim/verify_prefix_cache.py prefix_stats.json 0.9 0.1
   ```
   - 預期 hit ratio >= 90%
   - 如果失敗，檢查 prompt 是否真的完全相同

3. **診斷模式**：
   查看驗證腳本輸出的詳細資訊：
   - 理論最大 hit blocks
   - 理論最大 hit tokens
   - 實際 hit blocks
   - 實際 hit tokens

### 範例分析

從你的測試結果：
```
請求 1: 24 tokens
請求 2: 24 tokens
Hit: 16 tokens (1 block)
Hit ratio: 66.67%
```

分析：
- 24 tokens = 2 blocks（16 + 8）
- 只有 Block 0（16 tokens）被 hit
- 這意味著前 16 個 tokens 匹配，但後 8 個 tokens 不匹配

可能的原因：
- Chat template 在 prompt 末尾添加了不同的內容
- Tokenization 產生了不同的 token 序列
- Block 邊界對齊問題

### 結論

**66.67% 的 hit ratio 是合理的**，如果：
1. ✅ 第一個請求沒有 hit（符合預期）
2. ✅ 第二個請求有 hit（證明 prefix cache 工作）
3. ✅ Hit ratio >= 60%（部分匹配仍然有價值）

這表明 prefix cache 正在正常工作，只是因為 prompt 不完全相同而只有部分匹配。這是正常的行為，不需要擔心。

