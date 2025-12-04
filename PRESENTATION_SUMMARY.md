# 系統簡報摘要

## 系統簡介

這是一個基於 **vLLM** 的 LLM 服務系統，專門用於評估 **Prefix Caching（前綴緩存）** 技術在實際工作負載中的性能表現。

### 核心價值
- ✅ **無需 GPU**：通過模擬執行器，可以在 CPU 上運行實驗
- ✅ **真實工作負載**：使用 ShareGPT 真實對話數據
- ✅ **詳細統計**：收集 prefix cache hit ratio、block reuse 等指標
- ✅ **易於實驗**：支持大規模、可重複的實驗

---

## 系統架構（三大核心組件）

### 1. 模擬執行器 (SimulatorExecutor)
**作用**：模擬模型推理，無需實際 GPU 和模型權重

**關鍵特性**：
- 從 trace 文件讀取預錄製的 prompt-response 對
- 模擬 prefill 和 decode 階段的延遲
- 與 vLLM 調度器完全兼容

### 2. 客戶端模擬器 (Client Simulator)
**作用**：重放真實的用戶對話請求

**關鍵特性**：
- 加載 ShareGPT 數據集
- 模擬多客戶端並發請求
- 自動格式化對話（使用 chat template）
- 支持 prefix sharing

### 3. 前綴緩存系統 (Prefix Caching)
**作用**：重用相同前綴的 KV cache，減少重複計算

**關鍵特性**：
- 基於內容哈希（content hash）的 block 緩存
- 自動檢測和重用相同前綴
- 詳細的統計收集（請求級和 block 級）

---

## 工作流程（5 個步驟）

```
步驟 1: 準備數據
  ├─ Trace 文件（prompt → response 映射）
  └─ ShareGPT 數據集（真實對話）

步驟 2: 啟動服務器
  ├─ 加載 Trace 文件
  ├─ 初始化模擬執行器
  └─ 啟動 API 服務（port 8000）

步驟 3: 客戶端發送請求
  ├─ 加載並格式化對話
  ├─ 生成 prefix key
  └─ 發送 HTTP 請求

步驟 4: 服務器處理
  ├─ 查找 prefix cache
  ├─ 記錄 cache hit/miss
  ├─ 模擬推理過程
  └─ 返回響應

步驟 5: 收集統計
  ├─ 請求級統計（hit ratio）
  ├─ Block 級統計（reuse interval）
  └─ 導出 JSON 結果
```

---

## Prefix Caching 工作原理

### 核心概念
當多個請求有相同的前綴（prefix）時，可以重用已計算的 KV cache，避免重複計算。

### 工作流程
```
新請求到達
    ↓
計算 Prompt 的 Block Hashes
    ↓
查找 Cache
    ├─ Hit? → 重用已緩存的 Blocks（節省計算）
    └─ Miss? → 分配新 Blocks，計算後加入 Cache
    ↓
執行推理（模擬）
    ↓
返回響應
```

### 關鍵指標
- **Prefix Hit Ratio**: 每個請求有多少比例的 tokens 從 cache 命中
- **Block Hit Count**: 每個 block 被重用了多少次
- **Reuse Interval**: Block 被重用的時間間隔

---

## 實驗結果示例

系統會生成以下分析圖表：

1. **Prefix Hit Ratio CDF**
   - 顯示不同請求的 cache hit ratio 分佈
   - 位置：`m2t2/prefix_hit_ratio_cdf.png`

2. **Block Hit Count CDF**
   - 顯示 blocks 被重用的次數分佈
   - 位置：`m2t2/block_hit_count_cdf.png`

3. **Reuse Interval CDF**
   - 顯示 block 重用間隔的分佈
   - 位置：`m2t3/reuse_interval_cdf.png`

4. **Benefit Fraction Bar**
   - 顯示 prefix caching 帶來的效益
   - 位置：`m2t2/benefit_fraction_bar.png`

---

## 技術亮點

### 1. 模擬執行環境
- 不需要實際 GPU 和模型權重
- 通過 trace 文件模擬模型行為
- 支持可配置的延遲模擬

### 2. 真實工作負載
- 使用 ShareGPT 真實對話數據
- 支持多輪對話重放
- 可選的時間戳或 Poisson 分佈請求到達

### 3. 詳細統計收集
- 請求級統計：每個請求的 hit ratio
- Block 級統計：每個 block 的重用情況
- 時間序列分析：reuse interval 分佈

### 4. 易於擴展
- 支持大量並發請求
- 流式處理大型數據集
- 高效的哈希查找

---

## 使用方式

### 快速開始
```bash
# 1. 啟動服務器
./run_sim_server.sh

# 2. 運行客戶端模擬
./run_client.sh sim/clean50.json single

# 3. 獲取統計信息
curl http://127.0.0.1:8000/dump_prefix_stats
```

### 配置參數
- `--block-size`: KV cache block 大小（默認 16）
- `--max-concurrent-requests`: 最大並發請求數
- `--poisson-lambda`: 請求到達速率（每秒）
- `--prefix-sharing`: 是否啟用 prefix sharing

---

## 應用場景

1. **性能評估**
   - 評估 prefix caching 在不同工作負載下的效果
   - 分析不同配置參數的影響

2. **策略研究**
   - 研究 prefix sharing 策略
   - 優化 cache 管理算法

3. **成本分析**
   - 估算 prefix caching 帶來的計算節省
   - 分析 cache hit ratio 與性能的關係

---

## 總結

這個系統提供了一個**完整的模擬環境**，用於：
- ✅ 評估 prefix caching 性能
- ✅ 分析真實工作負載
- ✅ 無需 GPU 資源進行實驗
- ✅ 生成詳細的統計報告

**核心優勢**：可以在沒有實際 GPU 和模型的情況下，進行大規模、可重複的 prefix caching 實驗，大大降低了實驗成本。

