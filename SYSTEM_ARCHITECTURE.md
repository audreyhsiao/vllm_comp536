# 系統架構與工作流程說明

## 系統概述

這是一個基於 vLLM 的 LLM 服務系統，主要用於評估 **Prefix Caching（前綴緩存）** 在實際工作負載中的性能表現。系統包含模擬執行環境，可以在不需要實際 GPU 和模型權重的情況下進行實驗。

---

## 主要組件

### 1. **SimulatorExecutor（模擬執行器）**
- **位置**: `vllm/executor/simulator_executor.py`
- **功能**: 
  - 模擬模型推理過程，無需實際 GPU 或模型
  - 從 trace 文件讀取預錄製的 prompt 和 response
  - 模擬 prefill 和 decode 階段的延遲
  - 與 vLLM 的調度器和 block manager 完全兼容

### 2. **Client Simulator（客戶端模擬器）**
- **位置**: `sim/client_simulator.py`
- **功能**:
  - 重放 ShareGPT 數據集中的真實對話
  - 模擬多個客戶端同時發送請求
  - 支持時間戳或 Poisson 分佈的請求到達時間
  - 自動格式化對話（使用 chat template）
  - 生成 prefix key 用於 prefix sharing

### 3. **Prefix Stats Collector（前綴緩存統計收集器）**
- **位置**: `vllm/prefix_stats_collector.py`
- **功能**:
  - 收集每個請求的 prefix hit ratio
  - 追蹤每個 block 的 hit count 和 reuse interval
  - 提供統計快照和 JSON 導出

### 4. **Prefix Caching Block Allocator（前綴緩存塊分配器）**
- **位置**: `vllm/core/block/prefix_caching_block.py`
- **功能**:
  - 基於內容哈希（content hash）緩存 blocks
  - 重用相同內容的 blocks 避免重複計算
  - 支持 copy-on-write 操作

### 5. **Block Manager（塊管理器）**
- **位置**: `vllm/core/block_manager.py`
- **功能**:
  - 管理 KV cache 的 block 分配
  - 集成 prefix caching 邏輯
  - 記錄 prefix cache hit 事件

---

## 系統架構圖

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Simulator                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ShareGPTLoader│  │TimingSimulator│  │ChatFormatter │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                  │                  │
│         └─────────────────┴──────────────────┘                  │
│                            │                                     │
│                            ▼                                     │
│                   ┌─────────────────┐                            │
│                   │ VLLMHttpBackend │                            │
│                   └────────┬────────┘                            │
└────────────────────────────┼─────────────────────────────────────┘
                             │ HTTP Request
                             │ (OpenAI-compatible API)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      vLLM API Server                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              OpenAI API Server                            │   │
│  │  - /v1/completions                                       │   │
│  │  - /dump_prefix_stats                                    │   │
│  └────────────────┬─────────────────────────────────────────┘   │
│                   │                                               │
│                   ▼                                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    LLM Engine                            │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │   │
│  │  │  Scheduler   │  │Block Manager │  │   Executor   │   │   │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │   │
│  │         │                 │                  │           │   │
│  │         └─────────────────┴──────────────────┘           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                   │                                               │
│                   ▼                                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              SimulatorExecutor                           │   │
│  │  ┌──────────────┐  ┌──────────────┐                      │   │
│  │  │  TraceStore  │  │ Delay Sim    │                      │   │
│  │  └──────┬───────┘  └──────┬───────┘                      │   │
│  │         │                 │                               │   │
│  │         └─────────────────┘                               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                   │                                               │
│                   ▼                                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         PrefixCachingBlockAllocator                      │   │
│  │  ┌──────────────┐  ┌──────────────┐                      │   │
│  │  │ Cached Blocks│  │Content Hash  │                      │   │
│  │  │   (Dict)     │  │   Mapping    │                      │   │
│  │  └──────────────┘  └──────────────┘                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                   │                                               │
│                   ▼                                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         PrefixStatsCollector                            │   │
│  │  - Request-level stats (hit ratio, hit tokens)          │   │
│  │  - Block-level stats (hit count, reuse interval)        │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 工作流程圖

### 完整工作流程

```
1. 準備階段
   │
   ├─► 準備 Trace 文件 (JSONL 格式)
   │   └─► {"prompt_token_ids": [...], "response_token_ids": [...]}
   │
   ├─► 準備 ShareGPT 數據集
   │   └─► 包含真實對話數據
   │
   └─► 啟動模擬服務器
       └─► run_sim_server.sh

2. 服務器啟動
   │
   ├─► 加載 Trace 文件到 TraceStore
   ├─► 初始化 SimulatorExecutor
   ├─► 初始化 PrefixCachingBlockAllocator
   ├─► 初始化 PrefixStatsCollector
   └─► 啟動 OpenAI API Server (port 8000)

3. 客戶端模擬
   │
   ├─► 加載 ShareGPT 數據集
   ├─► 格式化對話（使用 chat template）
   ├─► 生成 prefix key（SHA1 hash）
   └─► 發送 HTTP 請求到服務器
       │
       ├─► 請求到達時間（時間戳或 Poisson）
       └─► 並發控制（semaphore）

4. 請求處理（服務器端）
   │
   ├─► API Server 接收請求
   ├─► LLM Engine 調度請求
   │   │
   │   ├─► Scheduler 選擇要執行的請求
   │   └─► Block Manager 分配 KV cache blocks
   │       │
   │       ├─► 計算 prompt 的 block hashes
   │       ├─► 查找已緩存的 blocks（prefix cache lookup）
   │       ├─► 記錄 cache hit 事件到 PrefixStatsCollector
   │       └─► 分配新的 blocks（如果沒有 cache hit）
   │
   ├─► SimulatorExecutor 執行推理
   │   │
   │   ├─► 從 TraceStore 查找對應的 response
   │   ├─► 模擬 prefill 延遲
   │   └─► 模擬 decode 延遲（逐 token）
   │
   └─► 返回響應給客戶端

5. 統計收集
   │
   ├─► 每個請求記錄：
   │   ├─► total_prompt_tokens
   │   ├─► prefix_hit_tokens
   │   ├─► prefix_hit_ratio
   │   └─► hit_block_ids
   │
   └─► 每個 block 記錄：
       ├─► hit_count
       ├─► reuse_interval_avg
       ├─► reuse_interval_min
       └─► reuse_interval_max

6. 結果分析
   │
   ├─► 通過 /dump_prefix_stats API 獲取統計
   ├─► 生成 JSON 統計文件
   └─► 可視化分析（CDF 圖、柱狀圖等）
```

### Prefix Caching 工作流程詳解

```
新請求到達
    │
    ▼
計算 Prompt 的 Block Hashes
    │
    ▼
查找 Prefix Cache
    │
    ├─► Cache Hit?
    │   │
    │   ├─► 是 ──► 重用已緩存的 Blocks
    │   │         │
    │   │         ├─► 記錄 hit 事件
    │   │         ├─► 更新 block 的 last_accessed_time
    │   │         └─► 計算 reuse interval
    │   │
    │   └─► 否 ──► 分配新的 Blocks
    │             │
    │             ├─► 計算 KV cache
    │             ├─► 填充 block 內容
    │             └─► 將 block 加入 cache（promote_to_immutable）
    │
    ▼
執行 Prefill（模擬）
    │
    ▼
執行 Decode（逐 token，模擬）
    │
    ▼
返回響應
```

---

## 數據流

### 1. Trace 文件格式
```json
{"prompt_token_ids": [1, 2, 3, ...], "response_token_ids": [4, 5, 6, ...]}
{"prompt_token_ids": [10, 11, 12, ...], "response_token_ids": [13, 14, 15, ...]}
```

### 2. ShareGPT 數據格式
```json
{
  "id": "conversation_id",
  "conversations": [
    {"from": "human", "value": "User message"},
    {"from": "gpt", "value": "Assistant response"}
  ],
  "t": 1234567890.123  // 可選時間戳
}
```

### 3. Prefix Stats 輸出格式
```json
{
  "block_size": 16,
  "requests": {
    "request_1": {
      "total_prompt_tokens": 100,
      "prefix_hit_tokens": 32,
      "prefix_hit_ratio": 0.32,
      "hit_block_ids": [1, 2]
    }
  },
  "blocks": {
    "1": {
      "hit_count": 5,
      "reuse_interval_count": 4,
      "reuse_interval_avg": 2.5,
      "reuse_interval_min": 1.0,
      "reuse_interval_max": 5.0
    }
  }
}
```

---

## 關鍵技術特點

### 1. **無 GPU 運行**
- SimulatorExecutor 完全在 CPU 上運行
- 不需要實際的模型權重
- 通過 trace 文件模擬模型行為

### 2. **真實工作負載重放**
- 使用 ShareGPT 真實對話數據
- 支持多輪對話
- 可選的時間戳或 Poisson 分佈請求到達

### 3. **Prefix Caching 評估**
- 詳細的統計收集（請求級和 block 級）
- 支持 prefix sharing（通過 prefix_key）
- 追蹤 cache hit ratio 和 reuse interval

### 4. **可擴展性**
- 支持大量並發請求
- 流式處理大型數據集
- 高效的 block 查找（基於哈希）

---

## 使用示例

### 啟動服務器
```bash
./run_sim_server.sh
```

### 運行客戶端模擬
```bash
./run_client.sh sim/clean50.json single
```

### 獲取統計信息
```bash
curl http://127.0.0.1:8000/dump_prefix_stats
```

---

## 輸出結果

系統會生成以下類型的結果：

1. **Prefix Hit Ratio CDF**: 顯示不同請求的 prefix hit ratio 分佈
2. **Block Hit Count CDF**: 顯示 blocks 被重用的次數分佈
3. **Reuse Interval CDF**: 顯示 block 重用間隔的分佈
4. **Benefit Fraction Bar**: 顯示 prefix caching 帶來的效益

這些圖表保存在 `m2t2/` 和 `m2t3/` 目錄中。

---

## 總結

這個系統提供了一個完整的模擬環境，用於：
- 評估 prefix caching 在實際工作負載中的性能
- 分析不同配置參數的影響（block size、cache size 等）
- 研究 prefix sharing 策略的效果
- 無需實際 GPU 資源即可進行大規模實驗

