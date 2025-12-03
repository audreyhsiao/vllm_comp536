# Milestone 2 更改说明

本文档详细描述了在 Milestone 2 中，我们基于原始 vLLM 代码库所做的所有更改和新增功能。

## 目录

1. [概述](#概述)
2. [核心功能更改](#核心功能更改)
3. [新增组件](#新增组件)
4. [配置参数扩展](#配置参数扩展)
5. [使用示例](#使用示例)
6. [技术细节](#技术细节)

---

## 概述

Milestone 2 的主要目标是在 vLLM 中实现一个**模拟执行环境**，用于评估 prefix caching（前缀缓存）在实际工作负载中的性能表现。我们实现了以下核心功能：

1. **SimulatorExecutor**：一个模拟执行器，可以模拟模型推理过程而无需实际加载模型或使用 GPU
2. **Client Simulator**：一个客户端模拟器，用于重放 ShareGPT 数据集中的真实对话
3. **Trace-based 响应回放**：基于预录制的 trace 文件回放模型响应
4. **Prefix Cache 支持**：完整支持 vLLM 的 prefix caching 功能，并添加了详细的指标收集

---

## 核心功能更改

### 1. SimulatorExecutor（模拟执行器）

**文件位置**：`vllm/executor/simulator_executor.py`

#### 功能描述

`SimulatorExecutor` 是一个新的执行器实现，继承自 `ExecutorBase`。它允许 vLLM 在没有实际 GPU 或模型的情况下运行，通过读取预录制的 trace 文件来模拟模型的行为。

#### 主要特性

- **Trace 文件支持**：从 JSONL 格式的 trace 文件中读取 prompt 和对应的 response token IDs
- **延迟模拟**：可以模拟 prefill 和 decode 阶段的延迟
- **无 GPU 运行**：完全在 CPU 上运行，不需要实际的模型权重
- **兼容性**：与 vLLM 的调度器和 block manager 完全兼容

#### 核心实现

```python
class SimulatorExecutor(ExecutorBase):
    uses_ray = False  # 非 Ray 模式
    
    def __init__(self, vllm_config: VllmConfig, **kwargs):
        # 从配置中读取模拟器参数
        self.trace_path = getattr(vllm_config, "sim_trace_path", None)
        self.prefill_ms = float(getattr(vllm_config, "sim_prefill_ms_per_tok", 0.0))
        self.decode_ms_base = float(getattr(vllm_config, "sim_decode_ms_base", 0.0))
        self.decode_ms_per_seq = float(getattr(vllm_config, "sim_decode_ms_per_seq", 0.0))
        
        # 加载 trace 文件
        self.trace = TraceStore(self.trace_path) if self.trace_path else None
        self.req_state: Dict[str, TraceCursor] = {}
```

#### Trace 文件格式

Trace 文件应为 JSONL 格式，每行包含一个 JSON 对象：

```json
{"prompt_token_ids": [1, 2, 3, ...], "response_token_ids": [4, 5, 6, ...]}
{"prompt_token_ids": [10, 11, 12, ...], "response_token_ids": [13, 14, 15, ...]}
```

#### 延迟模拟

- **Prefill 延迟**：`sim_prefill_ms_per_tok * token_chunk_size`（毫秒）
- **Decode 延迟**：`sim_decode_ms_base + sim_decode_ms_per_seq`（毫秒）

### 2. 执行器注册

**文件位置**：`vllm/engine/llm_engine.py`

在 `LLMEngine._get_executor_cls()` 方法中添加了对 `sim` 执行器的支持：

```python
if distributed_executor_backend == "sim":
    from vllm.executor.simulator_executor import SimulatorExecutor
    return SimulatorExecutor
```

---

## 新增组件

### 1. Client Simulator（客户端模拟器）

**文件位置**：`sim/client_simulator.py`

#### 功能描述

`ClientSimulator` 是一个完整的客户端模拟器，用于重放 ShareGPT 数据集中的真实对话。它模拟多个客户端同时向 vLLM 服务器发送请求的场景。

#### 主要组件

##### ShareGPTLoader

负责从本地文件或 HuggingFace 数据集加载 ShareGPT 格式的对话数据：

```python
class ShareGPTLoader:
    @staticmethod
    def load_from_file(file_path: str) -> List[ShareGPTConversation]:
        # 支持 JSON 数组和 JSONL 格式
        # 使用流式解析以处理大文件
        
    @staticmethod
    def load_from_huggingface(dataset_name: str, ...) -> List[ShareGPTConversation]:
        # 从 HuggingFace 数据集加载
```

##### TimingSimulator

模拟请求到达的时间：

- **基于时间戳**：如果数据集中包含时间戳，使用真实的时间间隔
- **Poisson 分布**：如果没有时间戳，使用 Poisson 过程模拟请求到达

```python
class TimingSimulator:
    def get_next_arrival_time(self, conversation: ShareGPTConversation) -> float:
        if self.use_timestamps and conversation.timestamp is not None:
            # 使用数据集中的时间戳
            return ts - self.first_timestamp
        else:
            # 使用 Poisson 分布
            inter_arrival = np.random.exponential(1.0 / self.poisson_lambda)
            return self.current_time + inter_arrival
```

##### ChatTemplateFormatter

使用模型的 chat template 格式化对话：

- 自动检测并使用 tokenizer 的 chat template
- 支持自定义 chat template
- 生成 prefix key 用于 prefix sharing

```python
class ChatTemplateFormatter:
    def format_conversation(self, messages: List[Dict[str, str]], 
                           add_generation_prompt: bool = True) -> str:
        # 使用 tokenizer 的 apply_chat_template 方法
        
    def get_prefix_key(self, messages: List[Dict[str, str]], 
                      use_full_conversation: bool = False) -> str:
        # 生成用于 prefix sharing 的 key（SHA1 hash）
```

##### VLLMHttpBackend

通过 HTTP 与 vLLM 服务器通信：

- 支持 OpenAI 兼容的 API
- 自动重试机制
- 错误处理和健康检查
- 支持 `prompt_token_ids` 直接发送（当服务器使用 `--skip-tokenizer-init` 时）

```python
class VLLMHttpBackend:
    async def add_request(
        self,
        prompt: Optional[str] = None,
        prompt_token_ids: Optional[List[int]] = None,
        max_tokens: int = 128,
        prefix_key: Optional[str] = None,
        ...
    ):
        # 发送请求到 vLLM 服务器
        # 支持 prefix_key 用于 prefix sharing
```

##### ClientSimulator

主模拟器类，协调所有组件：

- 并发控制（semaphore）
- 速率限制
- 统计信息收集
- 上下文长度管理

```python
class ClientSimulator:
    async def replay_conversations(
        self,
        conversations: List[ShareGPTConversation],
        prefix_sharing: bool = True,
        use_full_conversation_prefix: bool = False,
        mode: str = "multi"  # "single" 或 "multi"
    ):
        # 重放对话列表
        # mode="single": 每个对话只发送第一个 user turn
        # mode="multi": 发送所有 user turns
```

#### 关键特性

1. **上下文长度管理**：自动截断过长的 prompt，确保不超过模型的最大上下文长度
2. **Prefix Sharing 支持**：生成 prefix key 并发送给服务器
3. **多轮对话支持**：可以重放完整的多轮对话
4. **统计信息**：收集请求成功率、延迟、吞吐量等指标

---

## 配置参数扩展

### 1. SimulatorExecutor 配置参数

**文件位置**：`vllm/engine/arg_utils.py`

在 `EngineArgs` 类中添加了以下参数：

```python
@dataclass
class EngineArgs:
    # Simulator 相关参数
    sim_prefill_ms_per_tok: float = 0.7      # Prefill 阶段每个 token 的延迟（毫秒）
    sim_decode_ms_base: float = 1.0          # Decode 阶段基础延迟（毫秒）
    sim_decode_ms_per_seq: float = 0.5       # Decode 阶段每个序列的额外延迟（毫秒）
    sim_trace_path: Optional[str] = None    # Trace 文件路径
```

### 2. Prefix Stats Collector（前缀缓存统计收集器）

**文件位置**：`vllm/prefix_stats_collector.py`

#### 功能描述

`PrefixStatsCollector` 是一个全局的统计收集器，用于收集和分析 prefix cache 的使用情况。它提供了详细的统计信息，包括：

- **每个请求的统计**：prefix hit tokens、hit ratio、hit block IDs
- **每个 block 的统计**：hit count、reuse interval（重用间隔）

#### 主要功能

```python
class PrefixStatsCollector:
    def record_prompt_length(self, request_id: str, prompt_tokens: int):
        # 记录请求的 prompt token 数量
        
    def record_hit(self, request_id: str, block_ids: Iterable[int], 
                   timestamp: Optional[float] = None):
        # 记录 prefix cache hit 事件
        
    def snapshot(self) -> Dict[str, Any]:
        # 生成统计快照，包含所有请求和 block 的统计信息
        
    def dump(self, path: str):
        # 将统计信息保存到 JSON 文件
```

#### 统计信息格式

`snapshot()` 方法返回的统计信息格式：

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

#### API 端点

在 OpenAI API 服务器中添加了 `/dump_prefix_stats` 端点：

```python
@app.get("/dump_prefix_stats")
def dump_prefix_stats():
    output_path = "prefix_stats.json"
    global_prefix_collector.dump(output_path)
    return {"status": "ok", "path": output_path}
```

可以通过 HTTP GET 请求获取统计信息：

```bash
curl http://127.0.0.1:8000/dump_prefix_stats
```

#### 集成点

`PrefixStatsCollector` 在以下位置被调用：

1. **BlockManager**：在分配和重用 block 时记录 hit 事件
2. **Scheduler**：在调度请求时记录 prompt 长度和 hit 事件
3. **LLMEngine**：在引擎级别集成统计收集

### 3. VllmConfig 扩展

**文件位置**：`vllm/config.py`

在 `VllmConfig` 类中添加了对应的配置字段：

```python
@dataclass
class VllmConfig:
    # Simulator 配置
    sim_trace_path: Optional[str] = None
    sim_prefill_ms_per_tok: float = 0.0
    sim_decode_ms_base: float = 0.0
    sim_decode_ms_per_seq: float = 0.0
```

### 4. 命令行参数

这些参数可以通过命令行传递给 vLLM：

```bash
--sim-trace-path /path/to/trace.jsonl
--sim-prefill-ms-per-tok 0.7
--sim-decode-ms-base 1.0
--sim-decode-ms-per-seq 0.5
--distributed-executor-backend sim
```

---

## 使用示例

### 1. 启动模拟服务器

使用 `run_sim_server.sh` 脚本启动模拟服务器：

```bash
#!/usr/bin/env bash
TRACE=/path/to/trace.jsonl \
MODEL=/path/to/stub_model \
./run_sim_server.sh
```

脚本内容：

```bash
python -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port 8000 \
  --model facebook/opt-125m \
  --served-model-name trace-sim \
  --device cpu \
  --dtype float16 \
  --distributed-executor-backend sim \
  --sim-trace-path "$TRACE" \
  --sim-prefill-ms-per-tok 0 \
  --sim-decode-ms-base 0 \
  --sim-decode-ms-per-seq 0 \
  --skip-tokenizer-init \
  --max-seq-len 128 \
  --enable-prefix-caching \
  --disable-log-requests
```

### 2. 运行客户端模拟器

使用 `run_client.sh` 脚本运行客户端模拟器：

```bash
#!/usr/bin/env bash
json_file=$1

python sim/client_simulator.py \
  --max-concurrent-requests 500 \
  --poisson-lambda 50 \
  --dataset-file $json_file \
  --server-url http://127.0.0.1:8000 \
  --model-name facebook/opt-125m \
  --backend-model-name trace-sim \
  --mode single \
  --no-use-timestamps
```

### 3. 获取 Prefix Cache 统计信息

在运行模拟后，可以通过 API 获取详细的 prefix cache 统计信息：

```bash
# 获取统计信息并保存到文件
curl http://127.0.0.1:8000/dump_prefix_stats

# 统计信息会保存到 prefix_stats.json 文件
# 文件包含：
# - 每个请求的 prefix hit ratio
# - 每个 block 的 hit count 和 reuse interval
```

统计信息文件示例：

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

### 4. 完整工作流程

1. **准备 trace 文件**：将 prompt 和 response token IDs 保存为 JSONL 格式
2. **准备 stub model**：创建一个最小的模型目录（只需要 `config.json`）
3. **启动服务器**：使用 `run_sim_server.sh` 启动模拟服务器
4. **运行客户端**：使用 `run_client.sh` 运行客户端模拟器
5. **获取统计信息**：通过 `/dump_prefix_stats` API 获取 prefix cache 统计信息
6. **分析结果**：查看服务器日志、客户端统计信息和 prefix cache 统计文件

---

## 技术细节

### 1. Trace 文件格式

Trace 文件使用 JSONL 格式，每行一个 JSON 对象：

```json
{"prompt_token_ids": [1, 2, 3], "response_token_ids": [4, 5, 6]}
```

`TraceStore` 类使用 SHA1 哈希 prompt token IDs 作为 key，快速查找对应的 response：

```python
def _sha1_int(ids: List[int]) -> str:
    m = hashlib.sha1()
    m.update((",".join(map(str, ids))).encode("utf-8"))
    return m.hexdigest()
```

### 2. 兼容性处理

`SimulatorExecutor` 需要与不同版本的 vLLM 兼容。`_mk_group_output()` 函数处理了不同版本的 `CompletionSequenceGroupOutput` 构造方式：

```python
def _mk_group_output(samples: List[object]) -> CompletionSequenceGroupOutput:
    # 尝试使用 samples 参数构造
    # 如果失败，使用无参数构造 + setattr
    # 最后保底使用 outputs 属性
```

### 3. Prefix Cache 集成

虽然 prefix caching 是 vLLM 的原有功能，但我们在以下方面进行了集成：

1. **Metrics 收集**：在 `vllm/engine/metrics.py` 中添加了 prefix cache hit rate 的指标
2. **日志输出**：在 `LoggingStatLogger` 中输出 GPU 和 CPU 的 prefix cache hit rate
3. **Prometheus 支持**：添加了 `gauge_cpu_prefix_cache_hit_rate` 和 `gauge_gpu_prefix_cache_hit_rate` 指标

### 4. 上下文长度管理

客户端模拟器实现了智能的上下文长度管理：

```python
max_context_tokens = 2048
tokenizer = self.chat_formatter.tokenizer
if tokenizer is not None:
    enc = tokenizer(prompt, ...)
    input_ids = enc["input_ids"]
    total_tokens = len(input_ids) + new_tokens
    
    if total_tokens > max_context_tokens:
        # 只保留最后 allowed_input 个 token
        allowed_input = max_context_tokens - new_tokens
        input_ids = input_ids[-allowed_input:]
```

这确保了即使 prompt 很长，也能在模型的上下文长度限制内运行。

### 5. 错误处理和重试

`VLLMHttpBackend` 实现了完善的错误处理：

- **自动重试**：对于网络错误和 5xx 状态码，自动重试最多 3 次
- **指数退避**：重试延迟逐渐增加
- **健康检查**：在失败后检查服务器是否仍然可达
- **详细错误信息**：提供清晰的错误消息帮助调试

---

## 文件清单

### 新增文件

1. `vllm/executor/simulator_executor.py` - SimulatorExecutor 实现
2. `sim/client_simulator.py` - 客户端模拟器实现
3. `sim/README_client_simulator.md` - 客户端模拟器文档
4. `run_sim_server.sh` - 启动模拟服务器的脚本
5. `run_client.sh` - 运行客户端模拟器的脚本

### 修改文件

1. `vllm/engine/arg_utils.py` - 添加 simulator 相关参数
2. `vllm/config.py` - 添加 simulator 配置字段
3. `vllm/engine/llm_engine.py` - 注册 SimulatorExecutor
4. `vllm/engine/metrics.py` - 添加 prefix cache hit rate 指标（如果原本没有）
5. `vllm/prefix_stats_collector.py` - 新增 prefix cache 统计收集器
6. `vllm/entrypoints/openai/api_server.py` - 添加 `/dump_prefix_stats` API 端点
7. `vllm/core/block_manager.py` - 集成 prefix stats collector
8. `vllm/core/scheduler.py` - 集成 prefix stats collector

---

## 依赖项

### Python 包

- `transformers` - 用于加载 tokenizer 和 chat template
- `datasets` - 用于从 HuggingFace 加载数据集（可选）
- `aiohttp` - 用于异步 HTTP 请求
- `numpy` - 用于随机数生成和统计计算
- `ijson` - 用于流式解析大型 JSON 文件（可选，但推荐）

### 安装

```bash
pip install transformers datasets aiohttp numpy ijson
```

---

## 性能考虑

1. **内存使用**：`TraceStore` 将所有 trace 数据加载到内存中。对于大型 trace 文件，考虑使用数据库或流式处理
2. **并发控制**：客户端模拟器使用 semaphore 限制并发请求数，避免服务器过载
3. **速率限制**：支持请求速率限制，可以模拟真实的客户端行为
4. **上下文截断**：自动截断过长的 prompt，但可能会影响 prefix sharing 的效果

---

## 限制和已知问题

1. **Trace 文件大小**：大型 trace 文件会占用大量内存
2. **时间戳精度**：如果使用数据集中的时间戳，需要确保时间戳格式正确
3. **Token 对齐**：如果 trace 文件中的 token IDs 与服务器使用的 tokenizer 不匹配，会导致错误
4. **Prefix Key 生成**：当前使用简单的 SHA1 哈希，可能不够精确

---

## 未来改进方向

1. **数据库支持**：使用数据库存储 trace 数据，支持更大的数据集
2. **分布式客户端**：支持多个客户端进程同时运行
3. **更精确的延迟模拟**：根据 prompt 长度和模型大小动态调整延迟
4. **实时指标收集**：实时收集和展示 prefix cache 命中率等指标
5. **可视化工具**：开发 Web UI 用于监控和可视化实验结果

---

## 总结

Milestone 2 实现了一个完整的模拟环境，用于评估 prefix caching 在实际工作负载中的性能。通过 SimulatorExecutor 和 Client Simulator，我们可以在不需要实际 GPU 和模型的情况下，模拟真实的 LLM 服务场景，这对于研究和优化 prefix caching 策略非常有价值。

所有更改都保持了与原始 vLLM 代码的兼容性，可以无缝集成到现有的 vLLM 工作流中。

