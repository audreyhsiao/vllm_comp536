# vllm/executor/simulator_executor.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import json, time, os

from vllm.executor.executor_base import ExecutorBase
from vllm.sequence import ExecuteModelRequest
from vllm.model_executor.layers.sampler import SamplerOutput, SamplerOutputSample

class TraceStore:
    """把 trace 映射成: prompt_token_ids(tuple) -> completion_token_ids(list[int])"""
    def __init__(self, path: Optional[str]) -> None:
        self.by_prompt: Dict[Tuple[int, ...], List[int]] = {}
        if path and os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    obj = json.loads(line)
                    p = tuple(obj["prompt_token_ids"])
                    c = list(obj["completion_token_ids"])
                    self.by_prompt[p] = c

    def find_completion(self, prompt_tokens: Tuple[int, ...]) -> Optional[List[int]]:
        return self.by_prompt.get(prompt_tokens)

class _ReplayCost:
    def __init__(self, pt=0.7, db=1.0, ds=0.5) -> None:
        self.pt, self.db, self.ds = pt, db, ds
    def prefill_ms(self, tok_sum: int, bsz: int) -> int:
        return int(tok_sum * self.pt)
    def decode_ms(self, bsz: int) -> int:
        return int(self.db + self.ds * bsz)

class SimulatorExecutor(ExecutorBase):
    uses_ray = False

    def __init__(self, vllm_config):
        super().__init__(vllm_config=vllm_config)
        # 讀取成本與 trace
        self.cost = _ReplayCost(
            getattr(vllm_config, "sim_prefill_ms_per_tok", 0.7),
            getattr(vllm_config, "sim_decode_ms_base", 1.0),
            getattr(vllm_config, "sim_decode_ms_per_seq", 0.5),
        )
        self.traces = TraceStore(getattr(vllm_config, "sim_trace_path", None))
        # 每個 request 的 replay 游標
        self._cursor: Dict[str, Dict[str, object]] = {}  # rid -> {"comp": List[int], "pos": int, "eos": Optional[int]}

    # ---- LLMEngine 初始化 KV 所需 ----
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        num_gpu_blocks = 0
        num_cpu_blocks = getattr(self.vllm_config.cache_config,
                                 "num_cpu_blocks_override", None) or 8192
        return num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int):
        return

    # 由 LLMEngine.add_request 呼叫，註冊 prompt 對應的 completion
    def register_request_trace(self, request_id: str, prompt_token_ids: Tuple[int, ...]) -> None:
        comp = self.traces.find_completion(prompt_token_ids)
        if comp is None:
            # 找不到就給空列表；decode 會直接發 EOS
            comp = []
        self._cursor[request_id] = {"comp": comp, "pos": 0, "eos": None}

    def execute_model(self, execute_model_req: ExecuteModelRequest) -> List[Optional[SamplerOutput]]:
        metas = execute_model_req.seq_group_metadata_list or []
        if not metas:
            return []

        # 是否為 prefill 步
        is_prefill_batch = any(m.is_prompt for m in metas)
        if is_prefill_batch:
            tok_sum = sum((m.token_chunk_size or 0) for m in metas)
            time.sleep(self.cost.prefill_ms(tok_sum, len(metas)) / 1000.0)
            # prefill 步不回 token（v0 規則）
            return []

        # decode：每個序列產 1 token（用 trace 回放）
        time.sleep(self.cost.decode_ms(len(metas)) / 1000.0)
        outs: List[SamplerOutput] = []
        for m in metas:
            rid = m.request_id
            state = self._cursor.get(rid, {"comp": [], "pos": 0, "eos": None})
            comp: List[int] = state["comp"]  # type: ignore
            pos: int = state["pos"]          # type: ignore
            # 取下一個 token；若沒有，發 eos（讓引擎把序列收束）
            if pos < len(comp):
                tok = comp[pos]
                state["pos"] = pos + 1
            else:
                # 取 metadata 的 eos_token_id
                tok = m.eos_token_id if hasattr(m, "eos_token_id") and m.eos_token_id is not None else 0
            sample = SamplerOutputSample(output_token=tok, logprobs=None)
            outs.append(SamplerOutput(samples=[sample],
                                      model_forward_time=0.0,
                                      model_execute_time=0.0))
        return outs

    def stop_remote_worker_execution_loop(self): return
    def shutdown(self): return
    def check_health(self): return
    def _run_workers(self, cmd: str): return
