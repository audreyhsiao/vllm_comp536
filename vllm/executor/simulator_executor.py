# vllm/executor/simulator_executor.py
from __future__ import annotations

import time
import json
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from vllm.executor.executor_base import ExecutorBase
from vllm.sequence import ExecuteModelRequest
from vllm.model_executor.layers.sampler import (
    SamplerOutput,
    SamplerOutputSample,
)


def _sha1_int(ids: List[int]) -> str:
    m = hashlib.sha1()
    m.update((",".join(map(str, ids))).encode())
    return m.hexdigest()

class TraceStore:
    def __init__(self, path: str,
                 prompt_key: str = "prompt_token_ids",
                 resp_key: str = "response_token_ids") -> None:
        self._resp_by_key: Dict[str, List[int]] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                p_ids = rec[prompt_key]
                r_ids = rec[resp_key]
                self._resp_by_key[_sha1_int(p_ids)] = r_ids

    def lookup(self, prompt_ids: List[int]) -> Optional[List[int]]:
        return self._resp_by_key.get(_sha1_int(prompt_ids))


@dataclass
class TraceCursor:
    resp_ids: List[int]
    pos: int = 0

    def next_token(self) -> Optional[int]:
        if self.pos >= len(self.resp_ids):
            return None
        t = self.resp_ids[self.pos]
        self.pos += 1
        return t


# class _ReplayCost:
#     def __init__(self, pt: float = 0.0, db: float = 0.0, ds: float = 0.0):
#         # 預設全 0 → 測試時更快
#         self.pt, self.db, self.ds = pt, db, ds

#     def prefill_ms(self, tok_sum: int, bsz: int) -> int:
#         return int(tok_sum * self.pt)

#     def decode_ms(self, bsz: int) -> int:
#         return int(self.db + self.ds * bsz)

class SimulatorExecutor(ExecutorBase):
    """
    以 trace 逐步回放：
      - prefill：僅睡/略過，不產 token
      - decode：每個序列組回傳 1 顆 trace 中的下一個 token
    """
    uses_ray = False  # not using ray

    def __init__(self, vllm_config):
        super().__init__(vllm_config=vllm_config)
        self.cfg = vllm_config

        # 讀 CLI 注入的參數（changed in EngineArgs.create_engine_config 
        pt = getattr(vllm_config, "sim_prefill_ms_per_tok", 0.0)
        db = getattr(vllm_config, "sim_decode_ms_base", 0.0)
        ds = getattr(vllm_config, "sim_decode_ms_per_seq", 0.0)
        self.cost = _ReplayCost(pt, db, ds)

        trace_path = getattr(vllm_config, "sim_trace_path", None)
        self.trace: Optional[TraceStore] = (
            TraceStore(trace_path) if trace_path else None
        )

        self.req_state: Dict[str, TraceCursor] = {}

        self.eos_id: int = getattr(
            getattr(vllm_config, "model_config", object()), "eos_token_id", 2
        ) or 2


    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int):
        return

    # prefill/decode
    def execute_model(
        self, execute_model_req: ExecuteModelRequest
    ) -> List[Optional[SamplerOutput]]:
        metas = execute_model_req.seq_group_metadata_list or []
        if not metas:
            return []

        # 判斷這一批是否包含 prefill（v0：prefill 與 decode 會分開批）
        is_prefill_batch = any(getattr(m, "is_prompt", False) for m in metas)

        if is_prefill_batch:
            # 模擬 prefill 成本
            tok_sum = sum(int(getattr(m, "token_chunk_size", 0) or 0) for m in metas)
            ms = self.cost.prefill_ms(tok_sum, len(metas))
            if ms > 0:
                time.sleep(ms / 1000.0)
            # prefill: no token output
            return []

        # decode
        ms = self.cost.decode_ms(len(metas))
        if ms > 0:
            time.sleep(ms / 1000.0)

        outs: List[SamplerOutput] = []
        for meta in metas:
            tok = self._next_token_for_meta(meta)
            sample = SamplerOutputSample(output_token=tok, logprobs=None)
            outs.append(
                SamplerOutput(
                    samples=[sample],
                    model_forward_time=0.0,
                    model_execute_time=0.0,
                )
            )
        return outs

    # ---- 其他必要介面（no-op） ----
    def stop_remote_worker_execution_loop(self):  # noqa: D401
        return

    def shutdown(self):
        return

    def check_health(self):
        return

    def _run_workers(self, cmd: str):
        return
    
    def _next_token_for_meta(self, meta) -> int:
        rid = getattr(meta, "request_id", None) or ""
        cur = self.req_state.get(rid)
        if cur is None:
            p_ids = self._extract_prompt_ids(meta)
            resp = self.trace.lookup(p_ids) if (self.trace and p_ids) else None
            if not resp:
                return self.eos_id
            cur = TraceCursor(resp_ids=list(resp))
            self.req_state[rid] = cur

        nxt = cur.next_token()
        return nxt if nxt is not None else self.eos_id

    def _extract_prompt_ids(self, meta) -> List[int]:
        if hasattr(meta, "prompt_token_ids") and meta.prompt_token_ids:
            return list(meta.prompt_token_ids)
        if hasattr(meta, "token_ids") and meta.token_ids:
            return list(meta.token_ids)
        return []
