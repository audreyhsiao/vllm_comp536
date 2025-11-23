# vllm/executor/simulator_executor.py
from __future__ import annotations

import json
import hashlib
import time
import inspect
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

from vllm.config import VllmConfig
from vllm.executor.executor_base import ExecutorBase
from vllm.sequence import ExecuteModelRequest, CompletionSequenceGroupOutput
from vllm.logger import init_logger

logger = init_logger(__name__)

# ---------- trace 讀取與游標 ----------
def _sha1_int(ids: List[int]) -> str:
    m = hashlib.sha1()
    m.update((",".join(map(str, ids))).encode("utf-8"))
    return m.hexdigest()

class TraceStore:
    def __init__(self, path: str,
                 prompt_key: str = "prompt_token_ids",
                 resp_key: str = "response_token_ids") -> None:
        self._resp_by_key: Dict[str, List[int]] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                rec = json.loads(s)
                p_ids = list(rec[prompt_key])
                r_ids = list(rec[resp_key])
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

# ---------- 相容層：建立 group output ----------
def _mk_group_output(samples: List[object]) -> CompletionSequenceGroupOutput:
    """
    建 CompletionSequenceGroupOutput，兼容不同版本欄位名稱。
    單步調度下，上游希望拿到的是「一個群組輸出」，裡面有 .samples。
    """
    sig = inspect.signature(CompletionSequenceGroupOutput)
    params = sig.parameters
    # 優先走帶參數的建構
    if "samples" in params:
        kwargs = {"samples": samples}
        # 其他可選欄位清成 None（不同 vLLM 版本可能存在）
        for k in ("prompt_logprobs", "sampled_token_probs", "sampled_token_ids"):
            if k in params:
                kwargs[k] = None
        try:
            return CompletionSequenceGroupOutput(**kwargs)
        except TypeError:
            pass
    # 退路：無參數建構，之後 setattr
    obj = CompletionSequenceGroupOutput.__new__(CompletionSequenceGroupOutput)
    try:
        CompletionSequenceGroupOutput.__init__(obj)  # 若有無參數 __init__
    except Exception:
        pass
    try:
        setattr(obj, "samples", samples)
    except Exception:
        # 最後保底：至少不讓上游 NPE
        try:
            setattr(obj, "outputs", samples)
        except Exception:
            pass
    return obj

# ---------- 主體：模擬執行器 ----------
class SimulatorExecutor(ExecutorBase):
    uses_ray = False  # 非 Ray

    def __init__(self, vllm_config: VllmConfig, **kwargs):
        super().__init__(vllm_config=vllm_config, **kwargs)
        self.cfg = vllm_config

        # 參數（已在 EngineArgs -> VllmConfig 加入）
        self.trace_path: Optional[str] = getattr(vllm_config, "sim_trace_path", None)
        self.prefill_ms: float = float(getattr(vllm_config, "sim_prefill_ms_per_tok", 0.0) or 0.0)
        self.decode_ms_base: float = float(getattr(vllm_config, "sim_decode_ms_base", 0.0) or 0.0)
        self.decode_ms_per_seq: float = float(getattr(vllm_config, "sim_decode_ms_per_seq", 0.0) or 0.0)

        self.trace: Optional[TraceStore] = TraceStore(self.trace_path) if self.trace_path else None
        self.req_state: Dict[str, TraceCursor] = {}

        # EOS 預設（未初始化 tokenizer 時）
        self.eos_id: int = 2
        mc = getattr(vllm_config, "model_config", None)
        if mc is not None:
            eos = getattr(mc, "eos_token_id", None)
            if isinstance(eos, int):
                self.eos_id = eos

        logger.info("SimulatorExecutor is active. trace=%s prefill_ms/tok=%.3f decode_base=%.3f decode_per_seq=%.3f",
                    self.trace_path, self.prefill_ms, self.decode_ms_base, self.decode_ms_per_seq)


    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int):
        return

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        cc = getattr(self.cfg, "cache_config", None)
        override = getattr(cc, "num_gpu_blocks_override", None) if cc else None
        if isinstance(override, int) and override > 0:
            return override, 0
        return 1_000_000, 0  # 給很大 GPU 區塊；CPU swap 0

    def stop_remote_worker_execution_loop(self): return
    def shutdown(self): return
    def check_health(self): return
    def _run_workers(self, cmd: str): return

    def add_lora(self, *args, **kwargs): return
    def remove_lora(self, *args, **kwargs): return
    def list_loras(self): return []
    def pin_lora(self, *args, **kwargs): return

    def add_prompt_adapter(self, *args, **kwargs): return
    def remove_prompt_adapter(self, *args, **kwargs): return
    def list_prompt_adapters(self): return []
    def pin_prompt_adapter(self, *args, **kwargs): return

    def _init_executor(self): return  
    
    def _ensure_cursor_for(self, meta) -> None:
        rid = getattr(meta, "request_id", None) or ""
        if rid in self.req_state:
            return
        p_ids = self._extract_prompt_ids(meta)
        resp = self.trace.lookup(p_ids) if (self.trace and p_ids) else None
        if resp is None:
            resp = []
        self.req_state[rid] = TraceCursor(resp_ids=list(resp))
        
    def execute_model(self, execute_model_req: ExecuteModelRequest):
        """
        回傳 List[List[CompletionSequenceGroupOutput]]。
        外層：單步；內層：該步每個 seq group 的輸出。
        """
        metas = execute_model_req.seq_group_metadata_list or []
        group_outputs = []

        for meta in metas:
            rid = getattr(meta, "request_id", None) or ""
            first_time = rid not in self.req_state

            # 第一次看到：先建游標，可選擇模擬 prefill 延遲，但仍要回傳 1 顆 token
            if first_time:
                self._ensure_cursor_for(meta)
                chunk = getattr(meta, "token_chunk_size", 0) or 0
                if self.prefill_ms and chunk > 0:
                    time.sleep((self.prefill_ms * float(chunk)) / 1000.0)

            # 每一步（含第一次）一律回放 1 顆 token
            next_tok = self._next_token_for_meta(meta)

            delay_ms = float(self.decode_ms_base) + float(self.decode_ms_per_seq)
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)

            # 準備 sample（logprobs 至少要包含該 token，避免 append_token_id 斷言）
            sample = SimpleNamespace(
                output_token=next_tok,
                logprobs={next_tok: SimpleNamespace(logprob=0.0, rank=0)},
            )

            # 注意：對 SingleStepOutputProcessor 而言，CompletionSequenceGroupOutput.samples
            # 要裝的是「sample 物件」(有 output_token / logprobs)，不是 SamplerOutput
            group_outputs.append(_mk_group_output([sample]))

        # 外層一定是單步包一層
        return [group_outputs]

    # ---- 工具：取 prompt ids 與游標 ----
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
        if hasattr(meta, "seq_data") and isinstance(meta.seq_data, dict) and meta.seq_data:
            try:
                any_seq = next(iter(meta.seq_data.values()))
                if hasattr(any_seq, "prompt_token_ids") and any_seq.prompt_token_ids:
                    return list(any_seq.prompt_token_ids)
                if hasattr(any_seq, "token_ids") and any_seq.token_ids:
                    return list(any_seq.token_ids)
            except Exception:
                pass
        return []
