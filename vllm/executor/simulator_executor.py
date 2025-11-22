# vllm/executor/simulator_executor.py
from typing import List, Optional, Tuple
import time

from vllm.executor.executor_base import ExecutorBase
from vllm.sequence import ExecuteModelRequest
from vllm.model_executor.layers.sampler import SamplerOutput, SamplerOutputSample

class _ReplayCost:
    def __init__(self, pt=0.7, db=1.0, ds=0.5):
        self.pt, self.db, self.ds = pt, db, ds
    def prefill_ms(self, tok_sum:int, bsz:int) -> int:
        return int(tok_sum * self.pt)
    def decode_ms(self, bsz:int) -> int:
        return int(self.db + self.ds * bsz)

class SimulatorExecutor(ExecutorBase):
    uses_ray = False

    def __init__(self, vllm_config):
        super().__init__(vllm_config=vllm_config)
        self.cost = _ReplayCost(
            getattr(vllm_config, "sim_prefill_ms_per_tok", 0.7),
            getattr(vllm_config, "sim_decode_ms_base", 1.0),
            getattr(vllm_config, "sim_decode_ms_per_seq", 0.5),
        )

    # 讓 LLMEngine 初始化 KV block 數（交給 scheduler 的 block manager 用）
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        num_gpu_blocks = 0
        num_cpu_blocks = getattr(self.vllm_config.cache_config,
                                 "num_cpu_blocks_override", None) or 8192
        return num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int):
        # 模擬器不需要做真正初始化?
        return

    def execute_model(self, execute_model_req: ExecuteModelRequest) -> List[Optional[SamplerOutput]]:
        metas = execute_model_req.seq_group_metadata_list or []
        if not metas:
            return []

        # 判斷這步是否為 prefill（
        is_prefill_batch = any(m.is_prompt for m in metas)

        if is_prefill_batch:
            tok_sum = sum((m.token_chunk_size or 0) for m in metas)
            time.sleep(self.cost.prefill_ms(tok_sum, len(metas)) / 1000.0)
            # prefill 產生 0 個輸出（由下一步 decode 產出 token）
            return []

        # decode：每個 SG 產 1 token
        time.sleep(self.cost.decode_ms(len(metas)) / 1000.0)
        outs: List[SamplerOutput] = []
        for _ in metas:
            sample = SamplerOutputSample(output_token=42, logprobs=None)  # placeholder token
            outs.append(SamplerOutput(samples=[sample],
                                      model_forward_time=0.0,
                                      model_execute_time=0.0))
        return outs

    
    def stop_remote_worker_execution_loop(self): return
    def shutdown(self): return
    def check_health(self): return
    def _run_workers(self, cmd: str): return
