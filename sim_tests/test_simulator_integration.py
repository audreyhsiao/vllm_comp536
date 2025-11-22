# tests/test_simulator_integration.py
import json
from pathlib import Path

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.sampling_params import SamplingParams


def _make_engine(trace_path: str) -> LLMEngine:
    return LLMEngine.from_engine_args(EngineArgs(
        model="dummy/placeholder",
        device="cpu",
        distributed_executor_backend="sim",
        max_model_len=128,
        block_size=8,
        max_num_seqs=4,
        # 關鍵：跳過 tokenizer 初始化，避免 transformers 依賴
        skip_tokenizer_init=True,
        # 模擬器參數
        sim_trace_path=trace_path,
        sim_prefill_ms_per_tok=0.0,
        sim_decode_ms_base=0.0,
        sim_decode_ms_per_seq=0.0,
    ))


def test_trace_replay_decodes_exact_tokens(tmp_path: Path):
    tfile = tmp_path / "trace.jsonl"
    rec = {"prompt_token_ids":[10,11], "response_token_ids":[101,102,103]}
    tfile.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    eng = _make_engine(str(tfile))
    prompt = {"prompt_token_ids":[10,11]}
    eng.add_request("r1", prompt, SamplingParams(max_tokens=3, temperature=0.0))

    outs = []
    while eng.has_unfinished_requests():
        outs.extend(eng.step())

    ro = outs[-1]
    # 不同版本欄位名可能不同，兩個都試
    toks = getattr(ro.outputs[0], "token_ids", None) \
        or getattr(ro.outputs[0], "output_token_ids", None)
    assert toks == [101, 102, 103]