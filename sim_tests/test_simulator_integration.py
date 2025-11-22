# --- stub aiohttp to avoid Py3.9 typing crash during import ---
import sys, types
aiohttp_stub = types.ModuleType("aiohttp")
class ClientSession:  # minimal shim used by vllm.connections
    pass
aiohttp_stub.ClientSession = ClientSession
sys.modules.setdefault("aiohttp", aiohttp_stub)
# --------------------------------------------------------------

import json
from pathlib import Path

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.sampling_params import SamplingParams

def _make_stub_model_dir(tmp_path: Path) -> str:
    """建立一個最小 HuggingFace 風格的本地模型資料夾，只含 config.json，
    並提供 vLLM 能識別的 architectures（用 LlamaForCausalLM）。"""
    mdir = tmp_path / "stub_model"
    mdir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],  # 關鍵：讓 vLLM 找得到對應模型類別
        # 下面參數給最小可用值；我們不會真的載權重，但有些地方會讀到它們
        "vocab_size": 32000,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "num_hidden_layers": 2,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 128,
        "rope_theta": 10000.0
    }
    (mdir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return str(mdir)

def _make_engine(trace_path: str, model_dir: str) -> LLMEngine:
    # 關鍵：model 指向我們的本地 stub 模型資料夾；skip_tokenizer_init=True
    return LLMEngine.from_engine_args(EngineArgs(
        model=model_dir,
        device="cpu",
        distributed_executor_backend="sim",
        max_model_len=128,
        block_size=8,
        max_num_seqs=4,
        skip_tokenizer_init=True,         # 避免 transformers
        # 模擬器參數
        sim_trace_path=trace_path,
        sim_prefill_ms_per_tok=0.0,
        sim_decode_ms_base=0.0,
        sim_decode_ms_per_seq=0.0,
    ))

def test_trace_replay_decodes_exact_tokens(tmp_path: Path):
    """
    端到端驗證：
    - 送入 token prompt
    - decode 過程按 trace 回放：101,102,103
    """
    # 1) 準備 trace.jsonl
    tfile = tmp_path / "trace.jsonl"
    rec = {
        "prompt_token_ids": [10, 11],
        "response_token_ids": [101, 102, 103],
    }
    tfile.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    # 2) 準備本地 stub 模型（只含 config.json）
    model_dir = _make_stub_model_dir(tmp_path)

    # 3) 建 engine
    eng = _make_engine(str(tfile), model_dir)

    # 4) 加一筆 token prompt 請求
    eng.add_request(
        "r1",
        {"prompt_token_ids":[10,11]},
        SamplingParams(max_tokens=3, temperature=0.0)
    )

    # 5) 迭代到完成
    outs=[]
    while eng.has_unfinished_requests():
        outs.extend(eng.step())

    # 6) 驗證回放的 token
    ro = outs[-1]
    toks = getattr(ro.outputs[0], "token_ids", None) \
        or getattr(ro.outputs[0], "output_token_ids", None)
    assert toks == [101,102,103]
