# sim_tests/test_simulator_integration.py

# 1) 先在當前進程 stub 一份 aiohttp（避免本進程載入 vllm 時炸）
import sys, types, os, json
from pathlib import Path

aiohttp_stub = types.ModuleType("aiohttp")
class ClientSession: pass
aiohttp_stub.ClientSession = ClientSession
sys.modules.setdefault("aiohttp", aiohttp_stub)

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.sampling_params import SamplingParams

def _write_sitecustomize(tmp_path: Path) -> str:
    """寫一個 sitecustomize.py 到臨時資料夾，讓子行程也 stub aiohttp。"""
    sdir = tmp_path / "siteshim"
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "sitecustomize.py").write_text(
        "import sys, types\n"
        "m = types.ModuleType('aiohttp')\n"
        "class ClientSession: pass\n"
        "m.ClientSession = ClientSession\n"
        "sys.modules.setdefault('aiohttp', m)\n",
        encoding="utf-8",
    )
    # 把此資料夾插到 PYTHONPATH 最前（影響後續由 vLLM 產生的子行程）
    os.environ["PYTHONPATH"] = str(sdir) + (
        os.pathsep + os.environ["PYTHONPATH"] if "PYTHONPATH" in os.environ else ""
    )
    return str(sdir)

def _make_stub_model_dir(tmp_path: Path) -> str:
    """建立最小可用的本地 HF 風格模型資料夾（只含 config.json）。"""
    mdir = tmp_path / "stub_model"
    mdir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],   # 讓 registry 能識別
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
    # 關鍵：skip_tokenizer_init=True（避開 transformers），backend=sim（走你的模擬器）
    return LLMEngine.from_engine_args(EngineArgs(
        model=model_dir,
        device="cpu",
        distributed_executor_backend="sim",
        max_model_len=128,
        block_size=8,
        max_num_seqs=4,
        skip_tokenizer_init=True,
        # 模擬器參數（對齊你實作）
        sim_trace_path=trace_path,
        sim_prefill_ms_per_tok=0.0,
        sim_decode_ms_base=0.0,
        sim_decode_ms_per_seq=0.0,
    ))

def test_trace_replay_decodes_exact_tokens(tmp_path: Path):
    # 讓子行程也能 stub aiohttp
    _write_sitecustomize(tmp_path)

    # 準備 trace.jsonl
    tfile = tmp_path / "trace.jsonl"
    rec = {"prompt_token_ids":[10,11], "response_token_ids":[101,102,103]}
    tfile.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    # 準備本地 stub 模型
    model_dir = _make_stub_model_dir(tmp_path)

    # 建立引擎
    eng = _make_engine(str(tfile), model_dir)

    # 加入一筆 token prompt 請求
    eng.add_request(
        "r1",
        {"prompt_token_ids":[10,11]},
        SamplingParams(max_tokens=3, temperature=0.0)
    )

    # 迭代直到完成
    outs = []
    while eng.has_unfinished_requests():
        outs.extend(eng.step())

    # 取最後輸出驗證 token id 回放
    ro = outs[-1]
    toks = getattr(ro.outputs[0], "token_ids", None) \
        or getattr(ro.outputs[0], "output_token_ids", None)

   
    assert list(toks) == [101, 102, 103]
