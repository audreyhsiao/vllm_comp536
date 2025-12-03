from typing import List
import pandas as pd
from .standard_schema import Message, StandardExample

# TODO: 把 'sharegpt_convert' 改成你朋友那個檔案的實際模組名稱
# 例如檔案在 repo root 叫 sharegpt_convert.py，這行就 OK。
from sim.convert_sharegpt import convert_single_conv  # type: ignore


def _load_agentbank_subset(
    parquet_path: str,
    subset_name: str,        # "alfred" / "alfworld" / "apps"
    workload_type: str = "agent",
) -> List[StandardExample]:
    """
    將單一 AgentBank 子資料集（alfred/alfworld/apps）轉成 StandardExample list。
    """
    df = pd.read_parquet(parquet_path)
    out: List[StandardExample] = []

    for _, row in df.iterrows():
        # 組一個 raw dict 給 convert_single_conv 使用
        raw = {
            "id": row["id"],
            "conversations": row["conversations"],
        }
        norm = convert_single_conv(raw)
        # norm["conversations"] 已經是 [{"role","content"}, ...]

        messages = [
            Message(role=m["role"], content=m["content"])
            for m in norm["conversations"]
        ]

        ex = StandardExample(
            id=norm["id"],
            workload_type=workload_type,         # 高層一律視為 "agent"
            conversations=messages,
            timestamp=norm.get("timestamp"),
            meta={
                "source": "AgentBank",
                "agentbank_subset": subset_name,  # 保留 alfred / alfworld / apps 分類
            },
        )
        out.append(ex)

    return out


def load_agentbank_all() -> List[StandardExample]:
    """
    讀取 AgentBank 的三個子資料集並合併。
    你也可以依需求另外寫 load_alfred_only 等函式。
    """
    alfred = _load_agentbank_subset(
        "hf://datasets/Solaris99/AgentBank/alfred/train-00000-of-00001.parquet",
        subset_name="alfred",
    )
    alfworld = _load_agentbank_subset(
        "hf://datasets/Solaris99/AgentBank/alfworld/train-00000-of-00001.parquet",
        subset_name="alfworld",
    )
    apps = _load_agentbank_subset(
        "hf://datasets/Solaris99/AgentBank/apps/train-00000-of-00001.parquet",
        subset_name="apps",
    )
    return alfred + alfworld + apps
