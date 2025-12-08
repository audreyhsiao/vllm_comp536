from typing import List
import json

import pandas as pd

from .standard_schema import Message, StandardExample
from sim.convert_sharegpt import convert_single_conv  # 你現在轉檔函式的位置


def load_ccbench_standard() -> List[StandardExample]:
    """
    將 CC-Bench-trajectories 轉成 StandardExample list。
    用 pandas 讀 parquet，並處理「trajectory 是字串」的情況。
    """

    # 這個路徑你前面 AgentBank 已經成功用 hf:// 讀過，照樣來
    parquet_path = "hf://datasets/zai-org/CC-Bench-trajectories/train.parquet"
    df = pd.read_parquet(parquet_path)

    # 看看有哪些欄位（第一次跑可以開這行，之後註解掉）
    # print("CC-Bench columns:", list(df.columns))

    # 判斷哪個欄位裝的是軌跡
    if "conversations" in df.columns:
        conv_col = "conversations"
    elif "trajectory" in df.columns:
        conv_col = "trajectory"
    else:
        raise ValueError(
            "CC-Bench parquet 中找不到 conversations/trajectory 欄位，"
            f"實際欄位: {list(df.columns)}"
        )

    out: List[StandardExample] = []

    for _, row in df.iterrows():
        conv_data = row[conv_col]

        # 1️⃣ 如果是字串，可能是 JSON string，把它 parse 成 list
        if isinstance(conv_data, str):
            try:
                conv_list = json.loads(conv_data)
            except Exception:
                # 萬一不是合法 JSON，就當成只有一條 user 訊息
                conv_list = [{"role": "user", "content": conv_data}]
        else:
            # 已經是 list（理想情況）
            conv_list = conv_data

        raw = {
            "id": row["id"],
            "conversations": conv_list,
        }

        norm = convert_single_conv(raw)
        # 這裡的 norm["conversations"] 一定是 [{'role','content'}, ...]

        messages = [
            Message(role=m["role"], content=m["content"])
            for m in norm["conversations"]
        ]

        # 2️⃣ pandas 的 row 是 Series，不要用 row.get，改用 "欄位在不在 df.columns"
        meta = {"source": "CC-Bench"}
        for key in ["task_id", "task_category", "model_name"]:
            if key in df.columns:
                meta[key] = row[key]

        ex = StandardExample(
            id=norm["id"],
            workload_type="coding",
            conversations=messages,
            timestamp=norm.get("timestamp"),
            meta=meta,
        )
        out.append(ex)

    return out