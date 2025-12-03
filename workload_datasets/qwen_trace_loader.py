import json
from typing import List

from .standard_schema import Message, StandardExample


def load_qwen_as_standard(path: str, workload_type: str) -> List[StandardExample]:
    """
    將 Qwen KV trace 轉成 StandardExample list。

    - path: jsonl 路徑，例如
        "../data/qwen-bailian-usagetraces-anon/qwen_traceA_blksz_16.jsonl"
    - workload_type: "online_trace_A" 或 "online_trace_B"
    """
    out: List[StandardExample] = []

    with open(path, "r") as f:
        for line in f:
            rec = json.loads(line)

            chat_id = rec["chat_id"]
            turn = rec["turn"]
            ts = rec["timestamp"]

            # 造一個假 user message，方便 debug / 之後需要 text 介面時不會爆炸。
            fake_msg = Message(
                role="user",
                content=(
                    f"[QwenTrace chat_id={chat_id}, turn={turn}, "
                    f"in={rec['input_length']}, out={rec['output_length']}]"
                ),
            )

            ex = StandardExample(
                id=f"{workload_type}_{chat_id}_{turn}",
                workload_type=workload_type,      # "online_trace_A" / "online_trace_B"
                conversations=[fake_msg],
                timestamp=ts,
                meta={
                    "source": "QwenTrace",
                    "chat_id": chat_id,
                    "parent_chat_id": rec["parent_chat_id"],
                    "input_length": rec["input_length"],
                    "output_length": rec["output_length"],
                    "req_type": rec["type"],
                    # ★ 真正要給 KV cache simulator 用的 block id 序列
                    "blocks": rec["hash_ids"],
                },
            )
            out.append(ex)

    return out
