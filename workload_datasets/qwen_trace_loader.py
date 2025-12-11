import json
from typing import List, Dict

from .standard_schema import Message, StandardExample


def load_qwen_as_standard(path: str, workload_type: str) -> List[StandardExample]:
    """
    將 Qwen KV trace 轉成 StandardExample list。
    每個 record 變成一個單輪 user prompt，
    prompt 內容用 hash_ids 映射成假 token，保留 prefix 結構。
    """
    out: List[StandardExample] = []

    hash2token: Dict[int, str] = {}

    def tok(h: int) -> str:
        if h not in hash2token:
            # 也可以用更短的，如 T0/T1/... 看你爽
            hash2token[h] = f"[B{h}]"
        return hash2token[h]

    with open(path, "r") as f:
        for line in f:
            rec = json.loads(line)

            chat_id = rec["chat_id"]
            turn = rec["turn"]
            ts = rec["timestamp"]

            # 用 hash_ids 生成假 prompt
            blocks = rec["hash_ids"]
            block_tokens = [tok(h) for h in blocks]
            fake_prompt = " ".join(block_tokens)

            # 如果想保留 debug 資訊，也可以前面加一段 header
            # header = f"[QwenTrace chat_id={chat_id}, turn={turn}, in={rec['input_length']}, out={rec['output_length']}]\n"
            # content = header + fake_prompt
            content = fake_prompt

            fake_msg = Message(
                role="user",
                content=content,
            )

            ex = StandardExample(
                id=f"{workload_type}_{chat_id}_{turn}",
                workload_type=workload_type,
                conversations=[fake_msg],
                timestamp=ts,
                meta={
                    "source": "QwenTrace",
                    "chat_id": chat_id,
                    "parent_chat_id": rec["parent_chat_id"],
                    "input_length": rec["input_length"],
                    "output_length": rec["output_length"],
                    "req_type": rec["type"],
                    # 真正的 block 序列還是保留在 meta 裡，之後要直接用也行
                    "blocks": blocks,
                },
            )
            out.append(ex)

    return out
