from typing import List, Any
import json

import pandas as pd

from .standard_schema import Message, StandardExample


def _extract_messages_from_trajectory(traj: Any) -> List[Message]:
    """
    把 CC-Bench 的 trajectory 轉成 List[Message]。

    你貼的例子長這樣：
    [
      {
        "type": "summary",
        ...
      },
      {
        "type": "user",
        "message": {
          "role": "user",
          "content": "請用swift..."
        },
        ...
      },
      {
        "type": "assistant",
        "message": {
          "role": "assistant",
          "content": [
            { "type": "text", "text": "I'll create..." }
          ]
        },
        ...
      },
      {
        "type": "assistant",
        "message": {
          "role": "assistant",
          "content": [
            { "type": "tool_use", ... }
          ]
        },
        ...
      },
      ...
    ]

    規則：
    - 只保留 role in {user, assistant, system}
    - 如果 content 是字串 → 直接當內容
    - 如果 content 是 list[block] → 只拿 block.type == "text" 的 text 拼起來
    - 沒有任何文字（例如只有 tool_use）就略過
    """
    msgs: List[Message] = []

    # 1) trajectory 可能是 JSON 字串，先處理掉
    if isinstance(traj, str):
        try:
            traj = json.loads(traj)
        except Exception:
            # 爛到不行就當成一條 user 訊息
            return [Message(role="user", content=traj)]

    if not isinstance(traj, list):
        return []

    for node in traj:
        if not isinstance(node, dict):
            continue

        msg = node.get("message")
        if not isinstance(msg, dict):
            continue

        role = msg.get("role")
        if role not in ("user", "assistant", "system"):
            # summary / tool 之類的直接略過
            continue

        raw_content = msg.get("content")
        text = ""

        # case 1: content 是一個純字串
        if isinstance(raw_content, str):
            text = raw_content

        # case 2: content 是 list[block]（像你貼的 assistant 那種）
        elif isinstance(raw_content, list):
            parts: List[str] = []
            for block in raw_content:
                if isinstance(block, dict) and block.get("type") == "text":
                    t = block.get("text")
                    if isinstance(t, str):
                        parts.append(t)
            text = "".join(parts)

        # case 3: 其他型別，就硬轉字串（保底）
        else:
            if raw_content is not None:
                text = str(raw_content)

        # 如果完全沒有文字，通常是純 tool_use，就不要塞進 conversations
        if not text.strip():
            continue

        msgs.append(Message(role=role, content=text))

    return msgs


def load_ccbench_standard() -> List[StandardExample]:
    """
    將 CC-Bench-trajectories 轉成 StandardExample list。
    直接對應 parquet 裡的 `trajectory` schema，不再經過 convert_single_conv。
    """

    parquet_path = "hf://datasets/zai-org/CC-Bench-trajectories/train.parquet"
    df = pd.read_parquet(parquet_path)

    # print("CC-Bench columns:", list(df.columns))

    # 目前這個 dataset 是放在 trajectory 欄位
    if "trajectory" in df.columns:
        conv_col = "trajectory"
    elif "conversations" in df.columns:
        conv_col = "conversations"
    else:
        raise ValueError(
            "CC-Bench parquet 中找不到 conversations/trajectory 欄位，"
            f"實際欄位: {list(df.columns)}"
        )

    out: List[StandardExample] = []

    for _, row in df.iterrows():
        conv_data = row[conv_col]

        # 轉成 List[Message]
        messages = _extract_messages_from_trajectory(conv_data)

        # meta：保留一些有用欄位
        meta = {"source": "CC-Bench"}
        for key in ["task_id", "task_category", "model_name"]:
            if key in df.columns:
                meta[key] = row[key]

        ex = StandardExample(
            id=str(row["id"]),
            workload_type="coding",   # CC-Bench 是 coding 任務
            conversations=messages,
            timestamp=None,           # dataset 沒時間就先 None
            meta=meta,
        )
        out.append(ex)

    return out
