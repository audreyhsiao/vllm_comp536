import json
from typing import List, Dict, Any

from .standard_schema import StandardExample


def export_for_m2(
    examples: List[StandardExample],
    data_path: str,
    index_path: str,
) -> None:
    """
    把 StandardExample 轉成：
      1) M2 要吃的 JSON: [ { "id", "conversations": [...] }, ... ]
      2) 之後分析用的索引檔 JSONL: 每行一個 { id, workload_type, ... }

    :param examples:   要輸出的樣本列表
    :param data_path:  給 M2 用的主 JSON 檔路徑
    :param index_path: 給你自己分析用的 index JSONL 檔路徑
    """
    # 先確保是 list（避免傳來的是 generator）
    examples = list(examples)

    # -------- 1) 給 M2 用的主資料 --------
    data_payload = []
    for ex in examples:
        data_payload.append({
            "id": ex.id,
            "conversations": [
                {"role": m.role, "content": m.content}
                for m in ex.conversations
            ],
        })

    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data_payload, f, ensure_ascii=False, indent=2)

    # -------- 2) 之後 join 用的索引檔 --------
    # 一行一個 JSON： {"id", "workload_type", "source", ...}
    with open(index_path, "w", encoding="utf-8") as f:
        for ex in examples:
            rec: Dict[str, Any] = {
                "id": ex.id,
                "workload_type": ex.workload_type,   # agent / coding / online_trace_A
            }

            # 想保留的 meta 欄位可以在這裡加
            if ex.timestamp is not None:
                rec["timestamp"] = ex.timestamp

            # 來源資料集
            if "source" in ex.meta:
                rec["source"] = ex.meta["source"]

            # AgentBank 子類型
            if "agentbank_subset" in ex.meta:
                rec["agentbank_subset"] = ex.meta["agentbank_subset"]

            # CC-Bench 任務資訊
            if "task_id" in ex.meta:
                rec["task_id"] = ex.meta["task_id"]
            if "task_category" in ex.meta:
                rec["task_category"] = ex.meta["task_category"]
            if "model_name" in ex.meta:
                rec["model_name"] = ex.meta["model_name"]

            # QwenTrace 特有欄位也可以視需要加（可選）
            if "chat_id" in ex.meta:
                rec["chat_id"] = ex.meta["chat_id"]
            if "parent_chat_id" in ex.meta:
                rec["parent_chat_id"] = ex.meta["parent_chat_id"]

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
