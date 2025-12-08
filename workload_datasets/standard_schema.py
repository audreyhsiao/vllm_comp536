from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    """單一對話訊息。"""
    role: str        # "user" / "assistant" / "system"
    content: str     # 純文字內容


@dataclass
class StandardExample:
    """
    M3 用的標準樣本格式。

    - id:            此 conversation / request 的唯一 ID
    - workload_type: 高層類別，例如 "agent" / "coding" / "online_trace_A"
    - conversations: 多輪對話（可為空 list，但型態固定）
    - timestamp:     到達時間（沒有就 None）
    - meta:          各資料集特有資訊（source, subset, task_id, blocks ...）
    """
    id: str
    workload_type: str
    conversations: List[Message]
    timestamp: Optional[float]
    meta: Dict[str, Any]
