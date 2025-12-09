from typing import List, Any
import json

import pandas as pd

from .standard_schema import Message, StandardExample


def _sanitize_text(s: Any) -> str:
    """
    把文字裡的特殊換行字元（LS/PS）清掉，避免之後寫 JSON 出問題。
    """
    if not isinstance(s, str):
        s = str(s) if s is not None else ""
    # \u2028: LINE SEPARATOR, \u2029: PARAGRAPH SEPARATOR
    return s.replace("\u2028", "\n").replace("\u2029", "\n")


def _normalize_conv_obj(raw_conv: Any) -> Any:
    """
    把 parquet 裡的 conversations 一律先轉成 Python 原生物件：
    - bytes / bytearray → utf-8 解碼
    - str:
        - 如果看起來像 JSON list（[ 開頭 ] 結尾）→ json.loads
        - 否則就當一條單獨訊息
    - 其他型別直接回傳，後面再處理
    """
    conv = raw_conv

    # bytes → str
    if isinstance(conv, (bytes, bytearray)):
        try:
            conv = conv.decode("utf-8")
        except Exception:
            conv = str(conv)

    # str → 盡量當 JSON list parse
    if isinstance(conv, str):
        s = conv.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                conv = json.loads(s)
            except Exception:
                # 爛掉就留原字串，後面會當成一條訊息
                conv = s

    return conv


def _extract_agentbank_messages(raw_conv: Any) -> List[Message]:
    """
    最終轉成 List[Message] 的函式。

    支援幾種常見情況：
    1) raw_conv 是 list[dict]，每個 dict 長這樣：
         {"from": "human"/"gpt", "value": "..."}
    2) raw_conv 是 JSON 字串，包一個 list → 會先被 _normalize_conv_obj 解成 list
    3) raw_conv 是一條純字串 → 當成一個 role="unknown" 的訊息
    （同時在這裡做文字 sanitize）
    """
    conv = _normalize_conv_obj(raw_conv)

    # case A: 直接是一條字串 → 當成一個訊息
    if isinstance(conv, str):
        text = _sanitize_text(conv)
        if not text.strip():
            return []
        return [Message(role="unknown", content=text)]

    # case B: 是 dict，而且裡面真的就是一個訊息
    if isinstance(conv, dict):
        role = conv.get("from", "unknown")
        text = conv.get("value", "") or conv.get("content", "")
        text = _sanitize_text(text)
        if not text.strip():
            return []
        if not isinstance(role, str):
            role = str(role)
        return [Message(role=role, content=text)]

    # case C: list / 其他可疊代 → 嘗試逐個 element 當訊息
    msgs: List[Message] = []

    try:
        iterable = list(conv)
    except TypeError:
        # 完全不是 iterable → 放棄
        return msgs

    for elem in iterable:
        if isinstance(elem, dict):
            role = elem.get("from", "unknown")
            text = elem.get("value", "") or elem.get("content", "")
            text = _sanitize_text(text)
            if not text.strip():
                continue
            if not isinstance(role, str):
                role = str(role)
            msgs.append(Message(role=role, content=text))
        else:
            s = _sanitize_text(elem)
            if s.strip():
                msgs.append(Message(role="unknown", content=s))

    return msgs


def _load_agentbank_subset(
    parquet_path: str,
    subset_name: str,        # "alfred" / "alfworld" / "apps"
    workload_type: str = "agent",
) -> List[StandardExample]:
    """
    將單一 AgentBank 子資料集（alfred/alfworld/apps）轉成 StandardExample list。
    """
    df = pd.read_parquet(parquet_path)
    # print(f"[AgentBank] Loaded {parquet_path}, num rows = {len(df)}")
    # print("[AgentBank] columns:", list(df.columns))

    out: List[StandardExample] = []

    # 嘗試找對話欄位
    if "conversations" in df.columns:
        conv_col = "conversations"
    elif "trajectory" in df.columns:
        conv_col = "trajectory"
    else:
        raise ValueError(
            f"AgentBank parquet {parquet_path} 中找不到 conversations/trajectory 欄位，"
            f"實際欄位: {list(df.columns)}"
        )

    # inspect 一下第一列
    if len(df) > 0:
        sample = df.iloc[0][conv_col]
        # print(f"[AgentBank] sample {conv_col} type = {type(sample)}")
        # sample_norm = _normalize_conv_obj(sample)
        # print(f"[AgentBank] sample normalized = {repr(str(sample_norm)[:200])}")

    for _, row in df.iterrows():
        conv_raw = row[conv_col]
        messages = _extract_agentbank_messages(conv_raw)

        ex = StandardExample(
            id=str(row["id"]),
            workload_type=workload_type,
            conversations=messages,
            timestamp=None,   # dataset 沒 timestamp 就先 None
            meta={
                "source": "AgentBank",
                "agentbank_subset": subset_name,
            },
        )
        out.append(ex)

    return out


def load_agentbank_all() -> List[StandardExample]:
    """
    讀取 AgentBank 的三個子資料集並合併。
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
