#!/usr/bin/env python3
import argparse
import json
from typing import Any, Dict, List


def clean_text(s: str) -> str:
    """移除特殊換行符，跟 convert_sharegpt.py 一樣。"""
    if not isinstance(s, str):
        return s
    return (
        s.replace("\u2028", "\n")
         .replace("\u2029", "\n")
         .replace("\r\n", "\n")
         .replace("\r", "\n")
    )


def convert_single_conv(raw: Dict[str, Any]) -> Dict[str, Any]:
    """把一個 AgentBank/ShareGPT style 的對話轉成統一格式。"""
    out: Dict[str, Any] = {}

    # id
    out["id"] = raw.get("id")

    # conversations
    messages: List[Dict[str, Any]] = []
    for m in raw.get("conversations", []):
        src_role = m.get("role")
        src_from = m.get("from")

        # role mapping（跟 convert_sharegpt.py 一樣）
        if src_role:
            role = src_role
        else:
            if src_from == "human":
                role = "user"
            elif src_from == "gpt":
                role = "assistant"
            elif src_from == "system":
                role = "system"
            else:
                role = "user"

        content = m.get("content")
        if content is None:
            content = m.get("value", "")

        content = clean_text(content)

        messages.append({
            "role": role,
            "content": content,
        })

    out["conversations"] = messages

    # 把時間欄位原封不動保留（如果有的話）
    for key in ["t", "timestamp"]:
        if key in raw:
            out[key] = raw[key]

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Normalize AgentBank JSONL into ShareGPT-like JSON list."
    )
    parser.add_argument("input", help="AgentBank jsonl 檔案路徑")
    parser.add_argument("output", help="輸出 JSON 檔案路徑（list 格式）")
    args = parser.parse_args()

    converted: List[Dict[str, Any]] = []

    with open(args.input, "r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw_obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] line {line_no} 解析失敗: {e}")
                continue

            conv = convert_single_conv(raw_obj)
            converted.append(conv)

    with open(args.output, "w", encoding="utf-8") as fout:
        json.dump(converted, fout, ensure_ascii=False, indent=2)

    print(f"[OK] 讀入 {len(converted)} 筆對話，已輸出到 {args.output}")


if __name__ == "__main__":
    main()
