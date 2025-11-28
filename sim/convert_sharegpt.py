#!/usr/bin/env python3
import argparse
import json
from typing import Any, Dict, List


# 移除特殊換行符：Line Separator (U+2028), Paragraph Separator (U+2029)
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return s
    return (
        s.replace("\u2028", "\n")
         .replace("\u2029", "\n")
         .replace("\r\n", "\n")  # 正規化
         .replace("\r", "\n")
    )


def convert_single_conv(raw: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # id
    out["id"] = raw.get("id")

    # conversations
    messages: List[Dict[str, Any]] = []
    for m in raw.get("conversations", []):
        # role mapping
        src_role = m.get("role")
        src_from = m.get("from")

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

        # content
        content = m.get("content")
        if content is None:
            content = m.get("value", "")

        content = clean_text(content)

        messages.append({
            "role": role,
            "content": content,
        })

    out["conversations"] = messages

    # 時間欄位（原封不動保留）
    for key in ["t", "timestamp"]:
        if key in raw:
            out[key] = raw[key]

    return out


def main():
    parser = argparse.ArgumentParser(description="Convert ShareGPT to clean JSON format.")
    parser.add_argument("input", help="ShareGPT 原始資料")
    parser.add_argument("output", help="輸出 JSON 檔案")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data_list = data.get("data") or data.get("conversations") or [data]
    else:
        data_list = data

    converted = [convert_single_conv(conv) for conv in data_list]

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
