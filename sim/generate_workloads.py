#!/usr/bin/env python3
import argparse
import json
import os
from typing import List, Dict

# ---------------------------------------------------
# 基本 prompt 模板：同一個 template_id 會使用完全相同的 prompt
# 這樣才會有強烈的 prefix sharing（A 的所有 request 完全一致）
# ---------------------------------------------------

TEMPLATES = {
    "A": (
        "You are a helpful AI assistant that answers billing questions for a SaaS product. "
        "Please carefully read the user's question and provide a clear, step-by-step explanation "
        "of the billing policy and what they should do next."
    ),
    "B": (
        "You are a helpful AI assistant that explains error messages for backend APIs. "
        "Given the user's error log, analyze possible root causes and suggest concrete debugging steps."
    ),
    "C": (
        "You are a creative writing assistant. The user will give you a writing prompt, and you should "
        "continue the story in a descriptive and imaginative way."
    ),
    "D": (
        "You are a coding assistant specialized in Python. The user will describe a function they want, "
        "and you should write clean, well-documented Python code that implements it."
    ),
}


def make_conv(conv_id: str, template_id: str) -> Dict:
    """建立一個 ShareGPT 風格的單輪對話：只有 user 一句話。"""
    prompt = TEMPLATES[template_id]
    return {
        "id": conv_id,
        "conversations": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }


# ---------------------------------------------------
# Workload A：FIFO 壞，LRU/LFU 好
#  - A/B 一直很熱，中間偶爾插 C 當雜訊
#  - 早期就把 A/B 塞滿 cache，之後又用很多次
# ---------------------------------------------------

def build_workload_A(num_requests: int) -> List[Dict]:
    seq: List[str] = []

    # Phase 1：暖機，A/B 交錯
    p1 = max(10, num_requests // 4)
    for i in range(p1):
        seq.append("A" if i % 2 == 0 else "B")

    # Phase 2：大量 A/B，中間偶爾插 C，讓 FIFO 做出錯誤淘汰
    p2 = max(20, num_requests // 2)
    for i in range(p2):
        if i % 10 == 0:
            seq.append("C")  # 偶爾插入冷門 C
        else:
            seq.append("A" if i % 2 == 0 else "B")

    # Phase 3：再回到 A/B 洪流
    remaining = max(0, num_requests - len(seq))
    for i in range(remaining):
        seq.append("A" if i % 2 == 0 else "B")

    seq = seq[:num_requests]

    convs: List[Dict] = []
    for i, t_id in enumerate(seq):
        conv_id = f"workloadA_{i+1}"
        convs.append(make_conv(conv_id, t_id))
    return convs


# ---------------------------------------------------
# Workload B：LFU 最好
#  - 整體來看 A/B 的出現次數遠高於 C/D（frequency skew）
#  - 但 C/D 在某些短區間內連續出現，容易誤導 LRU
# ---------------------------------------------------

def build_workload_B(num_requests: int) -> List[Dict]:
    seq: List[str] = []

    n = num_requests
    p1 = max(20, int(0.35 * n))
    p2 = max(5, int(0.10 * n))
    p3 = max(20, int(0.35 * n))
    p4 = max(5, n - (p1 + p2 + p3))

    # Phase 1：A/B 熱身（高頻）
    for i in range(p1):
        seq.append("A" if i % 2 == 0 else "B")

    # Phase 2：短期 C burst
    for _ in range(p2):
        seq.append("C")

    # Phase 3：再回到 A/B 大量使用（進一步拉高 A/B 的 frequency）
    for i in range(p3):
        seq.append("A" if i % 2 == 0 else "B")

    # Phase 4：短期 D burst
    for _ in range(p4):
        seq.append("D")

    seq = seq[:num_requests]

    convs: List[Dict] = []
    for i, t_id in enumerate(seq):
        conv_id = f"workloadB_{i+1}"
        convs.append(make_conv(conv_id, t_id))
    return convs


# ---------------------------------------------------
# Workload C：LRU 最好
#  - 前半段：A/B 非常熱門
#  - 後半段：完全 phase shift，改成 C/D 熱門
#  - LFU 卡在「歷史上 A/B 計數很高」，不容易轉向 C/D
# ---------------------------------------------------

def build_workload_C(num_requests: int) -> List[Dict]:
    seq: List[str] = []
    half = num_requests // 2

    # Phase 1：A/B
    for i in range(half):
        seq.append("A" if i % 2 == 0 else "B")

    # Phase 2：C/D
    for i in range(num_requests - half):
        seq.append("C" if i % 2 == 0 else "D")

    seq = seq[:num_requests]

    convs: List[Dict] = []
    for i, t_id in enumerate(seq):
        conv_id = f"workloadC_{i+1}"
        convs.append(make_conv(conv_id, t_id))
    return convs


# ---------------------------------------------------
# main
# ---------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate three synthetic workloads (A/B/C) for cache policy comparison."
    )
    parser.add_argument(
        "--out_dir",
        default="synthetic_traces",
        help="輸出資料夾（預設: synthetic_traces）",
    )
    parser.add_argument(
        "--num_requests",
        type=int,
        default=500,
        help="每個 workload 要產生多少個 request（對話），預設 500",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[info] Generating workloads with {args.num_requests} requests each...")

    wa = build_workload_A(args.num_requests)
    wb = build_workload_B(args.num_requests)
    wc = build_workload_C(args.num_requests)

    path_a = os.path.join(args.out_dir, "workload_A.json")
    path_b = os.path.join(args.out_dir, "workload_B.json")
    path_c = os.path.join(args.out_dir, "workload_C.json")

    with open(path_a, "w", encoding="utf-8") as f:
        json.dump(wa, f, ensure_ascii=False, indent=2)
    with open(path_b, "w", encoding="utf-8") as f:
        json.dump(wb, f, ensure_ascii=False, indent=2)
    with open(path_c, "w", encoding="utf-8") as f:
        json.dump(wc, f, ensure_ascii=False, indent=2)

    print(f"[info] Saved:")
    print(f"  {path_a}")
    print(f"  {path_b}")
    print(f"  {path_c}")


if __name__ == "__main__":
    main()
