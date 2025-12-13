import json
import random
from typing import List, Dict, Any


def tag_workload(examples: List[Dict[str, Any]], workload_type: str) -> List[Dict[str, Any]]:
    """幫每個 example 加上 workload_type 欄位"""
    out = []
    for ex in examples:
        ex = dict(ex)  # 做淺拷貝避免原資料被改
        ex["workload_type"] = workload_type
        out.append(ex)
    return out


def mix_workloads(
    agentbank: List[Dict[str, Any]],
    ccbench: List[Dict[str, Any]],
    qwen: List[Dict[str, Any]],
    ratios=(0.4, 0.3, 0.3),
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    將三種 workloads 依照給定比例混成一條 request 序列
    不打亂各自內部順序
    """
    rng = random.Random(seed)

    sources = {
        "agentbank": tag_workload(agentbank, "agentbank"),
        "ccbench": tag_workload(ccbench, "ccbench"),
        "qwen": tag_workload(qwen, "qwen"),
    }

    keys_all = ["agentbank", "ccbench", "qwen"]
    weights_all = list(ratios)

    # 每個 workload 目前讀到第幾筆
    idx = {k: 0 for k in keys_all}

    total_len = sum(len(v) for v in sources.values())
    mixed: List[Dict[str, Any]] = []

    while len(mixed) < total_len:
        # 篩掉已經用完的 workload
        active_keys = []
        active_weights = []
        for k, w in zip(keys_all, weights_all):
            if idx[k] < len(sources[k]):
                active_keys.append(k)
                active_weights.append(w)

        if not active_keys:
            break

        weight_sum = sum(active_weights)
        r = rng.random() * weight_sum

        acc = 0.0
        chosen = active_keys[-1]
        for k, w in zip(active_keys, active_weights):
            acc += w
            if r <= acc:
                chosen = k
                break

        ex = sources[chosen][idx[chosen]]
        idx[chosen] += 1
        mixed.append(ex)

    return mixed

with open("sim/dataset_agentbank_json/alfred.json") as f:
    agentbank = json.load(f)
with open("sim/dataset_ccbench/ccbench_50.json") as f:
    ccbench = json.load(f)
with open("sim/dataset_qwen/qwen_200.json") as f:
    qwen = json.load(f)

mixed = mix_workloads(agentbank, ccbench, qwen, ratios=(0.4, 0.3, 0.3), seed=0)

with open("mixed_workload.json", "w") as f:
    json.dump(mixed, f, ensure_ascii=False, indent=2)