import os
import sys
import random

# 把上一層（專案根）加進 sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from workload_datasets.agentbank_loader import load_agentbank_all
from workload_datasets.ccbench_loader import load_ccbench_standard
from workload_datasets.qwen_trace_loader import load_qwen_as_standard
from workload_datasets.export_for_m2 import export_for_m2


def mix_examples(agent, coding, qwen, weights=(1/3, 1/3, 1/3), total=3000, seed=0, shuffle=True):
    wa, wc, wq = weights
    s = wa + wc + wq
    wa, wc, wq = wa/s, wc/s, wq/s

    # target counts (rounding-safe)
    na = round(wa * total)
    nc = round(wc * total)
    nq = total - na - nc

    rng = random.Random(seed)
    a, c, q = list(agent), list(coding), list(qwen)
    if shuffle:
        rng.shuffle(a); rng.shuffle(c); rng.shuffle(q)

    return (a[:na] + c[:nc] + q[:nq]) if not shuffle else rng.sample(a[:na] + c[:nc] + q[:nq], k=min(total, len(a[:na] + c[:nc] + q[:nq])))

def main() -> None:
    # 1) 載入三種 workload 的 StandardExample
    agent_examples = load_agentbank_all()
    coding_examples = load_ccbench_standard()
    qwenA_examples = load_qwen_as_standard(
        "/home/p4/vllm_comp536/workload_datasets/qwen-bailian-usagetraces-anon/qwen_traceA_blksz_16.jsonl",
        workload_type="online_trace_A",
    )

    # print("AgentBank:", len(agent_examples))
    # print("CC-Bench:", len(coding_examples))
    # print("QwenTraceA:", len(qwenA_examples))

    # 2) 示範幾種情境

    # (a) 只用 AgentBank 當一個純 agent 的 M2 訓練集
    # export_for_m2(
    #     agent_examples,
    #     data_path="workload_datasets_test/m2_data_agent_only.json",
    #     index_path="workload_datasets_test/m2_index_agent_only.jsonl",
    # )
    

    # # (b) Agent + Coding 混在一起
    # mixed_agent_coding = agent_examples + coding_examples
    # export_for_m2(
    #     mixed_agent_coding,
    #     data_path="workload_datasets_test/m2_data_agent_coding.json",
    #     index_path="workload_datasets_test/m2_index_agent_coding.jsonl",
    # )

    # (c) 三種全部一起（簡單全部 concat）
    #  個只取前 10 筆（如果本來少於 10，slice 也不會壞掉）
    # agent_subset = agent_examples[:3]
    # coding_subset = coding_examples[:3]
    # qwenA_subset = qwenA_examples[:3]

    # all_examples = agent_subset + coding_subset + qwenA_subset

    # export_for_m2(
    #     all_examples,
    #     data_path="m2_data_all.json",
    #     index_path="m2_index_all.jsonl",
    # )


    # (d) 三種全部一起（依比例隨機抽樣混合）
    all_examples = mix_examples(
        agent_examples, coding_examples, qwenA_examples,
        weights=(0.5, 0.4, 0.1),
        total=15,
        seed=42
    )

    export_for_m2(all_examples, "m2_data_all.json", "m2_index_all.jsonl")


    print("Done. 已輸出 m2_data_*.json 和 m2_index_*.jsonl")


if __name__ == "__main__":
    main()
