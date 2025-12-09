import os
import sys

# 把上一層（專案根）加進 sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from workload_datasets.agentbank_loader import load_agentbank_all
from workload_datasets.ccbench_loader import load_ccbench_standard
from workload_datasets.qwen_trace_loader import load_qwen_as_standard
from workload_datasets.export_for_m2 import export_for_m2


def main() -> None:
    # 1) 載入三種 workload 的 StandardExample
    agent_examples = load_agentbank_all()
    coding_examples = load_ccbench_standard()
    qwenA_examples = load_qwen_as_standard(
        "workload_datasets/qwen-bailian-usagetraces-anon/qwen_traceA_blksz_16.jsonl",
        workload_type="online_trace_A",
    )

    print("AgentBank:", len(agent_examples))
    print("CC-Bench:", len(coding_examples))
    print("QwenTraceA:", len(qwenA_examples))

    # 2) 示範幾種情境

    # (a) 只用 AgentBank 當一個純 agent 的 M2 訓練集
    export_for_m2(
        agent_examples,
        data_path="workload_datasets_test/m2_data_agent_only.json",
        index_path="workload_datasets_test/m2_index_agent_only.jsonl",
    )
    

    # # (b) Agent + Coding 混在一起
    # mixed_agent_coding = agent_examples + coding_examples
    # export_for_m2(
    #     mixed_agent_coding,
    #     data_path="workload_datasets_test/m2_data_agent_coding.json",
    #     index_path="workload_datasets_test/m2_index_agent_coding.jsonl",
    # )

    # # (c) 三種全部一起（簡單全部 concat）
    # all_examples = agent_examples + coding_examples + qwenA_examples
    # export_for_m2(
    #     all_examples,
    #     data_path="workload_datasets_test/m2_data_all.json",
    #     index_path="workload_datasets_test/m2_index_all.jsonl",
    # )

    print("Done. 已輸出 m2_data_*.json 和 m2_index_*.jsonl")


if __name__ == "__main__":
    main()
