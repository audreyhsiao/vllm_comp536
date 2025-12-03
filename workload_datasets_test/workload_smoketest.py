import os
import sys

# 把上一層（專案根）加進 sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from workload_datasets.agentbank_loader import load_agentbank_all
from workload_datasets.ccbench_loader import load_ccbench_standard
from workload_datasets.qwen_trace_loader import load_qwen_as_standard
from workload_datasets.standard_schema import StandardExample



def describe(name: str, xs: list[StandardExample], n: int = 3) -> None:
    print(f"=== {name} ===")
    print("num examples:", len(xs))
    print()

    for i, ex in enumerate(xs[:n]):
        print(f"[{i}] id={ex.id}")
        print(f"  workload_type = {ex.workload_type}")
        print(f"  #messages     = {len(ex.conversations)}")
        print(f"  timestamp     = {ex.timestamp}")
        print(f"  meta          = {ex.meta}")
        if ex.conversations:
            first = ex.conversations[0]
            print(f"  first msg    = ({first.role}) {first.content[:80]!r}")
        print()


def main() -> None:
    
    agent_examples = load_agentbank_all()
    describe("AgentBank", agent_examples)

    coding_examples = load_ccbench_standard()
    describe("CC-Bench", coding_examples)

    qwenA = load_qwen_as_standard(
        "workload_datasets/qwen-bailian-usagetraces-anon/qwen_traceA_blksz_16.jsonl",
        workload_type="online_trace_A",
    )
    describe("QwenTraceA", qwenA)


if __name__ == "__main__":
    main()
