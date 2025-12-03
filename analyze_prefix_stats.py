#!/usr/bin/env python3
import argparse
import json
import os
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt


# ---------- 基本工具 ----------

def load_stats(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    reqs = list(data.get("requests", {}).values())
    blks = list(data.get("blocks", {}).values())
    return reqs, blks


def ensure_out_dir(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)


def cdf_from_array(arr: np.ndarray):
    """給一個 1D array，回傳 (xs, ys) 當作 CDF 曲線."""
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return arr, arr
    arr = np.sort(arr)
    n = len(arr)
    ys = np.arange(1, n + 1) / n
    return arr, ys


# ---------- 團隊實驗封裝 ----------

class Experiment:
    def __init__(self, name: str, reqs: List[Dict[str, Any]], blks: List[Dict[str, Any]]):
        self.name = name
        self.reqs = reqs
        self.blks = blks


# ---------- 圖 1：prefix_hit_ratio CDF ----------

def plot_prefix_hit_ratio_cdf(experiments: List[Experiment], out_dir: str):
    plt.figure()
    any_valid = False

    for exp in experiments:
        r = np.array(
            [req.get("prefix_hit_ratio", 0.0) for req in exp.reqs],
            dtype=float
        )
        if len(r) == 0:
            print(f"[warn] prefix_hit_ratio CDF: {exp.name} 沒有 request，略過。")
            continue

        xs, ys = cdf_from_array(r)
        if xs.size == 0:
            print(f"[warn] prefix_hit_ratio CDF: {exp.name} 無有效資料，略過。")
            continue

        plt.plot(xs, ys, label=exp.name)
        any_valid = True

    if not any_valid:
        print("[warn] prefix_hit_ratio CDF: 所有實驗都沒有資料，略過此圖。")
        plt.close()
        return

    plt.xlabel("Prefix hit ratio")
    plt.ylabel("CDF")
    plt.title("CDF of per-request prefix hit ratio")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "prefix_hit_ratio_cdf.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[info] saved {out_path}")


# ---------- 圖 2：有 hit 的 request 比例 bar chart ----------

def plot_benefit_fraction_bar(experiments: List[Experiment], out_dir: str):
    names = []
    values = []

    for exp in experiments:
        reqs = exp.reqs
        if not reqs:
            print(f"[warn] benefit fraction: {exp.name} 沒有 request，略過。")
            continue
        hits = sum(1 for r in reqs if r.get("prefix_hit_tokens", 0) > 0)
        frac = hits / len(reqs)
        names.append(exp.name)
        values.append(frac)

    if not names:
        print("[warn] benefit fraction: 所有實驗都沒有 request，略過此圖。")
        return

    x = np.arange(len(names))

    plt.figure()
    plt.bar(x, values)
    plt.xticks(x, names, rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Fraction of requests with prefix hits")
    plt.title("Requests that benefit from prefix sharing\n(prefix_hit_tokens > 0)")

    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "benefit_fraction_bar.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[info] saved {out_path}")


# ---------- 圖 3：block hit_count 分布（CDF） ----------

def plot_block_hit_count_cdf(experiments: List[Experiment], out_dir: str):
    plt.figure()
    any_valid = False

    for exp in experiments:
        h = np.array(
            [b.get("hit_count", 0) for b in exp.blks],
            dtype=float
        )
        h = h[h >= 1]  # 只看 hit_count >= 1 的 block

        if len(h) == 0:
            print(f"[warn] block hit_count CDF: {exp.name} 沒有 hit>=1 的 block，略過。")
            continue

        xs, ys = cdf_from_array(h)
        if xs.size == 0:
            print(f"[warn] block hit_count CDF: {exp.name} 無有效資料，略過。")
            continue

        plt.plot(xs, ys, label=exp.name)
        any_valid = True

    if not any_valid:
        print("[warn] block hit_count CDF: 所有實驗都沒有 hit>=1 的 block，略過此圖。")
        plt.close()
        return

    plt.xlabel("Block hit count")
    plt.ylabel("CDF")
    plt.title("CDF of block hit counts (hit_count >= 1)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "block_hit_count_cdf.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[info] saved {out_path}")


# ---------- 圖 4：reuse interval 分布（平均值 CDF，log-scale） ----------

def plot_reuse_interval_cdf(experiments: List[Experiment], out_dir: str):
    plt.figure()
    any_valid = False

    for exp in experiments:
        vals = []
        for b in exp.blks:
            cnt = b.get("reuse_interval_count", 0)
            if cnt and cnt > 0:
                avg = b.get("reuse_interval_avg", 0.0)
                if avg and avg > 0:
                    vals.append(avg)

        if not vals:
            print(f"[warn] reuse interval CDF: {exp.name} 沒有 interval>0 的 block，略過。")
            continue

        vals = np.array(vals, dtype=float)
        log_vals = np.log10(vals)
        xs, ys = cdf_from_array(log_vals)
        if xs.size == 0:
            print(f"[warn] reuse interval CDF: {exp.name} 無有效資料，略過。")
            continue

        plt.plot(xs, ys, label=exp.name)
        any_valid = True

    if not any_valid:
        print("[warn] reuse interval CDF: 所有實驗都沒有 interval>0 的 block，略過此圖。")
        plt.close()
        return

    plt.xlabel("log10(reuse_interval_avg) (seconds)")
    plt.ylabel("CDF")
    plt.title("CDF of block reuse intervals (average, log10 scale)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "reuse_interval_cdf.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[info] saved {out_path}")


# ---------- Summary：印在 console ----------

def print_summary(experiments: List[Experiment]):
    print("\n===== Summary: Prefix sharing stats =====")
    for exp in experiments:
        reqs = exp.reqs
        blks = exp.blks

        # requests
        if reqs:
            ratios = np.array(
                [r.get("prefix_hit_ratio", 0.0) for r in reqs],
                dtype=float
            )
            benefit = np.mean([r.get("prefix_hit_tokens", 0) > 0 for r in reqs])
            num_req = len(reqs)
            avg_ratio = float(np.mean(ratios))
            median_ratio = float(np.median(ratios))
            benefit_fraction = float(benefit)
        else:
            num_req = 0
            avg_ratio = 0.0
            median_ratio = 0.0
            benefit_fraction = 0.0

        # blocks
        if blks:
            hit_counts = np.array(
                [b.get("hit_count", 0) for b in blks],
                dtype=float
            )
            num_blk = len(blks)
            avg_hit = float(np.mean(hit_counts))
            max_hit = float(np.max(hit_counts))
        else:
            num_blk = 0
            avg_hit = 0.0
            max_hit = 0.0

        print(f"\n[{exp.name}]")
        print(f"  #requests              : {num_req}")
        print(f"  avg prefix_hit_ratio   : {avg_ratio:.4f}")
        print(f"  median prefix_hit_ratio: {median_ratio:.4f}")
        print(f"  fraction with hits     : {benefit_fraction:.4f}")
        print(f"  #blocks                : {num_blk}")
        print(f"  avg block hit_count    : {avg_hit:.4f}")
        print(f"  max block hit_count    : {max_hit:.0f}")
    print("==============================================\n")


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze prefix sharing stats for multiple experiments and plot figures."
    )
    # 多組：--stats NAME PATH，可以重複
    parser.add_argument(
        "--stats",
        metavar=("NAME", "JSON"),
        nargs=2,
        action="append",
        required=True,
        help="實驗名稱與 stats JSON 路徑，例如: --stats 'LFU-single' out/lfu_single.json",
    )
    parser.add_argument(
        "--out_dir",
        default="figs",
        help="Output directory for figures",
    )
    args = parser.parse_args()

    ensure_out_dir(args.out_dir)

    experiments: List[Experiment] = []
    for name, path in args.stats:
        reqs, blks = load_stats(path)
        print(f"[info] loaded {name}: {len(reqs)} requests, {len(blks)} blocks from {path}")
        experiments.append(Experiment(name, reqs, blks))

    print_summary(experiments)
    plot_prefix_hit_ratio_cdf(experiments, args.out_dir)
    plot_benefit_fraction_bar(experiments, args.out_dir)
    plot_block_hit_count_cdf(experiments, args.out_dir)
    plot_reuse_interval_cdf(experiments, args.out_dir)


if __name__ == "__main__":
    main()