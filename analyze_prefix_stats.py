import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt


# ---------- 基本工具 ----------

def load_stats(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    reqs = list(data.get("requests", {}).values())
    blks = list(data.get("blocks", {}).values())
    return reqs, blks


def ensure_out_dir(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)


def cdf_from_array(arr: np.ndarray):
    """給一個 1D array，回傳 (xs, ys) 當作 CDF 曲線."""
    arr = np.sort(arr)
    n = len(arr)
    ys = np.arange(1, n + 1) / n
    return arr, ys


# ---------- 圖 1：prefix_hit_ratio CDF ----------

def plot_prefix_hit_ratio_cdf(reqs_single, reqs_multi, out_dir):
    r_single = np.array([r.get("prefix_hit_ratio", 0.0) for r in reqs_single], dtype=float)
    r_multi = np.array([r.get("prefix_hit_ratio", 0.0) for r in reqs_multi], dtype=float)

    # 避免空 array
    if len(r_single) == 0 or len(r_multi) == 0:
        print("[warn] prefix_hit_ratio CDF: single 或 multi 沒有 request，略過此圖。")
        return

    xs_s, ys_s = cdf_from_array(r_single)
    xs_m, ys_m = cdf_from_array(r_multi)

    plt.figure()
    plt.plot(xs_s, ys_s, label="Single-turn")
    plt.plot(xs_m, ys_m, label="Multi-turn")
    plt.xlabel("Prefix hit ratio")
    plt.ylabel("CDF")
    plt.title("CDF of per-request prefix hit ratio")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "prefix_hit_ratio_cdf.png"))
    plt.close()
    print("[info] saved prefix_hit_ratio_cdf.png")


# ---------- 圖 2：有 hit 的 request 比例 bar chart ----------

def plot_benefit_fraction_bar(reqs_single, reqs_multi, out_dir):
    def calc_fraction(reqs):
        if not reqs:
            return 0.0, 0
        hits = sum(1 for r in reqs if r.get("prefix_hit_tokens", 0) > 0)
        return hits / len(reqs), len(reqs)

    frac_s, n_s = calc_fraction(reqs_single)
    frac_m, n_m = calc_fraction(reqs_multi)

    if n_s == 0 or n_m == 0:
        print("[warn] benefit fraction: single 或 multi 沒有 request，略過此圖。")
        return

    labels = ["Single-turn", "Multi-turn"]
    values = [frac_s, frac_m]

    plt.figure()
    x = np.arange(len(labels))
    plt.bar(x, values)
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel("Fraction of requests with prefix hits")
    plt.title("Requests that benefit from prefix sharing\n(prefix_hit_tokens > 0)")
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "benefit_fraction_bar.png"))
    plt.close()
    print("[info] saved benefit_fraction_bar.png")


# ---------- 圖 3：block hit_count 分布（CDF） ----------

def plot_block_hit_count_cdf(blks_single, blks_multi, out_dir):
    h_single = np.array([b.get("hit_count", 0) for b in blks_single], dtype=float)
    h_multi = np.array([b.get("hit_count", 0) for b in blks_multi], dtype=float)

    # 只看 hit_count >= 1 的 block，比較有意義
    h_single = h_single[h_single >= 1]
    h_multi = h_multi[h_multi >= 1]

    if len(h_single) == 0 or len(h_multi) == 0:
        print("[warn] block hit_count CDF: single 或 multi 沒有 hit>=1 的 block，略過此圖。")
        return

    xs_s, ys_s = cdf_from_array(h_single)
    xs_m, ys_m = cdf_from_array(h_multi)

    plt.figure()
    plt.plot(xs_s, ys_s, label="Single-turn")
    plt.plot(xs_m, ys_m, label="Multi-turn")
    plt.xlabel("Block hit count")
    plt.ylabel("CDF")
    plt.title("CDF of block hit counts (hit_count >= 1)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "block_hit_count_cdf.png"))
    plt.close()
    print("[info] saved block_hit_count_cdf.png")


# ---------- 圖 4：reuse interval 分布（平均值 CDF，log-scale） ----------

def plot_reuse_interval_cdf(blks_single, blks_multi, out_dir):
    def extract_intervals(blks):
        vals = []
        for b in blks:
            cnt = b.get("reuse_interval_count", 0)
            if cnt and cnt > 0:
                avg = b.get("reuse_interval_avg", 0.0)
                if avg and avg > 0:
                    vals.append(avg)
        return np.array(vals, dtype=float)

    i_single = extract_intervals(blks_single)
    i_multi = extract_intervals(blks_multi)

    if len(i_single) == 0 or len(i_multi) == 0:
        print("[warn] reuse interval CDF: single 或 multi 沒有 interval>0 的 block，略過此圖。")
        return

    # 為了視覺化，通常 interval 很長，用 log10
    log_s = np.log10(i_single)
    log_m = np.log10(i_multi)

    xs_s, ys_s = cdf_from_array(log_s)
    xs_m, ys_m = cdf_from_array(log_m)

    plt.figure()
    plt.plot(xs_s, ys_s, label="Single-turn")
    plt.plot(xs_m, ys_m, label="Multi-turn")
    plt.xlabel("log10(reuse_interval_avg) (seconds)")
    plt.ylabel("CDF")
    plt.title("CDF of block reuse intervals (average, log10 scale)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reuse_interval_cdf.png"))
    plt.close()
    print("[info] saved reuse_interval_cdf.png")


# ---------- Summary 表的一些 aggregate（印在 console 給你抄到 report） ----------

def print_summary(reqs_single, blks_single, reqs_multi, blks_multi):
    def req_stats(reqs):
        if not reqs:
            return {
                "num_requests": 0,
                "avg_ratio": 0.0,
                "median_ratio": 0.0,
                "benefit_fraction": 0.0,
            }
        ratios = np.array([r.get("prefix_hit_ratio", 0.0) for r in reqs], dtype=float)
        benefit = np.mean([r.get("prefix_hit_tokens", 0) > 0 for r in reqs])
        return {
            "num_requests": len(reqs),
            "avg_ratio": float(np.mean(ratios)),
            "median_ratio": float(np.median(ratios)),
            "benefit_fraction": float(benefit),
        }

    def blk_stats(blks):
        if not blks:
            return {
                "num_blocks": 0,
                "avg_hit_count": 0.0,
                "max_hit_count": 0.0,
            }
        hit_counts = np.array([b.get("hit_count", 0) for b in blks], dtype=float)
        return {
            "num_blocks": len(blks),
            "avg_hit_count": float(np.mean(hit_counts)),
            "max_hit_count": float(np.max(hit_counts)),
        }

    rs = req_stats(reqs_single)
    rm = req_stats(reqs_multi)
    bs = blk_stats(blks_single)
    bm = blk_stats(blks_multi)

    print("\n===== Summary: Single-turn vs Multi-turn =====")
    print("Single-turn:")
    print(f"  #requests            : {rs['num_requests']}")
    print(f"  avg prefix_hit_ratio : {rs['avg_ratio']:.4f}")
    print(f"  median prefix_hit_ratio: {rs['median_ratio']:.4f}")
    print(f"  fraction with hits   : {rs['benefit_fraction']:.4f}")
    print(f"  #blocks              : {bs['num_blocks']}")
    print(f"  avg block hit_count  : {bs['avg_hit_count']:.4f}")
    print(f"  max block hit_count  : {bs['max_hit_count']:.0f}")

    print("\nMulti-turn:")
    print(f"  #requests            : {rm['num_requests']}")
    print(f"  avg prefix_hit_ratio : {rm['avg_ratio']:.4f}")
    print(f"  median prefix_hit_ratio: {rm['median_ratio']:.4f}")
    print(f"  fraction with hits   : {rm['benefit_fraction']:.4f}")
    print(f"  #blocks              : {bm['num_blocks']}")
    print(f"  avg block hit_count  : {bm['avg_hit_count']:.4f}")
    print(f"  max block hit_count  : {bm['max_hit_count']:.0f}")
    print("==============================================\n")


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze prefix sharing stats (single vs multi) and plot figures."
    )
    parser.add_argument("--single", required=True, help="Single-turn stats JSON file")
    parser.add_argument("--multi", required=True, help="Multi-turn stats JSON file")
    parser.add_argument("--out_dir", default="figs", help="Output directory for figures")
    args = parser.parse_args()

    ensure_out_dir(args.out_dir)

    reqs_single, blks_single = load_stats(args.single)
    reqs_multi, blks_multi = load_stats(args.multi)

    print_summary(reqs_single, blks_single, reqs_multi, blks_multi)
    plot_prefix_hit_ratio_cdf(reqs_single, reqs_multi, args.out_dir)
    plot_benefit_fraction_bar(reqs_single, reqs_multi, args.out_dir)
    plot_block_hit_count_cdf(blks_single, blks_multi, args.out_dir)
    plot_reuse_interval_cdf(blks_single, blks_multi, args.out_dir)


if __name__ == "__main__":
    main()
