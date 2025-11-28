# vllm/prefix_stats_collector.py
#
# 全域 prefix sharing 統計收集器。

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Any


@dataclass
class RequestStats:
    total_prompt_tokens: int = 0
    prefix_hit_tokens: int = 0      # raw tokens (before clamp)
    hit_block_ids: List[int] = field(default_factory=list)


@dataclass
class BlockStats:
    # 總共被 hit 幾次（包含第一次）
    hit_count: int = 0
    # 上一次 reuse 的時間，用來算 interval（不輸出）
    last_ts: Optional[float] = None
    # 以下是 interval 的聚合統計（常數空間）
    interval_count: int = 0
    interval_sum: float = 0.0
    interval_min: float = float("inf")
    interval_max: float = 0.0


class PrefixStatsCollector:
    def __init__(self, block_size: int = 1) -> None:
        self._lock = threading.Lock()
        self._block_size = block_size

        # request_id -> RequestStats
        self._requests: Dict[str, RequestStats] = {}

        # block_id -> BlockStats
        self._blocks: Dict[int, BlockStats] = {}

    # ------------------------------------------------------------------
    # config
    # ------------------------------------------------------------------
    def set_block_size(self, block_size: int) -> None:
        with self._lock:
            self._block_size = max(1, int(block_size))

    def get_block_size(self) -> int:
        with self._lock:
            return self._block_size

    # ------------------------------------------------------------------
    # per-request
    # ------------------------------------------------------------------
    def record_prompt_length(self, request_id: str, prompt_tokens: int) -> None:
        if not request_id or prompt_tokens is None:
            return

        with self._lock:
            stats = self._requests.get(request_id)
            if stats is None:
                stats = RequestStats()
                self._requests[request_id] = stats
            stats.total_prompt_tokens = int(prompt_tokens)

    def record_hit(
        self,
        request_id: str,
        block_ids: Iterable[int],
        timestamp: Optional[float] = None,
    ) -> None:
        block_ids = list(block_ids)
        if not request_id or not block_ids:
            return
        if timestamp is None:
            timestamp = time.time()

        with self._lock:
            # per-request
            req_stats = self._requests.get(request_id)
            if req_stats is None:
                req_stats = RequestStats()
                self._requests[request_id] = req_stats

            # 去重，只增加新 hit 的 block
            existing = set(req_stats.hit_block_ids)
            new_blocks = [b for b in block_ids if b not in existing]
            if new_blocks:
                req_stats.hit_block_ids.extend(new_blocks)
                # raw prefix hit tokens，後續 snapshot 時會 clamp
                req_stats.prefix_hit_tokens += len(new_blocks) * self._block_size

            # per-block 聚合統計（常數空間）
            for b in block_ids:
                blk = self._blocks.get(b)
                if blk is None:
                    blk = BlockStats()
                    self._blocks[b] = blk

                # hit 次數累加
                blk.hit_count += 1

                # 只在有前一次時間時更新 interval 統計
                if blk.last_ts is not None:
                    interval = timestamp - blk.last_ts
                    if interval < 0:
                        # 時間戳如果出現倒退，就直接略過這筆 interval
                        pass
                    else:
                        blk.interval_count += 1
                        blk.interval_sum += interval
                        if interval < blk.interval_min:
                            blk.interval_min = interval
                        if interval > blk.interval_max:
                            blk.interval_max = interval

                # 更新最後一次時間
                blk.last_ts = timestamp

    # ------------------------------------------------------------------
    # snapshot / dump
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            block_size = self._block_size

            # -------------------------
            # per-request 輸出
            # -------------------------
            req_out: Dict[str, Any] = {}
            for rid, rs in self._requests.items():
                total = rs.total_prompt_tokens

                unique_blocks = set(rs.hit_block_ids)
                raw_hit_tokens = len(unique_blocks) * block_size

                # clamp：不能超過 total_prompt_tokens
                if total is not None and total > 0:
                    hit_tokens = min(total, raw_hit_tokens)
                else:
                    hit_tokens = 0

                ratio = (hit_tokens / total) if total and total > 0 else 0.0

                req_out[rid] = {
                    "total_prompt_tokens": total,
                    "prefix_hit_tokens": hit_tokens,
                    "prefix_hit_ratio": ratio,
                    "hit_block_ids": list(unique_blocks),
                }

            # -------------------------
            # per-block 聚合輸出
            # -------------------------
            blk_out: Dict[str, Any] = {}
            for bid, bs in self._blocks.items():
                if bs.interval_count > 0:
                    avg_interval = bs.interval_sum / bs.interval_count
                    min_interval = bs.interval_min
                    max_interval = bs.interval_max
                else:
                    avg_interval = 0.0
                    min_interval = 0.0
                    max_interval = 0.0

                blk_out[str(bid)] = {
                    "hit_count": bs.hit_count,
                    "reuse_interval_count": bs.interval_count,
                    "reuse_interval_avg": avg_interval,
                    "reuse_interval_min": min_interval,
                    "reuse_interval_max": max_interval,
                }

            return {
                "block_size": block_size,
                "requests": req_out,
                "blocks": blk_out,
            }

    def dump(self, path: str) -> None:
        data = self.snapshot()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def reset(self) -> None:
        with self._lock:
            self._requests.clear()
            self._blocks.clear()


global_prefix_collector = PrefixStatsCollector()
