import enum
import heapq
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional


class EvictionPolicy(enum.Enum):
    """Enum for eviction policy used by make_evictor to instantiate the correct
       Evictor subclass.
    """
    LRU = enum.auto()
    LFU = enum.auto()
    FIFO = enum.auto()
    WORKLOAD_AWARE = enum.auto() 


class Evictor(ABC):
    """The Evictor subclasses should be used by the BlockAllocator class to
    handle eviction of freed Blocks.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __contains__(self, block_id: int) -> bool:
        pass

    @abstractmethod
    def evict(self) -> Tuple[int, int]:
        """Runs the eviction algorithm and returns the evicted block's
        content hash along with physical block id along with physical block id
        """
        pass

    @abstractmethod
    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        """Adds block to the evictor, making it a candidate for eviction"""
        pass

    @abstractmethod
    def update(self, block_id: int, last_accessed: float):
        """Update corresponding block's access time in metadata"""
        pass

    @abstractmethod
    def remove(self, block_id: int):
        """Remove a given block id from the cache."""
        pass

    @property
    @abstractmethod
    def num_blocks(self) -> int:
        pass


class BlockMetaData:
    """Data structure for storing key data describe cached block, so that
    evitor could use to make its decision which one to choose for eviction

    Here we use physical block id as the dict key, as there maybe several
    blocks with the same content hash, but their physical id is unique.
    """

    def __init__(self, content_hash: int, num_hashed_tokens: int,
                 last_accessed: float, workload_type: Optional[int] = None):
        self.content_hash = content_hash
        self.num_hashed_tokens = num_hashed_tokens
        self.last_accessed = last_accessed
        # NEW: 記錄被 access 的次數（LFU 用）
        self.access_count: int = 1
        self.workload_type: Optional[int] = workload_type


class LRUEvictor(Evictor):
    """Evicts in a least-recently-used order using the last_accessed timestamp
    that's recorded in the Block. If there are multiple blocks with
    the same last_accessed time, then the one with the largest num_hashed_tokens
    will be evicted. If two blocks each have the lowest last_accessed time and
    highest num_hashed_tokens value, then one will be chose arbitrarily
    """

    # CLEANUP_THRESHOLD determines the maximum allowable size of the priority
    # queue relative to the free table size. When this threshold is exceeded,
    # a cleanup operation is triggered to reduce memory usage.
    CLEANUP_THRESHOLD = 50

    def __init__(self):
        self.free_table: Dict[int, BlockMetaData] = {}
        self.priority_queue = []

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def evict(self) -> Tuple[int, int]:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        while self.priority_queue:
            # We do not remove outdated entries from the priority queue at the
            # time of updating the last_accessed timestamp. Instead, outdated
            # entries are filtered out here during eviction. Outdated entries
            # would either not in the free table, or have older last accessed
            # time.
            last_accessed, _, block_id, content_hash = heapq.heappop(
                self.priority_queue)
            if (block_id in self.free_table and
                    self.free_table[block_id].last_accessed == last_accessed):
                self.free_table.pop(block_id)
                return block_id, content_hash

        raise ValueError("No usable cache memory left")

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        self.free_table[block_id] = BlockMetaData(content_hash,
                                                  num_hashed_tokens,
                                                  last_accessed)
        heapq.heappush(
            self.priority_queue,
            (last_accessed, -num_hashed_tokens, block_id, content_hash))
        self._cleanup_if_necessary()

    def update(self, block_id: int, last_accessed: float):
        self.free_table[block_id].last_accessed = last_accessed

    def _cleanup_if_necessary(self):
        if len(self.priority_queue) > LRUEvictor.CLEANUP_THRESHOLD * len(
                self.free_table):
            self._cleanup()

    def _cleanup(self):
        new_priority_queue: List[Tuple[float, int, int, int]] = []

        for block_id, block in self.free_table.items():
            new_priority_queue.append(
                (block.last_accessed, -block.num_hashed_tokens, block_id,
                 block.content_hash))
        heapq.heapify(new_priority_queue)

        self.priority_queue = new_priority_queue

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        self.free_table.pop(block_id)

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)

class LFUEvictor(Evictor):
    """Evicts the Least Frequently Used block.
    Primary key: access_count (smaller is evicted first)
    Tie-breaker: last_accessed (older is evicted first)
    """

    CLEANUP_THRESHOLD = 50

    def __init__(self):
        self.free_table: Dict[int, BlockMetaData] = {}
        # (access_count, last_accessed, block_id, content_hash)
        self.priority_queue: List[Tuple[int, float, int, int]] = []

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def evict(self) -> Tuple[int, int]:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        while self.priority_queue:
            access_count, last_accessed, block_id, content_hash = heapq.heappop(
                self.priority_queue
            )
            meta = self.free_table.get(block_id)
            # 濾掉舊 entry：不在 free_table 或 meta 已經更新過
            if meta is not None and \
               meta.access_count == access_count and \
               meta.last_accessed == last_accessed:
                self.free_table.pop(block_id)
                return block_id, content_hash

        raise ValueError("No usable cache memory left")

    def add(self, block_id: int, content_hash: int,
            num_hashed_tokens: int, last_accessed: float):
        meta = BlockMetaData(content_hash, num_hashed_tokens, last_accessed)
        meta.access_count = 1  # 新加入視為第一次使用
        self.free_table[block_id] = meta
        heapq.heappush(
            self.priority_queue,
            (meta.access_count, meta.last_accessed, block_id, content_hash),
        )
        self._cleanup_if_necessary()

    def update(self, block_id: int, last_accessed: float):
        meta = self.free_table.get(block_id)
        if meta is None:
            return
        meta.access_count += 1
        meta.last_accessed = last_accessed
        # 丟入新的狀態（舊的會在 evict 時被濾掉）
        heapq.heappush(
            self.priority_queue,
            (meta.access_count, meta.last_accessed, block_id,
             meta.content_hash),
        )
        self._cleanup_if_necessary()

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor"
            )
        self.free_table.pop(block_id)

    def _cleanup_if_necessary(self):
        if len(self.priority_queue) > self.CLEANUP_THRESHOLD * len(
                self.free_table):
            self._cleanup()

    def _cleanup(self):
        new_priority_queue: List[Tuple[int, float, int, int]] = []
        for block_id, meta in self.free_table.items():
            new_priority_queue.append(
                (meta.access_count, meta.last_accessed, block_id,
                 meta.content_hash)
            )
        heapq.heapify(new_priority_queue)
        self.priority_queue = new_priority_queue

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)

class FIFOEvictor(Evictor):
    """Evicts blocks in First-In-First-Out order (insertion order)."""

    def __init__(self):
        self.free_table: Dict[int, BlockMetaData] = {}
        # (insert_seq, block_id, content_hash)
        self.priority_queue: List[Tuple[int, int]] = []
        self._next_seq: int = 0

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def evict(self) -> Tuple[int, int]:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        while self.priority_queue:
            insert_seq, block_id = heapq.heappop(self.priority_queue)
            meta = self.free_table.get(block_id)
            if meta is None:
                # 過期 entry，略過
                continue

            self.free_table.pop(block_id)
            return block_id, meta.content_hash

        raise ValueError("No usable cache memory left")

    def add(self, block_id: int, content_hash: int,
        num_hashed_tokens: int, last_accessed: float):
        meta = BlockMetaData(content_hash, num_hashed_tokens, last_accessed)
        self.free_table[block_id] = meta
        self._next_seq += 1
        heapq.heappush(self.priority_queue, (self._next_seq, block_id))

    def update(self, block_id: int, last_accessed: float):
        # FIFO 不會因 access 改變 eviction 順序，
        # 但我們還是更新 last_accessed 以便其他地方用（比如 debug / stats）
        meta = self.free_table.get(block_id)
        if meta is not None:
            meta.last_accessed = last_accessed

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor"
            )
        self.free_table.pop(block_id)
        # priority_queue 裡舊 entry 懶得清，evict 時會被過濾掉

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)
    
class WorkloadAwareEvictor(Evictor):
    """
    Mixed workload policy

    目標：
    - 同一 workload 內部維持類 LRU / LFU 行為
    - 不同 workload 根據整體 cache 使用貢獻調整保護強度
    """

    def __init__(self):
        self.free_table: Dict[int, BlockMetaData] = {}
        # per-workload 累積 access 次數
        self.workload_accesses: Dict[int, int] = {}
        # 追蹤目前最大 timestamp 作為 now_step
        self.current_time: float = 0.0

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)

    def evict(self) -> Tuple[int, int]:
        if not self.free_table:
            raise ValueError("No usable cache memory left")

        # 計算 per-workload penalty
        workload_penalty = self._compute_workload_penalty()

        victim_id: Optional[int] = None
        victim_score: float = float("-inf")

        for block_id, meta in self.free_table.items():
            score = self._eviction_score(meta, workload_penalty)
            if score > victim_score:
                victim_score = score
                victim_id = block_id

        if victim_id is None:
            raise ValueError("No usable cache memory left")

        meta = self.free_table.pop(victim_id)
        return victim_id, meta.content_hash

    def add(self, block_id: int, content_hash: int,
            num_hashed_tokens: int, last_accessed: float,
            workload_type: Optional[int] = None):
        self.current_time = max(self.current_time, last_accessed)
        meta = BlockMetaData(
            content_hash,
            num_hashed_tokens,
            last_accessed,
            workload_type=workload_type,
        )
        self.free_table[block_id] = meta
        self._record_access(meta)

    def update(self, block_id: int, last_accessed: float):
        meta = self.free_table.get(block_id)
        if meta is None:
            return
        self.current_time = max(self.current_time, last_accessed)
        meta.last_accessed = last_accessed
        meta.access_count += 1
        self._record_access(meta)

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor"
            )
        self.free_table.pop(block_id)

    # internal helpers

    def _record_access(self, meta: BlockMetaData):
        """統計 per-workload 使用強度"""
        if meta.workload_type is None:
            return
        self.workload_accesses[meta.workload_type] = \
            self.workload_accesses.get(meta.workload_type, 0) + 1

    def _compute_workload_penalty(self) -> Dict[int, float]:
        """hit 多的 workload 給較小 penalty"""
        if not self.workload_accesses:
            return {}

        max_access = max(self.workload_accesses.values())
        if max_access <= 0:
            max_access = 1

        penalty: Dict[int, float] = {}
        for w, cnt in self.workload_accesses.items():
            importance = cnt / max_access   # 0~1, 大代表常用
            penalty[w] = 1.0 - importance   # 0~1, 大代表容易被丟
        return penalty

    def _eviction_score(self,
                        meta: BlockMetaData,
                        workload_penalty: Dict[int, float]) -> float:
        """score 越大越優先被 evict"""

        # recency
        age = max(0.0, self.current_time - meta.last_accessed)

        # frequency
        inv_freq = 1.0 / (1.0 + meta.access_count)

        # prefix 長度保護（越長越不想丟）
        size_penalty = 1.0 / max(1, meta.num_hashed_tokens)

        # workload 層級 penalty
        wl_pen = 1.0
        if meta.workload_type is not None:
            wl_pen = workload_penalty.get(meta.workload_type, 1.0)

        # 係數可以調，這裡給一組合理起點
        return (
            0.5 * age +
            0.2 * inv_freq +
            0.1 * size_penalty +
            0.2 * wl_pen
        )


def make_evictor(eviction_policy: EvictionPolicy) -> Evictor:
    if eviction_policy == EvictionPolicy.LRU:
        return LRUEvictor()
    elif eviction_policy == EvictionPolicy.LFU:
        return LFUEvictor()
    elif eviction_policy == EvictionPolicy.FIFO:
        return FIFOEvictor()
    elif eviction_policy == EvictionPolicy.WORKLOAD_AWARE:
        return WorkloadAwareEvictor()
    else:
        raise ValueError(f"Unknown cache eviction policy: {eviction_policy}")
