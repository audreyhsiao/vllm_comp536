# vllm/sim/simulator_backend.py
import asyncio
import hashlib
from typing import Optional, Dict, Any, List

# 直接重用你 Canvas 的核心類
from sim_v0_online import PrefixSharingSimulatorV0, SimConfig, ReplayLikeCost

class SimulatorBackend:
    """
    A thin wrapper that adapts the v0 online simulator to a vLLM-like backend.
    - add_request(): submit a request with an external request_id
    - step(): drain newly produced streaming events
    - finalize(): shutdown
    Events pushed to the engine look like: {"id": <external_id>, "token": <str>, "finished": bool}
    """
    def __init__(self,
                 block_size: int,
                 num_blocks: int,
                 max_prefill_tokens: int,
                 max_decode_batch: int,
                 prefill_ms_per_tok: float = 0.7,
                 decode_ms_base: float = 1.0,
                 decode_ms_per_seq: float = 0.5):
        cfg = SimConfig(
            block_size=block_size,
            num_blocks=num_blocks,
            max_prefill_tokens=max_prefill_tokens,
            max_decode_batch=max_decode_batch,
            scheduler_version="v0",
        )
        cost = ReplayLikeCost(prefill_ms_per_tok, decode_ms_base, decode_ms_per_seq)
        self._sim = PrefixSharingSimulatorV0(cfg, cost)

        # 外部 req_id <-> 內部 rid 的對應
        self._ext2int: Dict[str, str] = {}
        self._int2ext: Dict[str, str] = {}

        # 往 LLMEngine 回報的事件佇列
        self._events: asyncio.Queue = asyncio.Queue()
        self._forwarders: Dict[str, asyncio.Task] = {}

        self._started = False
        self._closed = False

    async def _ensure_started(self):
        if not self._started:
            await self._sim.start()
            self._started = True

    async def add_request(self,
                          prompt: str,
                          max_tokens: int,
                          prefix_key: Optional[str],
                          stream: bool,
                          request_id: str) -> str:
        """
        Submit a request with a *caller-provided* request_id (external id).
        Returns the same request_id for convenience.
        """
        await self._ensure_started()

        # 預設用 prompt hash 當 prefix key（若未提供）
        pk = prefix_key or hashlib.sha1((prompt or "").encode()).hexdigest()[:16]

        # 送進模擬器，獲得內部 rid
        internal_rid = await self._sim.submit(prompt=prompt, max_tokens=max_tokens, prefix_key=pk, stream=stream)

        # 綁定 ext<->int
        self._ext2int[request_id] = internal_rid
        self._int2ext[internal_rid] = request_id

        # 啟動 forwarder：把該請求的 token 串流轉成後端事件
        async def _forward():
            try:
                async for tok in self._sim.stream(internal_rid):
                    await self._events.put({"id": request_id, "token": tok, "finished": False})
                # 結束訊號
                await self._events.put({"id": request_id, "finished": True})
            finally:
                self._forwarders.pop(internal_rid, None)

        self._forwarders[internal_rid] = asyncio.create_task(_forward())
        return request_id

    async def step(self, max_events: int = 1024) -> List[Dict[str, Any]]:
        """
        Drain up to max_events streaming events produced since last call.
        NOTE:
        - 真正的「時間前進」已由 PrefixSharingSimulatorV0 內部 driver 控制（cost model + sleep）。
        - 這裡只負責把已產生的事件取出回傳給引擎。
        """
        out: List[Dict[str, Any]] = []
        try:
            while len(out) < max_events:
                out.append(self._events.get_nowait())
        except asyncio.QueueEmpty:
            pass
        return out

    async def finalize(self):
        if self._closed:
            return
        self._closed = True
        # 停掉所有 forwarder
        for t in list(self._forwarders.values()):
            t.cancel()
        self._forwarders.clear()
        # 關閉模擬器
        await self._sim.close()

    # （可選）提供報表讓上層查詢
    def report(self) -> Dict[str, Any]:
        return self._sim.report()
