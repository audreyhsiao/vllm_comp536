"""
v0 Online Prefix‑Sharing Simulator (CPU, no real LLM)
- Online server using FastAPI (optional). If you only need the core simulator,
  you can ignore FastAPI part at the bottom.
- Keeps vLLM "v0" style behavior: prefill first, decode one token per step,
  continuous batching, simple oldest‑first fairness, paged KV with prefix sharing.
- No GPU, no model. Time is simulated by a CostModel.

Run (server):
  pip install fastapi uvicorn pydantic
  uvicorn sim_v0_online:app --reload --port 8000

Try:
  curl -X POST http://localhost:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"Hello","max_tokens":32,"stream":false}'

  # streaming (Server-Sent Events)
  curl -N -X POST http://localhost:8000/v1/completions?stream=true \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"Hello","max_tokens":8,"prefix_key":"hello"}'
"""
from __future__ import annotations
import asyncio
import hashlib
import json
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple, AsyncGenerator
from collections import deque, OrderedDict

# =========================
# Data Models
# =========================
@dataclass
class Request:
    req_id: str
    arrival_ms: int
    prompt_len: int
    gen_len: int
    prefix_key: Optional[str] = None
    stream_q: asyncio.Queue = field(default_factory=asyncio.Queue)  # emits token strings

@dataclass
class Sequence:
    req: Request
    consumed_prompt: int = 0
    produced: int = 0
    state: str = "PREFILL"  # PREFILL -> DECODE -> FINISHED

    def need_prefill(self) -> int:
        return max(0, self.req.prompt_len - self.consumed_prompt)

    def need_decode(self) -> int:
        return max(0, self.req.gen_len - self.produced)

# =========================
# KV Cache with Prefix Sharing (paged)
# =========================
class KVBlockManager:
    def __init__(self, block_size: int, num_blocks: int):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.free_blocks = list(range(num_blocks))
        self.refcnt: Dict[int, int] = {i: 0 for i in range(num_blocks)}
        self.lru = OrderedDict()  # only refcnt==0 blocks live here
        self.prefix_map: Dict[str, List[int]] = {}  # prefix_key -> blocks
        self.seq_blocks: Dict[str, List[int]] = {}  # req_id -> blocks
        self.evictions = 0

    def _pop_free(self) -> Optional[int]:
        if self.free_blocks:
            return self.free_blocks.pop()
        if self.lru:
            bid, _ = self.lru.popitem(last=False)
            self.evictions += 1
            return bid
        return None

    def _touch_lru(self, bid: int):
        if self.refcnt[bid] == 0:
            if bid in self.lru:
                self.lru.pop(bid, None)
            self.lru[bid] = True

    def _hold(self, bid: int):
        self.refcnt[bid] += 1
        if bid in self.lru:
            self.lru.pop(bid, None)

    def _release(self, bid: int):
        assert self.refcnt[bid] > 0
        self.refcnt[bid] -= 1
        if self.refcnt[bid] == 0:
            self._touch_lru(bid)

    def share_prefix(self, seq: Sequence):
        rid = seq.req.req_id
        blocks = self.seq_blocks.setdefault(rid, [])
        k = seq.req.prefix_key
        if k and k in self.prefix_map:
            for bid in self.prefix_map[k]:
                self._hold(bid)
                blocks.append(bid)

    def materialize_prefix(self, seq: Sequence, tokens_to_materialize: int) -> bool:
        if tokens_to_materialize <= 0:
            return True
        need_blocks = math.ceil(tokens_to_materialize / self.block_size)
        rid = seq.req.req_id
        blocks = self.seq_blocks.setdefault(rid, [])
        for _ in range(need_blocks):
            bid = self._pop_free()
            if bid is None:
                return False
            self._hold(bid)
            blocks.append(bid)
        return True

    def register_prefix_template(self, prefix_key: str, prompt_len: int) -> bool:
        if prefix_key in self.prefix_map:
            return True
        needed = math.ceil(prompt_len / self.block_size)
        tpl: List[int] = []
        for _ in range(needed):
            bid = self._pop_free()
            if bid is None:
                for b in tpl:
                    self._release(b)
                return False
            self._hold(bid)
            tpl.append(bid)
        self.prefix_map[prefix_key] = tpl
        return True

    def ensure_decode_capacity(self, batch_size: int) -> bool:
        free_like = len(self.free_blocks) + len(self.lru)
        return free_like >= batch_size

    def append_decode_token(self, seq: Sequence):
        rid = seq.req.req_id
        blocks = self.seq_blocks.setdefault(rid, [])
        total_tokens = seq.consumed_prompt + seq.produced
        at_boundary = (total_tokens % self.block_size == 0)
        if at_boundary:
            bid = self._pop_free()
            if bid is None:
                raise RuntimeError("Out of KV blocks in DECODE")
            self._hold(bid)
            blocks.append(bid)

    def release_sequence(self, seq: Sequence):
        rid = seq.req.req_id
        blocks = self.seq_blocks.pop(rid, [])
        for b in blocks:
            self._release(b)

# =========================
# Cost Model (no inference)
# =========================
class CostModel:
    def prefill_ms(self, tokens_sum: int, batch_size: int) -> int:
        raise NotImplementedError
    def decode_ms(self, batch_size: int) -> int:
        raise NotImplementedError

class ReplayLikeCost(CostModel):
    def __init__(self, prefill_ms_per_tok=0.7, decode_ms_base=1.0, decode_ms_per_seq=0.5):
        self.pt = prefill_ms_per_tok
        self.db = decode_ms_base
        self.ds = decode_ms_per_seq
    def prefill_ms(self, tokens_sum: int, batch_size: int) -> int:
        return int(math.ceil(tokens_sum * self.pt))
    def decode_ms(self, batch_size: int) -> int:
        return int(math.ceil(self.db + self.ds * batch_size))

# =========================
# Simulator (v0 semantics, ONLINE)
# =========================
class SimConfig:
    def __init__(self,
                 block_size: int = 16,
                 num_blocks: int = 200000,
                 max_prefill_tokens: int = 8192,
                 max_decode_batch: int = 32,
                 scheduler_version: str = "v0"):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.max_prefill_tokens = max_prefill_tokens
        self.max_decode_batch = max_decode_batch
        self.scheduler_version = scheduler_version  # keep for future; using v0 now

class PrefixSharingSimulatorV0:
    """Online simulator: accepts live requests; schedules steps; streams tokens.
    v0 behavior: prefill-first, decode one-token-per-step, continuous batching.
    """
    def __init__(self, cfg: SimConfig, cost: CostModel):
        self.cfg = cfg
        self.kv = KVBlockManager(cfg.block_size, cfg.num_blocks)
        self.cost = cost

        self.time_ms = 0
        self._epoch_start = time.time()  # wall-clock anchor

        # online queues/state
        self.waiting: Deque[Request] = deque()
        self.active: List[Sequence] = []
        self.completed: Dict[str, Dict] = {}

        # async control
        self._cv = asyncio.Condition()
        self._run_task: Optional[asyncio.Task] = None
        self._closing = False

    # ---- helpers ----
    def _now_ms(self) -> int:
        return int((time.time() - self._epoch_start) * 1000)

    async def start(self):
        if self._run_task is None:
            self._run_task = asyncio.create_task(self._driver())

    async def close(self):
        self._closing = True
        if self._run_task:
            await self._run_task

    # ---- public API ----
    async def submit(self, prompt: str, max_tokens: int, prefix_key: Optional[str], stream: bool = True) -> str:
        req_id = str(uuid.uuid4())
        prompt_len = len(prompt.split()) if prompt else 0  # toy tokenization
        r = Request(req_id=req_id,
                    arrival_ms=self._now_ms(),
                    prompt_len=prompt_len,
                    gen_len=max_tokens,
                    prefix_key=prefix_key)
        async with self._cv:
            # create template if first time
            if r.prefix_key and r.prefix_key not in self.kv.prefix_map:
                ok = self.kv.register_prefix_template(r.prefix_key, r.prompt_len)
                if not ok:
                    # backpressure: reject if no KV
                    raise RuntimeError("No KV capacity to register prefix template")
            self.waiting.append(r)
            self._cv.notify_all()
        return req_id

    async def stream(self, req_id: str) -> AsyncGenerator[str, None]:
        # Find request
        target: Optional[Request] = None
        # check waiting
        for r in list(self.waiting):
            if r.req_id == req_id:
                target = r
                break
        # check active
        if target is None:
            for s in self.active:
                if s.req.req_id == req_id:
                    target = s.req
                    break
        # check completed
        if target is None:
            # might be completed already
            for rid, m in self.completed.items():
                if rid == req_id:
                    # emit nothing; already finished
                    return
        # Wait and forward stream events
        if target is None:
            # not seen yet; wait a bit until admitted
            await asyncio.sleep(0.01)
        # now read from stream_q
        rmap = {r.req_id: r for r in list(self.waiting)}
        for s in self.active:
            rmap[s.req.req_id] = s.req
        target = rmap.get(req_id, target)
        if target is None:
            # might have finished 
            return
        q = target.stream_q
        while True:
            tok = await q.get()
            if tok is None:
                break
            yield tok

    # ---- driver (event loop) ----
    async def _driver(self):
        while not self._closing:
            async with self._cv:
                # admit waiting into active
                while self.waiting:
                    r = self.waiting.popleft()
                    seq = Sequence(r)
                    self.kv.share_prefix(seq)
                    self.active.append(seq)
                # choose a step (v0 policy)
                prefill_cands = [s for s in self.active if s.need_prefill() > 0]
                decode_cands = [s for s in self.active if s.need_prefill() == 0 and s.need_decode() > 0]

                if prefill_cands:
                    # Greedy pack by tokens until budget
                    budget = self.cfg.max_prefill_tokens
                    batch: List[Tuple[Sequence, int]] = []
                    used = 0
                    for s in prefill_cands:
                        need = s.need_prefill()
                        if need <= 0:
                            continue
                        take = min(need, max(1, budget - used))
                        if take <= 0:
                            break
                        # reserve KV for uncovered tokens
                        if not self.kv.materialize_prefix(s, take):
                            # cannot allocate now; skip this seq in this round
                            continue
                        batch.append((s, take))
                        used += take
                        if used >= budget:
                            break
                    if batch:
                        dur = self.cost.prefill_ms(tokens_sum=used, batch_size=len(batch)) / 1000.0
                        # simulate compute time
                        await asyncio.sleep(dur)
                        for s, take in batch:
                            s.consumed_prompt += take
                            if s.need_prefill() == 0 and s.need_decode() == 0:
                                s.state = "FINISHED"
                        self._collect_finishes()
                        continue  # go next loop

                if decode_cands:
                    bsz = min(len(decode_cands), self.cfg.max_decode_batch)
                    if not self.kv.ensure_decode_capacity(bsz):
                        # back off a bit if KV tight
                        await asyncio.sleep(0.001)
                        continue
                    batch = decode_cands[:bsz]
                    dur = self.cost.decode_ms(bsz) / 1000.0
                    await asyncio.sleep(dur)
                    for s in batch:
                        # append one token
                        self.kv.append_decode_token(s)
                        s.produced += 1
                        # emit a fake token to client stream
                        fake_token = "▁"  # placeholder token
                        try:
                            s.req.stream_q.put_nowait(fake_token)
                        except asyncio.QueueFull:
                            pass
                        if s.need_decode() == 0 and s.need_prefill() == 0:
                            s.state = "FINISHED"
                    self._collect_finishes()
                    continue

                # idle: wait for new work
                await self._cv.wait()

    def _collect_finishes(self):
        keep: List[Sequence] = []
        for s in self.active:
            if s.state == "FINISHED":
                self.completed[s.req.req_id] = {
                    "latency_ms": self._now_ms() - s.req.arrival_ms,
                    "prompt": s.req.prompt_len,
                    "gen": s.req.gen_len,
                }
                # close the stream
                try:
                    s.req.stream_q.put_nowait(None)
                except asyncio.QueueFull:
                    pass
                self.kv.release_sequence(s)
            else:
                keep.append(s)
        self.active = keep

    # -------- metrics
    def report(self) -> Dict:
        return {
            "finished": len(self.completed),
            "kv_evictions": self.kv.evictions,
            "kv_templates": {k: len(v) for k, v in self.kv.prefix_map.items()},
        }

# =========================
# Optional: FastAPI server for ONLINE usage
# =========================
try:
    from fastapi import FastAPI, Request as FastRequest
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel

    app = FastAPI(title="v0 Prefix‑Sharing Simulator")
    _cfg = SimConfig()
    _sim = PrefixSharingSimulatorV0(_cfg, ReplayLikeCost())

    @app.on_event("startup")
    async def _startup():
        await _sim.start()

    class CompBody(BaseModel):
        prompt: str
        max_tokens: int = 32
        prefix_key: Optional[str] = None
        stream: bool = False

    @app.post("/v1/completions")
    async def completions(body: CompBody, request: FastRequest):
        # default prefix_key = hash of prompt if not provided
        pk = body.prefix_key or hashlib.sha1(body.prompt.encode()).hexdigest()[:16]
        rid = await _sim.submit(prompt=body.prompt, max_tokens=body.max_tokens, prefix_key=pk, stream=body.stream)
        if body.stream:
            async def sse_gen():
                async for tok in _sim.stream(rid):
                    yield f"data: {json.dumps({'id': rid, 'token': tok})}\n\n"
                yield f"data: {json.dumps({'id': rid, 'event':'done'})}\n\n"
            return StreamingResponse(sse_gen(), media_type="text/event-stream")
        else:
            # non-stream: accumulate tokens until done
            out = []
            async for tok in _sim.stream(rid):
                out.append(tok)
            return JSONResponse({
                "id": rid,
                "text": ''.join(out),
                "metrics": _sim.completed.get(rid, {})
            })

    @app.get("/report")
    async def report():
        return JSONResponse(_sim.report())
except Exception:
    # FastAPI not installed; simulator core still usable
    app = None
