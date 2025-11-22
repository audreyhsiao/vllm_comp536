# vllm/sim/trace_simulator.py
from typing import Dict, List, Optional, Tuple
import json, hashlib

def _sha1_int(ids: List[int]) -> str:
    m = hashlib.sha1()
    m.update((",".join(map(str, ids))).encode())
    return m.hexdigest()

class TraceStore:
    """
    Audrey: 假設 JSONL trace 是以下格式：
    
      - "prompt_token_ids": [int, ...]
      - "response_token_ids": [int, ...]
    """
    def __init__(self, path: str):
        self._resp_by_key: Dict[str, List[int]] = {}
        with open(path, "r") as f:
            for line in f:
                rec = json.loads(line)
                p_ids = rec["prompt_token_ids"]
                r_ids = rec["response_token_ids"]
                self._resp_by_key[_sha1_int(p_ids)] = r_ids

    def lookup(self, prompt_ids: List[int]) -> Optional[List[int]]:
        return self._resp_by_key.get(_sha1_int(prompt_ids))

class TraceCursor:
    def __init__(self, resp_ids: List[int]):
        self.resp = resp_ids
        self.pos = 0

    def next_token(self) -> Optional[int]:
        if self.pos >= len(self.resp):
            return None
        tok = self.resp[self.pos]
        self.pos += 1
        return tok
