import math

class ReplayLikeCost:
    def __init__(self, prefill_ms_per_tok=0.7, decode_ms_base=1.0, decode_ms_per_seq=0.5):
        self.pt = prefill_ms_per_tok
        self.db = decode_ms_base
        self.ds = decode_ms_per_seq

    def prefill_ms(self, tokens_sum: int, batch_size: int) -> int:
        return int(math.ceil(tokens_sum * self.pt))

    def decode_ms(self, batch_size: int) -> int:
        return int(math.ceil(self.db + self.ds * batch_size))
