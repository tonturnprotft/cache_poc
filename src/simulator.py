from typing import Type
import pandas as pd

class CacheSim:
    """
    Replays a trace DataFrame through any byte-aware cache policy.
    """
    def __init__(self, capacity_mb: int, policy_cls: Type):
        self.cap_bytes = capacity_mb * 1024 * 1024
        self.policy    = policy_cls(self.cap_bytes)

    # ----------------------------------------------------------
# src/simulator.py
    def replay(self, df, key_func=None, ts_attr="ts") -> float:
        key_func = key_func or (lambda r: f"{r.video}_{r.ladder}")
        hits = 0
        for row in df.itertuples(index=False):
            key = key_func(row)
            ts  = getattr(row, ts_attr)
            if self.policy.request(key, row.bytes, ts):
                hits += 1
        return hits / len(df)