from typing import Callable, Type
import pandas as pd

class CacheSim:
    """
    Replays a trace through a policy object that implements:
      request(key, size, ts, user=None, vid_idx=None) -> bool
      resize(new_cap)  (optional but used in dynamic mode)
    """
    def __init__(self, capacity_mb: int, policy_ctor: Callable):
        self.cap_bytes = capacity_mb * 1024 * 1024
        self.policy    = policy_ctor(self.cap_bytes)

    def replay(self, df: pd.DataFrame, key_func: Callable=None,
               ts_attr="ts") -> float:
        key_func = key_func or (lambda r: f"{r.video}_{r.ladder}")
        hits = 0
        it = df.itertuples(index=False)
        for row in it:
            key  = key_func(row)
            ts   = getattr(row, ts_attr)
            hit = self.policy.request(
                key, row.bytes, ts,
                getattr(row, "user", None),
                getattr(row, "vid_idx", None)
            )
            if hit:
                hits += 1
        return hits / len(df)

    # optional helper for dynamic resize simulations
    def resize(self, new_cap_mb):
        self.policy.resize(new_cap_mb * 1024 * 1024)