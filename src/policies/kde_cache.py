# src/policies/kde_cache.py
import pandas as pd
from collections import OrderedDict
from statsmodels.nonparametric.kde import KDEUnivariate

class KDECache:
    """
    LRU storage + KDE score for eviction.
    Lower score = less likely to be requested 'now' → evict first.
    """
    def __init__(self, capacity_bytes:int, kde_model:KDEUnivariate):
        self.cap   = capacity_bytes
        self.cache = OrderedDict()          # key -> (size, score)
        self.size  = 0
        self.kde   = kde_model

    def _score(self, ts:pd.Timestamp) -> float:
        sec = ts.value // 1_000_000_000
        return float(self.kde.evaluate(sec))

    def request(self, key:str, obj_size:int, ts:pd.Timestamp) -> bool:
        hit = key in self.cache
        if hit:
            # refresh recency info
            size, score = self.cache.pop(key)
            self.cache[key] = (size, score)
            return True

        # MISS: insert after evicting lowest-score items
        score_now = self._score(ts)
        while self.size + obj_size > self.cap and self.cache:
            # find worst (min score); fall back to oldest if equal
            worst = min(self.cache.items(), key=lambda kv: kv[1][1])[0]
            sz, _ = self.cache.pop(worst)
            self.size -= sz

        if obj_size <= self.cap:
            self.cache[key] = (obj_size, score_now)
            self.size += obj_size
        return False