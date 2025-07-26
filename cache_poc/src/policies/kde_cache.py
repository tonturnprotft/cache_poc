import numpy as np
from collections import OrderedDict

class KDECache:
    """
    LRU-like storage but evicts the lowest KDE score first.
    Expects a scikit-learn KernelDensity model in self.kde.
    cache[key] = (size_bytes, score)
    """
    def __init__(self, capacity_bytes, kde_model):
        self.cap   = capacity_bytes
        self.kde   = kde_model
        self.cache = OrderedDict()
        self.size  = 0

    def _score(self, ts):
        sec = ts.value // 1_000_000_000
        return float(np.exp(self.kde.score_samples([[sec]])))

    def request(self, key, obj_size, ts, *_, **__):
        hit = key in self.cache
        if hit:
            # refresh recency marker without changing score
            size, score = self.cache.pop(key)
            self.cache[key] = (size, score)
            return True

        score_now = self._score(ts)

        # Evict until there is space
        while self.size + obj_size > self.cap and self.cache:
            worst = min(self.cache.items(), key=lambda kv: kv[1][1])[0]
            sz, _ = self.cache.pop(worst)
            self.size -= sz

        if obj_size <= self.cap:
            self.cache[key] = (obj_size, score_now)
            self.size      += obj_size
        return False

    # For dynamic resizing later
    def resize(self, new_cap):
        self.cap = new_cap
        # If we suddenly shrink, evict lowest scores until under cap
        while self.size > self.cap and self.cache:
            worst = min(self.cache.items(), key=lambda kv: kv[1][1])[0]
            sz, _ = self.cache.pop(worst)
            self.size -= sz