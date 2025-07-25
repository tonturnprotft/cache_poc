# src/policies/lfu.py
import heapq
from collections import defaultdict

class LFU:
    """
    Least-Frequently-Used cache (byte-accurate).
    Uses a min-heap keyed by (freq, seq_no) to break ties by recency.
    """
    def __init__(self, capacity_bytes: int):
        self.capacity = capacity_bytes
        self.freq     = defaultdict(int)         # key -> hit count
        self.store    = {}                       # key -> size
        self.heap     = []                       # (freq, seq, key)
        self.seq      = 0                        # monotone timestamp
        self.size     = 0

    # ----------------------------------------------------------
    def request(self, key: str, obj_size: int) -> bool:
        self.seq += 1
        if key in self.store:                    # ------------- HIT
            self.freq[key] += 1
            heapq.heappush(self.heap, (self.freq[key], self.seq, key))
            return True

        # ------------- MISS
        # Evict until room
        while self.size + obj_size > self.capacity and self.heap:
            f, _, k = heapq.heappop(self.heap)
            # skip stale heap entries
            if self.freq[k] != f: continue
            self.size -= self.store.pop(k)
            self.freq.pop(k)

        # Insert if fits
        if obj_size <= self.capacity:
            self.store[key] = obj_size
            self.freq[key]  = 1
            heapq.heappush(self.heap, (1, self.seq, key))
            self.size += obj_size
        return False