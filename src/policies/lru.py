# src/policies/lru.py
from collections import OrderedDict

class LRU:
    """
    Classic Least-Recently-Used cache with byte-accurate capacity.
    Stores {key: size_in_bytes}.
    """
    def __init__(self, capacity_bytes: int):
        self.capacity = capacity_bytes
        self.cache    = OrderedDict()   # key -> bytes
        self.size     = 0               # current bytes in cache

    # ----------------------------------------------------------
    def request(self, key: str, obj_size: int) -> bool:
        """
        Process one GET.
        Return True on hit, False on miss (after inserting).
        """
        hit = key in self.cache
        if hit:
            # move to MRU position
            self.cache.move_to_end(key)
        else:
            # evict until we have room
            while self.size + obj_size > self.capacity and self.cache:
                k, sz = self.cache.popitem(last=False)   # LRU item
                self.size -= sz
            # finally insert new item (unless it's bigger than the cache)
            if obj_size <= self.capacity:
                self.cache[key] = obj_size
                self.size += obj_size
        return hit