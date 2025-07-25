# src/policies/base.py
class BasePolicy:
    def __init__(self, capacity_bytes:int):
        self.cap = capacity_bytes
    def request(self, key:str, size:int, ts) -> bool: ...
    def resize(self, new_cap:int):  # used later
        self.cap = new_cap