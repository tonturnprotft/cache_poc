# src/metrics.py
from ..simulator import CacheSim

EDGE_LAT_MS = 5
MISS_LAT_MS = {"480p": 60, "720p": 40, "1080p": 20}  # tweak if you like

def replay_with_metrics(df, policy_ctor, cap_mb,key_func=lambda r: r.key):
    sim = CacheSim(cap_mb, policy_ctor)
    hits = reqs = 0
    bytes_hit = bytes_total = 0
    lat_sum = 0.0

    for row in df.itertuples(index=False):
        key = key_func(row)
        hit = sim.policy.request(key, row.bytes, row.ts,
                                 getattr(row, "user", None),
                                 getattr(row, "vid_idx", None))
        reqs += 1
        bytes_total += row.bytes
        if hit:
            hits += 1
            bytes_hit += row.bytes
            lat_sum += EDGE_LAT_MS
        else:
            lat_sum += MISS_LAT_MS[row.ladder]

    return {
        "hit_ratio": hits / reqs,
        "bytes_saved_pct": bytes_hit / bytes_total,
        "avg_latency_ms": lat_sum / reqs
    }