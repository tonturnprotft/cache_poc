import pandas as pd
import torch
import joblib
from pathlib import Path

from .simulator import CacheSim
from .policies.lru import LRU
from .policies.lfu import LFU
from .policies.kde_cache import KDECache
from .policies.prefetch_wrap import PrefetchWrapper
from .models.lstm_model import NextVidLSTM
from .policies.metrics import replay_with_metrics, EDGE_LAT_MS, MISS_LAT_MS

# ---- Paths ----
ROOT = Path(__file__).resolve().parents[1]
TRACE_PATH = ROOT / "data" / "trace.parquet"
KDE_PATH   = ROOT / "models" / "kde.pkl"
LSTM_PATH  = ROOT / "models" / "lstm.pt"

# ---- Load trace ----
df = pd.read_parquet(TRACE_PATH).sort_values("ts")

# unified key function (works for chunked or original trace)
KEY_FUNC = lambda r: f"{r.video}_{r.ladder}" 

# ensure vid_idx exists (needed by prefetch wrapper)
if "vid_idx" not in df.columns:
    cats = df.video.astype("category")
    df["vid_idx"] = cats.cat.codes

# ---- Load models ----
kde_pack  = joblib.load(KDE_PATH)
kde_model = kde_pack["kde"]

state  = torch.load(LSTM_PATH, map_location="cpu")
model  = NextVidLSTM(len(state["vid2idx"]))
model.load_state_dict(state["state"])
vid2idx, idx2vid = state["vid2idx"], state["idx2vid"]

# ---- Policy factories ----
base_kde   = lambda cap: KDECache(cap, kde_model)
prefetch_p = lambda cap: PrefetchWrapper(base_kde(cap), model, idx2vid, topN=3)

# --- metrics -----

met_rows = []
for cap in [100, 500, 1000]:
    for name, ctor in [("LRU", LRU), ("LFU", LFU),
                       ("KDE", base_kde), ("LSTM", prefetch_p)]:
        m = replay_with_metrics(df, ctor, cap, key_func=KEY_FUNC)
        met_rows.append((name, cap, m["hit_ratio"],
                         m["bytes_saved_pct"], m["avg_latency_ms"]))

met = pd.DataFrame(met_rows,
                   columns=["policy","cap_MB","hit_ratio",
                            "bytes_saved_pct","avg_latency_ms"])
met.to_csv(ROOT/"results_metrics.csv", index=False)
print("\nBandwidth & latency metrics:")
print(met)

# ---- Run evaluation ----
sizes = [100, 500, 1000]  # MB
rows  = []
for cap in sizes:
    rows.append(("LRU",   cap, CacheSim(cap, LRU).replay(df, key_func=KEY_FUNC)))
    rows.append(("LFU",   cap, CacheSim(cap, LFU).replay(df, key_func=KEY_FUNC)))
    rows.append(("KDE",   cap, CacheSim(cap, base_kde).replay(df, key_func=KEY_FUNC)))
    rows.append(("LSTM",  cap, CacheSim(cap, prefetch_p).replay(df, key_func=KEY_FUNC)))

res = pd.DataFrame(rows, columns=["policy", "cap_MB", "hit_ratio"])
res.to_csv(ROOT / "results.csv", index=False)
print(res)
# --- Dynamic resize & surge test ---
def build_surge(df, factor=10):
    prime = df[df.ts.dt.hour.between(19, 23)]
    surge = pd.concat([prime]*factor, ignore_index=True)
    surge['ts'] = pd.to_datetime(surge['ts'].values)  # keep dtype
    surge.reset_index(drop=True, inplace=True)
    return surge

def dynamic_replay(df_in, base_ctor, cap_low=200, cap_high=1000,
                   window=2000, hi_req=1500, lo_req=400,
                   never_shrink_when_high=True):
    sim = CacheSim(cap_low, base_ctor)
    hits = reqs = 0
    buf  = 0     # requests in current window
    high_mode = False
    for i, row in enumerate(df_in.itertuples(index=False)):
        reqs += 1
        buf  += 1

        # window boundary
        if (i+1) % window == 0:
            if buf > hi_req and (not high_mode):
                sim.resize(cap_high)
                high_mode = True
            elif buf < lo_req and (not never_shrink_when_high or not high_mode):
                sim.resize(cap_low)
                high_mode = False
            buf = 0

        key = KEY_FUNC(row)
        if sim.policy.request(key, row.bytes, row.ts,
                              getattr(row,"user",None),
                              getattr(row,"vid_idx",None)):
            hits += 1
    return hits/reqs

surge_df = build_surge(df, factor=10)
rows_dyn = []
for name, ctor in [("KDE", base_kde), ("LSTM", prefetch_p)]:
    static_hr = CacheSim(500, ctor).replay(surge_df, key_func=KEY_FUNC)
    dyn_hr    = dynamic_replay(surge_df, ctor, cap_low=200, cap_high=1000)
    rows_dyn.append((name, static_hr, dyn_hr))

dyn_res = pd.DataFrame(rows_dyn, columns=["policy", "static_500MB", "dynamic_200_1000MB"])
print("\nDynamic resizing on surge traffic:")
print(dyn_res)
dyn_res.to_csv(ROOT/"results_dynamic.csv", index=False)
def dynamic_metrics(df_in, base_ctor):
    sim = CacheSim(200, base_ctor)  # start low
    hits=reqs=0
    bytes_hit=bytes_total=0
    lat_sum=0.0
    buf=0
    window=2000; hi_req=1500; lo_req=400
    high_mode=False

    for i, row in enumerate(df_in.itertuples(index=False)):
        reqs += 1
        buf  += 1
        bytes_total += row.bytes

        # window boundary
        if (i+1) % window == 0:
            if buf > hi_req and not high_mode:
                sim.resize(1000)   # MB
                high_mode = True
            elif buf < lo_req and not high_mode:
                sim.resize(200)    # MB
            buf = 0

        key = KEY_FUNC(row)
        hit = sim.policy.request(key, row.bytes, row.ts,
                                 getattr(row,"user",None),
                                 getattr(row,"vid_idx",None))
        if hit:
            hits += 1
            bytes_hit += row.bytes
            lat_sum += EDGE_LAT_MS
        else:
            lat_sum += MISS_LAT_MS[row.ladder]

    return {
        "hit_ratio": hits/reqs,
        "bytes_saved_pct": bytes_hit/bytes_total,
        "avg_latency_ms": lat_sum/reqs
    }

dyn_met_rows=[]
for name, ctor in [("KDE", base_kde), ("LSTM", prefetch_p)]:
    static_m = replay_with_metrics(surge_df, ctor, 500, key_func=KEY_FUNC)
    dyn_m    = dynamic_metrics(surge_df, ctor)
    dyn_met_rows.append((name,
                         static_m["hit_ratio"], static_m["bytes_saved_pct"], static_m["avg_latency_ms"],
                         dyn_m["hit_ratio"],    dyn_m["bytes_saved_pct"],    dyn_m["avg_latency_ms"]))
dyn_met = pd.DataFrame(dyn_met_rows,
    columns=["policy",
             "static_hit","static_bytes_pct","static_latency_ms",
             "dyn_hit","dyn_bytes_pct","dyn_latency_ms"])
dyn_met.to_csv(ROOT/"results_dynamic_metrics.csv", index=False)
print("\nDynamic metrics (surge):")
print(dyn_met)