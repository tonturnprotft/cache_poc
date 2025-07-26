import pandas as pd, pathlib, re, tqdm

SRC  = pathlib.Path("data/raw/All_ViewingActivity.csv")
DST  = pathlib.Path("data/trace.parquet")

# ❶ device → (bitrate bits/s, ladder tag)
DEVICE_BR = {
    "Device Type 0": (2_000_000, "480p"),   # phones
    "Device Type 1": (3_000_000, "720p"),   # tablets
    "Device Type 2": (5_000_000, "1080p"),  # laptops / TVs
    # fallback for any un-mapped code
    "_default"     : (3_000_000, "720p"),
}

def hms_to_sec(hms):
    h, m, s = map(int, hms.split(":"))
    return h*3600 + m*60 + s

print("Loading…")
df = pd.read_csv(SRC, parse_dates=["Start Time"])

print("Deriving bytes & ladder…")
br_list, lad_list = [], []
for dev in df["Device Type"]:
    br, lad = DEVICE_BR.get(dev, DEVICE_BR["_default"])
    br_list.append(br)
    lad_list.append(lad)

df["bytes"]   = (df["Duration"]
                   .map(hms_to_sec)        # seconds watched
                   .mul(br_list)           # bits
                   .floordiv(8))           # → bytes
df["ladder"]  = lad_list
df.rename(columns={
    "Start Time"  : "ts",
    "Profile Name": "user",
    "Title"       : "video"
}, inplace=True)

keep = df[["ts", "user", "video", "bytes", "ladder"]]
keep.to_parquet(DST, compression="zstd")
print("Wrote", DST, "rows:", len(keep))