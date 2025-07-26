# Efficient Cache Utilization & Management PoC (Video Streaming)

> **TL;DR**  
> Modular cache simulator for video workloads using a Netflix viewing log.  
> Policies: **LRU, LFU, KDE-popularity, LSTM-prefetch (gated/budgeted)**, plus **dynamic cache resizing**.  
> Best static hit ratio: **33.0 % (LRU @ 1000 MB)**.  
> Dynamic resizing during a traffic surge improves hit ratio by **≈5 % relative** and slightly reduces latency.

---

## 1. Repository Layout

```
cache_poc/
├─ src/                 # simulator, policies, metrics, eval scripts
├─ models/              # kde.pkl, lstm.pt (pre-fit models)
├─ data/                # trace.parquet (processed Netflix trace)
├─ figs/                # result plots (generated)
├─ notebooks/           # exploration/training/plot notebooks
├─ results*.csv         # evaluation tables
└─ README.md
```

---

## 2. Implemented Policies

| Policy | Idea | Notes |
|--------|------|-------|
| **LRU** | Evict least-recently-used | Strong baseline on our trace |
| **LFU** | Evict least-frequently-used | Close to LRU here |
| **KDE-Cache** | Popularity score via Kernel Density Estimation over time; evict lowest score | scikit-learn `KernelDensity`, bw ≈ 1 day |
| **LSTM Prefetch** | Predict next video and selectively prefetch | Prefetch only if *p ≥ 0.25*, and cap prefetch bytes at 10 % of cache to avoid pollution |
| **Dynamic Resizing** | Resize cache between 200–1000 MB based on request-rate windows | Simple heuristic controller (+~5 % on surge set) |

> We experimented with chunk-level objects (splitting videos into segments) but reverted for the PoC because reuse was minimal in this small dataset. See `notebooks/06_chunkify.ipynb` for that experiment.

---

## 3. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -U pip wheel
pip install -r requirements.txt          # pandas, pyarrow, torch, sklearn, statsmodels, etc.
pip install -e cache_poc                 # optional but recommended (editable install)
```

Python ≥ 3.10 recommended. No special system deps.

---

## 4. Data

We used a Netflix viewing history export (personal data) as a small VoD trace. A similar exploration is on Kaggle:

- Kaggle reference: https://www.kaggle.com/code/kimyuri/netflix-watch-log-eda1-clickstream

### Processing Steps (already done)

1. Parse CSV → `data/trace.parquet` (`src/ingest_netflix.py` or equivalent).
2. Derive:
   - **bytes** watched = duration × (device bitrate / 8). Bitrate → ladder mapping: 480p/720p/1080p.
   - **vid_idx** (categorical index) for LSTM.
   - **cache key** = `video_ladder`.

If you need to rebuild:

```bash
python -m cache_poc.src.ingest_netflix   # if present
```

---

## 5. Models

- **KDE**: Fit once over timestamps → `models/kde.pkl`  
  ```bash
  python -m cache_poc.src.models.fit_kde
  ```

- **LSTM**: Trained in `notebooks/03_lstm_train.ipynb` (epochs, win, etc.). Saved to `models/lstm.pt`.

---

## 6. Run Evaluations

```bash
python -m cache_poc.src.evaluate
```

This prints tables and writes:

- `results.csv` – hit ratio vs cache size (100, 500, 1000 MB)
- `results_metrics.csv` – bytes saved %, avg latency
- `results_dynamic.csv` – static vs dynamic hit ratios on surge traffic
- `results_dynamic_metrics.csv` – same with bandwidth/latency

---

## 7. Plots

Use `notebooks/05_results.ipynb` to regenerate:

- `fig_hit_vs_size.png`
- `fig_bar_500.png`
- `fig_dynamic.png`
- `fig_bytes_saved_500.png`
- `fig_latency_500.png`
- (optional) `fig_dyn_metrics_*.png`

They are saved under `figs/`.

---

## 8. Key Results (Object-Level Trace)

| Policy | 100 MB | 500 MB | 1000 MB |
|--------|-------:|-------:|--------:|
| **LRU** | 0.1479 | **0.2805** | **0.3302** |
| LFU     | 0.1470 | 0.2795 | 0.3213 |
| KDE     | 0.1477 | 0.2773 | 0.3157 |
| LSTM    | 0.1478 | 0.2776 | 0.3161 |

**Dynamic (surge, 10× prime time):**  
- KDE: 0.1962 → **0.2054** (+4.7 % rel)  
- LSTM: 0.1963 → **0.2062** (+5.0 % rel)

Bytes served from cache ↑ (~17.46 % → ~18.70 %), latency ↓ (~41.56 ms → ~41.22 ms).

---

## 9. Future Work

- Larger traces (ISP/YouTube) & real segment-level keys  
- Size-aware admission (AdaptSize, LHD) & hybrid selectors (LeCaR)  
- Geo-distributed multi-POP simulation (parent/child caches)  
- QoE metrics (stall time, bitrate switches), CDN egress cost modeling  
- RL/PID controller for dynamic resizing

---

## 10. Known Issues

- Small personal dataset → low absolute hit ratios; LRU dominates  
- Bytes & latencies are synthesized; ABR integration is simplistic  
- Chunk-level attempt reduced hits due to lack of reuse (kept as an experiment)

---

## 11. License & Credits

- Code: MIT (edit as needed)  
- Data: Netflix viewing export (user-owned). Kaggle inspiration by **kimyuri** (link above).

If you publish or present, consider citing key caching papers (AdaptSize, LeCaR, etc.) and the Kaggle notebook.

---

## 12. Handy Commands

```bash
# Fit KDE
python -m cache_poc.src.models.fit_kde

# Evaluate all policies & dynamic mode
python -m cache_poc.src.evaluate

# Retrain LSTM (open notebook)
jupyter notebook notebooks/03_lstm_train.ipynb
```

Happy caching! 🚀
