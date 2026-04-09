# Phase 1 — Foundations & Polish

> Goal: take a stock IBM-capstone notebook dump and turn it into a
> reproducible, deployable, recruiter-presentable project — without
> changing the data science yet.

This document records every change made in Phase 1, why it was made,
and what it unlocks for later phases.

---

## TL;DR

| Area | Before | After |
|---|---|---|
| Reproducibility | Unpinned `dash`, `pandas`, `plotly` | All deps pinned to exact versions |
| Cold start | Fetched 12 KB CSV from IBM S3 on every boot | Dataset bundled locally + `lru_cache` |
| UI | Default white Dash, single page, 2 charts | Cyborg dark theme, KPI cards, 3 tabs |
| Repo layout | 7 notebooks in root with `(1)` in filenames | Numbered `01_…07_` inside `notebooks/` |
| Deploy | Render only, no container | Dockerfile, gunicorn-ready, `PORT` env, healthcheck |
| Hygiene | No README, no .gitignore, no .dockerignore | All present |
| Data integrity | Silently overwrote CSV's `Outcome` column | Renamed computed col to `LandingResult` |

---

## 1. Dependency pinning

**File:** `requirements.txt`

**Before**
```
dash
pandas
plotly
gunicorn
```

**After**
```
dash==2.17.1
dash-bootstrap-components==1.6.0
pandas==2.2.2
plotly==5.22.0
gunicorn==22.0.0
```

**Why:** unpinned dependencies are the #1 cause of "works on my machine"
failures. Recruiters and CI systems both care. Adding
`dash-bootstrap-components` unlocks the entire Bootstrap component
library + themes — needed for the new UI.

---

## 2. Local dataset + caching

**Files:** `data/spacex_launch_data.csv` (new), `app.py`

**Before:** `app.py` did `pd.read_csv("https://cf-courses-data.s3...")`
on every cold start.

**After:**
```python
DATA_PATH = Path(__file__).parent / "data" / "spacex_launch_data.csv"

@lru_cache(maxsize=1)
def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.read_csv(REMOTE_URL)  # graceful fallback
    df["LandingResult"] = df["Class"].map({1: "Success", 0: "Failure"})
    return df
```

**Why**
- Cold starts on free tiers (Render, Fly, HF Spaces) are painful enough
  without an extra network round-trip.
- The IBM S3 URL is not under our control — if it rotates, the app dies.
- `lru_cache` means the CSV is read once per process, not per request.
- Remote fallback keeps the code functional even if `data/` is missing.

---

## 3. Dashboard rewrite (`app.py`)

The single most visible change. The original was the stock IBM lab
template — same as every Coursera student's submission.

### What's new

**Theme & layout**
- Switched to `dash-bootstrap-components` with the **Cyborg** dark theme.
- Bootstrap Icons for KPI card glyphs.
- Container-based responsive layout.

**KPI strip** (recomputed reactively)
- Total launches
- Success rate (%)
- Heaviest payload (kg)
- Top booster (most successful version)

**Three analytical tabs**
1. **Success by site** — donut chart. Aggregates successful launches by
   `LaunchSite` for the "All sites" view, success/failure split for a
   single site.
2. **Payload vs outcome** — scatter colored by `BoosterVersion`, bubble
   sized by payload, hover-rich (`LaunchSite`, `Orbit`).
3. **Outcome by orbit** — bar of success rate per orbit type, sorted
   descending, color-encoded by rate.

**Engineering improvements**
- Type hints throughout.
- Module docstring + inline section headers.
- `logging` instead of print.
- `_filter()` helper — single source of truth for the site + payload
  filter, eliminates the duplicated logic the IBM template had.
- `PORT` environment variable so the same code runs on Render, HF
  Spaces, Fly.io, Cloud Run.
- `host="0.0.0.0"` so the container is reachable.

### Bug caught during smoke test

The IBM CSV already contains an `Outcome` column (raw landing strings
like `True ASDS`, `None None`). My first rewrite did
`df["Outcome"] = df["Class"].map(...)`, silently destroying the original
data. Renamed the computed column to `LandingResult` so the raw column
is preserved for future use (e.g., distinguishing landing-pad vs
drone-ship recoveries).

---

## 4. Repository structure

**Before** (everything in root, lowercase, parens, spaces):
```
SpaceX_Machine Learning Prediction_Part_5.ipynb
edadataviz.ipynb
jupyter-labs-eda-sql-coursera_sqllite.ipynb
jupyter-labs-spacex-data-collection-api (1).ipynb
jupyter-labs-webscraping (1).ipynb
lab_jupyter_launch_site_location (1).ipynb
labs-jupyter-spacex-Data wrangling.ipynb
app.py
requirements.txt
ds-capstone-template-coursera.pdf
```

**After**
```
.
├── app.py
├── requirements.txt
├── README.md
├── Dockerfile
├── .dockerignore
├── .gitignore
├── data/
│   └── spacex_launch_data.csv
├── notebooks/
│   ├── 01_data_collection_api.ipynb
│   ├── 02_data_collection_webscrape.ipynb
│   ├── 03_data_wrangling.ipynb
│   ├── 04_eda_sql.ipynb
│   ├── 05_eda_viz.ipynb
│   ├── 06_launch_site_map.ipynb
│   └── 07_ml_prediction.ipynb
├── assets/                  # Dash auto-loads CSS/images here
├── docs/
│   └── PHASE_1.md           # this file
└── ds-capstone-template-coursera.pdf
```

**Why:** notebook ordering now mirrors the data-science lifecycle —
collection → wrangling → EDA → modeling. A recruiter can read the names
top-to-bottom and immediately understand the workflow.

---

## 5. Containerization

**File:** `Dockerfile` (new)

```dockerfile
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 PORT=8050
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY data/ ./data/
EXPOSE 8050
HEALTHCHECK ...
CMD ["gunicorn", "app:server", "-b", "0.0.0.0:8050", "--workers", "2", "--timeout", "60"]
```

**Why**
- Layer-caches deps separately from code → fast rebuilds.
- `python:3.11-slim` keeps the image small.
- Built-in healthcheck — orchestrators (HF Spaces, Cloud Run) like this.
- Works identically on every host. **This is the change that frees you
  from Render lock-in.**

`.dockerignore` excludes notebooks, PDFs, and caches so the image stays
slim — only `app.py` and `data/` get copied in.

---

## 6. README.md

Built around a **business framing**, not a tech list:

> A SpaceX launch costs ~$62M; competitors quote upward of $165M. Most
> of that gap comes from booster reuse, so being able to predict re-use
> likelihood is directly tied to pricing strategy.

Sections:
- Demo (placeholder for screenshot/GIF)
- Project structure
- Notebook table (number, file, what it does)
- Run locally / Run with Docker / Run in production
- Deployment targets (HF Spaces, Fly.io, Cloud Run, Railway)
- Roadmap (Phase 2+)
- Tech stack
- Data source
- License

**Why:** the README is the only thing 90% of visitors will read. It has
to lead with *why this project exists*, not *what libraries it uses*.

---

## 7. Hygiene files

- **`.gitignore`** — Python caches, venvs, IDE folders, secrets, mlruns/
- **`.dockerignore`** — keeps the image lean

---

## Verification

Smoke test executed after every change:

```bash
python -c "
import app
print('Import OK')
app.update_pie('ALL', [0, 10000])
app.update_pie('CCAFS LC-40', [0, 10000])
app.update_scatter('ALL', [0, 10000])
app.update_orbit('ALL', [0, 10000])
app.update_kpis('ALL', [0, 10000])
print('All callbacks OK')
"
# → Loaded 90 rows from local cache, all callbacks render ✅
```

---

## What Phase 1 unlocks

- **Phase 2 (ML in the app)** — clean module layout means we can drop in
  `models/falcon9_clf.joblib` and a Prediction tab without rewriting.
- **Phase 3 (Deploy anywhere)** — Dockerfile means HF Spaces, Fly.io,
  Cloud Run, Railway are all one `git push` away.
- **Phase 4 (CI/CD)** — pinned deps + `app:server` entrypoint make
  GitHub Actions trivial.

---

## Files touched

| File | Status |
|---|---|
| `requirements.txt` | rewritten |
| `app.py` | rewritten |
| `data/spacex_launch_data.csv` | created |
| `notebooks/01..07_*.ipynb` | renamed/moved |
| `README.md` | created |
| `Dockerfile` | created |
| `.dockerignore` | created |
| `.gitignore` | created |
| `docs/PHASE_1.md` | created (this file) |
| `assets/` | created (empty) |
