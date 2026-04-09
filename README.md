# Falcon 9 Landing Predictor

> **End-to-end ML system predicting Falcon 9 first-stage landing success — data pipeline, model, and live Dash dashboard. Dockerized.**

A SpaceX launch costs ~$62M; competitors quote upward of $165M. Most of
that gap comes from booster reuse, so being able to predict re-use
likelihood is directly tied to pricing strategy. This project takes the
problem end-to-end: data collection → wrangling → EDA → interactive
dashboard → ML prediction.

---

## Demo

Interactive Dash dashboard with KPI cards, site filter, payload range slider,
and three analytical views:

- **Success by site** — donut of successful launches across launch sites
- **Payload vs outcome** — scatter colored by booster version
- **Outcome by orbit** — success rate per orbit

> _Add a screenshot or GIF here once deployed._

---

## Project structure

```
falcon9-landing-predictor/
├── app.py                  # Dash app entrypoint
├── data/
│   └── spacex_launch_data.csv   # bundled dataset (no remote fetch on boot)
├── notebooks/              # data collection, wrangling, EDA, ML
├── requirements.txt        # pinned deps
├── Dockerfile              # container build
└── README.md
```

## Notebooks

| # | Notebook | What it does |
|---|---|---|
| 1 | `01_data_collection_api.ipynb` | Pulls launch data from the SpaceX REST API |
| 2 | `02_data_collection_webscrape.ipynb` | Scrapes Falcon 9 launch records from Wikipedia |
| 3 | `03_data_wrangling.ipynb` | Cleans, encodes the target `Class` (landing success) |
| 4 | `04_eda_sql.ipynb` | EDA via SQL on a SQLite copy |
| 5 | `05_eda_viz.ipynb` | Pandas / seaborn EDA |
| 6 | `06_launch_site_map.ipynb` | Folium map of launch sites + proximity analysis |
| 7 | `07_ml_prediction.ipynb` | Trains LR / SVM / Tree / KNN, tunes with GridSearchCV |

---

## Run locally

```bash
pip install -r requirements.txt
python app.py
# open http://localhost:8050
```

## Run with Docker

```bash
docker build -t falcon9-landing-predictor .
docker run -p 8050:8050 falcon9-landing-predictor
```

## Run in production

```bash
gunicorn app:server -b 0.0.0.0:8050 --workers 2 --timeout 60
```

---

## Deployment

The app is a single Dash process exposing `server` for any WSGI host.
Recommended targets:

- **Hugging Face Spaces** (Docker SDK) — best for ML portfolio visibility
- **Fly.io** / **Google Cloud Run** — production-grade, scales to zero
- **Railway** — fastest "git push" experience

The bundled local dataset means cold starts are fast and the app
works fully offline.

---

## Roadmap

- [ ] Add Prediction tab — load `models/falcon9_clf.joblib` and serve live predictions
- [ ] Model performance tab (confusion matrix, ROC, PR curve, CV scores)
- [ ] Embed folium launch-site proximity map
- [ ] FastAPI `/predict` endpoint, called by Dash
- [ ] MLflow experiment tracking + DVC for data versioning
- [ ] GitHub Actions CI (ruff + pytest + docker build)
- [ ] Pytest coverage of data transforms

---

## Tech

`Python 3.11` · `Dash 2.17` · `Plotly 5.22` · `pandas 2.2` ·
`dash-bootstrap-components` · `gunicorn`

## Data source

[IBM Skills Network — SpaceX Falcon 9 dataset](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv)

## License

MIT
