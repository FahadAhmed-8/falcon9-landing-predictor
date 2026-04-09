# Phase 2 — Model in the App

> Goal: stop being "a notebook with a dashboard" and become "a deployed
> ML system." Take the trained classifier from `notebooks/07_ml_prediction.ipynb`,
> turn it into a reproducible training script, save it as a single
> end-to-end sklearn `Pipeline`, and serve live predictions from the
> Dash app.

This is the change that flips the project from "EDA dashboard" to
"applied ML." It is the single biggest jump in resume value.

---

## TL;DR

| Area | Before | After |
|---|---|---|
| Model training | Inside a Coursera notebook, not reproducible | `python -m src.models.train` — one command, deterministic |
| Feature engineering | 83-column hand-encoded CSV (one-hot Serial per booster) | Clean 9-feature `ColumnTransformer` (scale + one-hot + passthrough) |
| Model artifact | None | `models/falcon9_clf.joblib` — full sklearn `Pipeline`, raw inputs in → probability out |
| Metrics | Printed in cells, lost on kernel restart | `models/metrics.json` — leaderboard, CM, ROC, params |
| Dashboard | 3 EDA tabs only | **5 tabs** — added Predict + Model performance |
| Coupling | Form fields, schema, training script could drift | `src/models/schema.py` — single source of truth imported by both |

---

## 1. Problem with the notebook's approach

The Coursera notebook trains on `dataset_part_3.csv`, a pre-encoded
83-column CSV where every booster serial number (`Serial_B1037`,
`Serial_B1042`, …) is its own one-hot column. That works for getting a
high CV score in a classroom but is **useless for serving**:

- A user filling out a form has no idea what `Serial_B1037` means.
- Adding a new booster requires re-encoding the whole training set.
- The 83 columns are mostly zeros — the model is largely memorising.

I rebuilt the feature pipeline from `dataset_part_2.csv` (the
human-readable file we already bundle) using **9 meaningful features**:

| Feature | Type | Source |
|---|---|---|
| `PayloadMass` | numeric | scaled |
| `Flights` | numeric | scaled |
| `Block` | numeric | scaled |
| `ReusedCount` | numeric | scaled |
| `Orbit` | categorical (11) | one-hot |
| `LaunchSite` | categorical (3) | one-hot |
| `GridFins` | boolean | passthrough |
| `Reused` | boolean | passthrough |
| `Legs` | boolean | passthrough |

The feature schema lives in `src/models/schema.py` and is imported by
**both** the training script and the Dash form, so they cannot drift.

---

## 2. Reproducible training script

**File:** `src/models/train.py`

```bash
python -m src.models.train
```

What it does:

1. Loads `data/spacex_launch_data.csv` (90 rows, 60 land / 30 fail)
2. Stratified 80/20 train/test split (`random_state=2`)
3. Builds a `ColumnTransformer`:
   - `StandardScaler` on numerics
   - `OneHotEncoder(handle_unknown="ignore")` on categoricals
   - passthrough on booleans
4. **GridSearchCV (10-fold)** over the same 4 candidates the notebook
   used: Logistic Regression, SVM, Decision Tree, KNN
5. Picks the model with the best CV accuracy
6. Refits on the full training set, evaluates on the held-out test set
7. Saves:
   - `models/falcon9_clf.joblib` — the entire Pipeline (preprocessor + classifier)
   - `models/metrics.json` — winner, params, CV/test/F1/AUC, confusion matrix, ROC curve, full leaderboard

Each candidate is wrapped in its own `Pipeline` so the preprocessor is
fit *inside* the cross-validation loop — no test-set leakage.

### Why save the Pipeline (not just the classifier)

```python
joblib.dump(best.estimator, MODEL_PATH)   # Pipeline = preprocessor + clf
```

The Dash app can now do this:

```python
row = pd.DataFrame([{...raw form values...}])
proba = model.predict_proba(row)[0, 1]
```

No manual encoding, no scaler-on-the-side, no version mismatch. This
is the right way to ship sklearn models.

---

## 3. Results

10-fold CV on the training set; held-out 18-sample test set.

| Model | CV acc | Test acc | F1 | ROC AUC | Best params |
|---|---:|---:|---:|---:|---|
| **Decision Tree** ⭐ | **0.863** | 0.778 | 0.833 | 0.785 | entropy, depth=4, leaf=2 |
| KNN | 0.863 | 0.611 | 0.667 | 0.701 | k=7, p=1, uniform |
| Logistic Regression | 0.848 | 0.722 | 0.800 | 0.750 | C=0.1, l2 |
| SVM | 0.838 | 0.833 | 0.889 | 0.750 | rbf, C=31.6, γ=0.001 |

**Decision Tree** wins on CV (the notebook's tie-break) and is what gets
saved. Note that **SVM scores higher on the held-out test set** —
unsurprising on 18 samples, where one or two predictions move the
needle by 5%. The leaderboard is exposed in the Performance tab so
this trade-off is visible.

Confusion matrix (Decision Tree, test set):

|  | Pred: didn't land | Pred: landed |
|---|---:|---:|
| **Actual: didn't land** | 4 | 2 |
| **Actual: landed**      | 2 | 10 |

### Sanity-check predictions

After training, I hand-tested three scenarios against the saved model:

| Mission | Probability of landing |
|---|---:|
| Modern reusable LEO mission (Block 5, GridFins, Legs) | **82.4%** |
| Old expendable heavy GTO (Block 1, no fins/legs) | **0.0%** |
| Modern heavy GTO (Block 5, GridFins, Legs) | 60.0% |

Behaviour matches physical intuition — modern reusable hardware lands,
old expendable hardware doesn't, and heavy GTO missions are borderline.

---

## 4. Dashboard changes (`app.py`)

### New: Predict tab

A form with:
- Numerics: Payload mass (kg), Flights, Block, Reused count
- Dropdowns: Orbit (11), Launch site (3)
- Boolean toggle switches: Grid fins, Reused booster, Landing legs

On click → `model.predict_proba()` → returns:
- A coloured **LIKELY LAND / LIKELY FAIL** banner
- An animated probability bar (e.g. *82.4%*)

The form fields are populated from `src/models/schema.py`, so adding a
new orbit or launch site is a one-line change in one place.

### New: Model performance tab

- 4 KPI cards: best model, CV accuracy, test accuracy, ROC AUC
- Full **leaderboard table** with the row for the winning model
  highlighted green
- **Confusion matrix** heatmap (Plotly)
- **ROC curve** with diagonal random-classifier reference

### Engineering touches

- `load_model()` is `lru_cache`-d and **gracefully degrades** — if the
  joblib file is missing the Predict tab shows a warning instead of
  crashing the whole app.
- `load_data()` and `load_model()` are completely independent —
  visualisations still work even without a trained model.
- The Dash callback uses `prevent_initial_call=True` so the prediction
  doesn't fire before the form is filled.

---

## 5. Repository changes

```
src/
├── __init__.py
└── models/
    ├── __init__.py
    ├── schema.py         # NEW — features, choice menus
    └── train.py          # NEW — reproducible training script

models/                   # NEW
├── falcon9_clf.joblib    # 13 KB sklearn Pipeline
└── metrics.json          # leaderboard + CM + ROC

docs/
└── PHASE_2.md            # this file
```

`requirements.txt` gained:
```
scikit-learn==1.5.1
joblib==1.4.2
numpy==1.26.4
```

`Dockerfile` now also copies `src/` and `models/` so the container
ships with the trained artifact baked in — no separate model server,
no S3 bucket, no cold-boot training step.

`.gitignore` got an entry for `.claude/` (personal Claude Code config
that accidentally landed in the Phase 1 commit).

---

## 6. Verification

```bash
# 1. Train
python -m src.models.train
# → Best model: Decision Tree (CV=0.863)
# → Saved models/falcon9_clf.joblib
# → Saved models/metrics.json

# 2. Smoke-test the app
python -c "
import app
print('Model loaded:', app.model is not None)         # True
print('Metrics loaded:', app.metrics is not None)     # True
app.predict_landing(1, 5000, 1, 5.0, 0, 'LEO', 'CCAFS SLC 40', [1], [], [1])
app.update_pie('ALL', [0, 10000])
"
# → All green

# 3. Run locally
python app.py
# → http://localhost:8050 — try the Predict tab
```

---

## 7. What Phase 2 unlocks

- **Phase 3 — Deploy** — the Docker image now ships with the trained
  model, so any host that runs containers (Hugging Face Spaces, Fly.io,
  Cloud Run) gets a working ML demo with one push.
- **Phase 5 — FastAPI `/predict`** — `joblib.load(...)` already works
  end-to-end, so wrapping it in a FastAPI route is ~30 lines.
- **MLflow / DVC** — `train.py` is the natural place to drop
  `mlflow.log_metric` calls; no notebook surgery needed.

---

## Files touched in Phase 2

| File | Status |
|---|---|
| `requirements.txt` | added scikit-learn, joblib, numpy |
| `src/__init__.py` | created |
| `src/models/__init__.py` | created |
| `src/models/schema.py` | created |
| `src/models/train.py` | created |
| `models/falcon9_clf.joblib` | generated |
| `models/metrics.json` | generated |
| `app.py` | added load_model(), prediction_form(), performance_view(), Predict + Performance tabs, predict_landing callback |
| `Dockerfile` | copies src/ and models/ |
| `.gitignore` | excludes .claude/ |
| `README.md` | updated with model section, new structure, roadmap |
| `docs/PHASE_2.md` | created (this file) |
