"""Train the Falcon 9 first-stage landing classifier.

Reproducible end-to-end training:
1. Load the bundled launch dataset
2. Build a ColumnTransformer (scale numerics, one-hot categoricals)
3. GridSearchCV over Logistic Regression / SVM / Decision Tree / KNN
4. Pick the model with the best CV score, refit on train, evaluate on test
5. Save the full sklearn Pipeline as `models/falcon9_clf.joblib`
6. Save metrics + per-model leaderboard to `models/metrics.json`

Run:
    python -m src.models.train
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from src.models.schema import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "spacex_launch_data.csv"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "falcon9_clf.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

RANDOM_STATE = 2
TEST_SIZE = 0.2
CV_FOLDS = 10


@dataclass
class CandidateResult:
    name: str
    best_params: dict[str, Any]
    cv_score: float
    test_accuracy: float
    test_f1: float
    test_roc_auc: float
    estimator: Pipeline


def build_preprocessor() -> ColumnTransformer:
    """Scale numerics, one-hot encode categoricals, pass booleans through."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("bool", "passthrough", BOOLEAN_FEATURES),
        ]
    )


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    log.info("Loading dataset from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    # Ensure booleans are int (sklearn passthrough handles them either way)
    for col in BOOLEAN_FEATURES:
        df[col] = df[col].astype(int)
    X = df[FEATURES]
    y = df[TARGET]
    log.info("Dataset shape: %s, class balance: %s", X.shape, y.value_counts().to_dict())
    return X, y


def candidates(preprocessor: ColumnTransformer) -> dict[str, tuple[Pipeline, dict[str, list]]]:
    """Define each candidate model + its grid as a Pipeline."""
    return {
        "Logistic Regression": (
            Pipeline([("prep", preprocessor), ("clf", LogisticRegression(max_iter=1000))]),
            {
                "clf__C": [0.01, 0.1, 1, 10],
                "clf__penalty": ["l2"],
                "clf__solver": ["lbfgs"],
            },
        ),
        "SVM": (
            Pipeline([("prep", preprocessor), ("clf", SVC(probability=True))]),
            {
                "clf__C": np.logspace(-3, 3, 5).tolist(),
                "clf__gamma": np.logspace(-3, 3, 5).tolist(),
                "clf__kernel": ["linear", "rbf", "sigmoid"],
            },
        ),
        "Decision Tree": (
            Pipeline([("prep", preprocessor), ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))]),
            {
                "clf__criterion": ["gini", "entropy"],
                "clf__max_depth": [2, 4, 6, 8, 10, 12],
                "clf__min_samples_leaf": [1, 2, 4],
                "clf__min_samples_split": [2, 5, 10],
            },
        ),
        "KNN": (
            Pipeline([("prep", preprocessor), ("clf", KNeighborsClassifier())]),
            {
                "clf__n_neighbors": list(range(1, 11)),
                "clf__p": [1, 2],
                "clf__weights": ["uniform", "distance"],
            },
        ),
    }


def evaluate(name: str, gs: GridSearchCV, X_test: pd.DataFrame, y_test: pd.Series) -> CandidateResult:
    best = gs.best_estimator_
    y_pred = best.predict(X_test)
    # ROC AUC needs probabilities; some estimators may not have predict_proba
    try:
        y_proba = best.predict_proba(X_test)[:, 1]
    except (AttributeError, NotImplementedError):
        y_proba = best.decision_function(X_test)
    return CandidateResult(
        name=name,
        best_params={k: _jsonify(v) for k, v in gs.best_params_.items()},
        cv_score=float(gs.best_score_),
        test_accuracy=float(accuracy_score(y_test, y_pred)),
        test_f1=float(f1_score(y_test, y_pred)),
        test_roc_auc=float(roc_auc_score(y_test, y_proba)),
        estimator=best,
    )


def _jsonify(v: Any) -> Any:
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor()
    results: list[CandidateResult] = []

    for name, (pipe, grid) in candidates(preprocessor).items():
        log.info("Tuning %s …", name)
        gs = GridSearchCV(pipe, grid, cv=CV_FOLDS, n_jobs=-1, scoring="accuracy")
        gs.fit(X_train, y_train)
        result = evaluate(name, gs, X_test, y_test)
        log.info(
            "  CV=%.3f  test_acc=%.3f  test_f1=%.3f  test_auc=%.3f",
            result.cv_score, result.test_accuracy, result.test_f1, result.test_roc_auc,
        )
        results.append(result)

    # Pick the model with the best CV score (matches the notebook's logic)
    best = max(results, key=lambda r: r.cv_score)
    log.info("Best model: %s (CV=%.3f)", best.name, best.cv_score)

    # Confusion matrix + ROC curve for the winner — used by the Performance tab
    y_pred = best.estimator.predict(X_test)
    cm = confusion_matrix(y_test, y_pred).tolist()
    try:
        y_proba = best.estimator.predict_proba(X_test)[:, 1]
    except (AttributeError, NotImplementedError):
        y_proba = best.estimator.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    metrics = {
        "best_model": best.name,
        "best_params": best.best_params,
        "cv_score": best.cv_score,
        "test_accuracy": best.test_accuracy,
        "test_f1": best.test_f1,
        "test_roc_auc": best.test_roc_auc,
        "confusion_matrix": cm,
        "labels": ["Did not land", "Landed"],
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "leaderboard": [
            {
                "name": r.name,
                "cv_score": r.cv_score,
                "test_accuracy": r.test_accuracy,
                "test_f1": r.test_f1,
                "test_roc_auc": r.test_roc_auc,
                "best_params": r.best_params,
            }
            for r in sorted(results, key=lambda r: r.cv_score, reverse=True)
        ],
    }

    joblib.dump(best.estimator, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    log.info("Saved %s", MODEL_PATH)
    log.info("Saved %s", METRICS_PATH)


if __name__ == "__main__":
    main()
