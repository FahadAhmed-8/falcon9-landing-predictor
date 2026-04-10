"""Falcon 9 Landing Predictor — Launch Records Dashboard.

A Dash + Plotly dashboard exploring SpaceX launch outcomes by site,
payload, booster version and orbit. Data is bundled locally so the app
boots quickly and works offline.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html, no_update

from src.models.schema import BLOCKS, FEATURES, LAUNCH_SITES, ORBITS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("falcon9")

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "spacex_launch_data.csv"
MODEL_PATH = ROOT / "models" / "falcon9_clf.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
REMOTE_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
)


@lru_cache(maxsize=1)
def load_data() -> pd.DataFrame:
    """Load the SpaceX launch dataset, preferring the local copy."""
    if DATA_PATH.exists():
        log.info("Loading dataset from local cache: %s", DATA_PATH)
        df = pd.read_csv(DATA_PATH)
    else:
        log.warning("Local dataset missing — falling back to remote URL")
        df = pd.read_csv(REMOTE_URL)
    df["LandingResult"] = df["Class"].map({1: "Success", 0: "Failure"})
    return df


@lru_cache(maxsize=1)
def load_model():
    """Load the trained classifier pipeline + its metrics, if available."""
    if not MODEL_PATH.exists():
        log.warning("Model artifact missing at %s — Prediction tab will be disabled", MODEL_PATH)
        return None, None
    log.info("Loading model from %s", MODEL_PATH)
    model = joblib.load(MODEL_PATH)
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None
    return model, metrics


spacex_df = load_data()
model, metrics = load_model()
min_payload = float(spacex_df["PayloadMass"].min())
max_payload = float(spacex_df["PayloadMass"].max())
launch_sites = sorted(spacex_df["LaunchSite"].unique())

# ---------- App ----------
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG, dbc.icons.BOOTSTRAP],
    title="Falcon 9 Landing Predictor",
    suppress_callback_exceptions=True,
)
server = app.server  # for gunicorn


def kpi_card(title: str, value: str, icon: str, color: str = "primary") -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.I(className=f"bi {icon} me-2"),
                        html.Span(title, className="text-muted small"),
                    ]
                ),
                html.H3(value, className=f"text-{color} mb-0 mt-1"),
            ]
        ),
        className="shadow-sm",
    )


def build_kpis(df: pd.DataFrame) -> dbc.Row:
    total = len(df)
    successes = int(df["Class"].sum())
    success_rate = (successes / total * 100) if total else 0.0
    heaviest = df["PayloadMass"].max() if total else 0
    top_booster = (
        df[df["Class"] == 1]["BoosterVersion"].value_counts().idxmax()
        if successes
        else "—"
    )
    return dbc.Row(
        [
            dbc.Col(kpi_card("Total launches", f"{total}", "bi-rocket-takeoff", "info"), md=3),
            dbc.Col(
                kpi_card("Success rate", f"{success_rate:.1f}%", "bi-graph-up-arrow", "success"),
                md=3,
            ),
            dbc.Col(
                kpi_card("Heaviest payload", f"{heaviest:,.0f} kg", "bi-box-seam", "warning"),
                md=3,
            ),
            dbc.Col(kpi_card("Top booster", top_booster, "bi-trophy", "danger"), md=3),
        ],
        className="g-3 mb-4",
    )


controls = dbc.Card(
    dbc.CardBody(
        [
            html.Label("Launch site", className="fw-bold"),
            dcc.Dropdown(
                id="site-dropdown",
                options=[{"label": "All sites", "value": "ALL"}]
                + [{"label": s, "value": s} for s in launch_sites],
                value="ALL",
                clearable=False,
                className="mb-3",
            ),
            html.Label("Payload mass (kg)", className="fw-bold"),
            dcc.RangeSlider(
                id="payload-slider",
                min=0,
                max=10000,
                step=500,
                marks={i: str(i) for i in range(0, 10001, 2500)},
                value=[min_payload, max_payload],
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ]
    ),
    className="shadow-sm mb-4",
)


# ---------- Prediction tab ----------
def prediction_form() -> dbc.Card:
    if model is None:
        return dbc.Alert(
            [
                html.I(className="bi bi-exclamation-triangle me-2"),
                "Model artifact not found. Run ",
                html.Code("python -m src.models.train"),
                " to train and save the classifier, then reload this page.",
            ],
            color="warning",
        )
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Predict landing success", className="card-title"),
                html.P(
                    "Enter mission parameters and the trained classifier will return "
                    "the probability of a successful first-stage recovery.",
                    className="text-muted small",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Payload mass (kg)"),
                                dbc.Input(id="pred-payload", type="number", value=5000, min=0, step=100),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                html.Label("Flights (this booster)"),
                                dbc.Input(id="pred-flights", type="number", value=1, min=1, step=1),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                html.Label("Block"),
                                dcc.Dropdown(
                                    id="pred-block",
                                    options=[{"label": str(b), "value": b} for b in BLOCKS],
                                    value=5.0,
                                    clearable=False,
                                ),
                            ],
                            md=4,
                        ),
                    ],
                    className="g-3 mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Reused count"),
                                dbc.Input(id="pred-reusedcount", type="number", value=0, min=0, step=1),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                html.Label("Orbit"),
                                dcc.Dropdown(
                                    id="pred-orbit",
                                    options=[{"label": o, "value": o} for o in ORBITS],
                                    value="LEO",
                                    clearable=False,
                                ),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                html.Label("Launch site"),
                                dcc.Dropdown(
                                    id="pred-site",
                                    options=[{"label": s, "value": s} for s in LAUNCH_SITES],
                                    value="CCAFS SLC 40",
                                    clearable=False,
                                ),
                            ],
                            md=4,
                        ),
                    ],
                    className="g-3 mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Checklist(
                                id="pred-gridfins",
                                options=[{"label": " Grid fins", "value": 1}],
                                value=[1],
                                switch=True,
                            ),
                            md=4,
                        ),
                        dbc.Col(
                            dbc.Checklist(
                                id="pred-reused",
                                options=[{"label": " Reused booster", "value": 1}],
                                value=[],
                                switch=True,
                            ),
                            md=4,
                        ),
                        dbc.Col(
                            dbc.Checklist(
                                id="pred-legs",
                                options=[{"label": " Landing legs", "value": 1}],
                                value=[1],
                                switch=True,
                            ),
                            md=4,
                        ),
                    ],
                    className="g-3 mb-3",
                ),
                dbc.Button(
                    [html.I(className="bi bi-cpu me-2"), "Predict"],
                    id="pred-btn",
                    color="primary",
                    className="mt-2",
                ),
                html.Div(id="pred-result", className="mt-4"),
            ]
        ),
        className="shadow-sm",
    )


# ---------- Performance tab ----------
def performance_view() -> html.Div:
    if metrics is None:
        return dbc.Alert("No metrics available — train the model first.", color="warning")

    leaderboard_rows = [
        html.Tr(
            [
                html.Td(row["name"]),
                html.Td(f"{row['cv_score']:.3f}"),
                html.Td(f"{row['test_accuracy']:.3f}"),
                html.Td(f"{row['test_f1']:.3f}"),
                html.Td(f"{row['test_roc_auc']:.3f}"),
            ],
            className="table-success" if row["name"] == metrics["best_model"] else "",
        )
        for row in metrics["leaderboard"]
    ]
    leaderboard = dbc.Table(
        [
            html.Thead(
                html.Tr([html.Th(c) for c in ["Model", "CV acc", "Test acc", "F1", "ROC AUC"]])
            ),
            html.Tbody(leaderboard_rows),
        ],
        bordered=True,
        striped=True,
        hover=True,
        responsive=True,
        className="mb-4",
    )

    cm = metrics["confusion_matrix"]
    cm_fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=metrics["labels"],
            y=metrics["labels"],
            text=cm,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        )
    )
    cm_fig.update_layout(
        title=f"Confusion matrix — {metrics['best_model']}",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    roc = metrics["roc_curve"]
    roc_fig = go.Figure()
    roc_fig.add_trace(
        go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name=f"AUC = {metrics['test_roc_auc']:.3f}")
    )
    roc_fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash"))
    )
    roc_fig.update_layout(
        title="ROC curve",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(kpi_card("Best model", metrics["best_model"], "bi-trophy", "success"), md=3),
                    dbc.Col(kpi_card("CV accuracy", f"{metrics['cv_score']:.1%}", "bi-bullseye", "info"), md=3),
                    dbc.Col(kpi_card("Test accuracy", f"{metrics['test_accuracy']:.1%}", "bi-check2-circle", "primary"), md=3),
                    dbc.Col(kpi_card("ROC AUC", f"{metrics['test_roc_auc']:.3f}", "bi-graph-up", "warning"), md=3),
                ],
                className="g-3 mb-4",
            ),
            html.H5("Leaderboard"),
            leaderboard,
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(figure=cm_fig), md=6),
                    dbc.Col(dcc.Graph(figure=roc_fig), md=6),
                ]
            ),
        ]
    )


app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                [
                    html.H1(
                        [
                            html.I(className="bi bi-rocket-takeoff-fill me-2"),
                            "Falcon 9 Landing Predictor",
                        ],
                        className="mt-4",
                    ),
                    html.P(
                        "Falcon 9 first-stage landing analytics — explore launch outcomes by "
                        "site, payload and booster version.",
                        className="text-muted",
                    ),
                    html.Hr(),
                ]
            )
        ),
        html.Div(id="kpi-row"),
        controls,
        dbc.Tabs(
            [
                dbc.Tab(
                    dcc.Graph(id="success-pie-chart"),
                    label="Success by site",
                    tab_id="pie",
                ),
                dbc.Tab(
                    dcc.Graph(id="success-payload-scatter-chart"),
                    label="Payload vs outcome",
                    tab_id="scatter",
                ),
                dbc.Tab(
                    dcc.Graph(id="orbit-bar"),
                    label="Outcome by orbit",
                    tab_id="orbit",
                ),
                dbc.Tab(
                    html.Div(prediction_form(), className="mt-3"),
                    label="Predict",
                    tab_id="predict",
                ),
                dbc.Tab(
                    html.Div(performance_view(), className="mt-3"),
                    label="Model performance",
                    tab_id="perf",
                ),
            ],
            id="tabs",
            active_tab="pie",
            className="mb-4",
        ),
        html.Footer(
            html.Small(
                [
                    "Built with Dash & Plotly · ",
                    html.A("source", href="https://github.com/FahadAhmed-8/falcon9-landing-predictor", className="text-decoration-none"),
                ],
                className="text-muted",
            ),
            className="text-center my-4",
        ),
    ],
    fluid=True,
    className="px-4",
)


# ---------- Callbacks ----------
def _filter(site: str, payload_range: list[float]) -> pd.DataFrame:
    low, high = payload_range
    df = spacex_df[(spacex_df["PayloadMass"] >= low) & (spacex_df["PayloadMass"] <= high)]
    if site != "ALL":
        df = df[df["LaunchSite"] == site]
    return df


@app.callback(
    Output("kpi-row", "children"),
    Input("site-dropdown", "value"),
    Input("payload-slider", "value"),
)
def update_kpis(site: str, payload_range: list[float]):
    return build_kpis(_filter(site, payload_range))


@app.callback(
    Output("success-pie-chart", "figure"),
    Input("site-dropdown", "value"),
    Input("payload-slider", "value"),
)
def update_pie(site: str, payload_range: list[float]):
    df = _filter(site, payload_range)
    if site == "ALL":
        agg = df[df["Class"] == 1].groupby("LaunchSite", as_index=False).size()
        fig = px.pie(
            agg,
            values="size",
            names="LaunchSite",
            title="Successful launches by site",
            hole=0.45,
        )
    else:
        fig = px.pie(
            df,
            names="LandingResult",
            title=f"Success vs failure — {site}",
            hole=0.45,
            color="LandingResult",
            color_discrete_map={"Success": "#2ecc71", "Failure": "#e74c3c"},
        )
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
    return fig


@app.callback(
    Output("success-payload-scatter-chart", "figure"),
    Input("site-dropdown", "value"),
    Input("payload-slider", "value"),
)
def update_scatter(site: str, payload_range: list[float]):
    df = _filter(site, payload_range)
    fig = px.scatter(
        df,
        x="PayloadMass",
        y="LandingResult",
        color="BoosterVersion",
        size="PayloadMass",
        hover_data=["LaunchSite", "Orbit"],
        title=("Payload vs outcome — all sites" if site == "ALL" else f"Payload vs outcome — {site}"),
    )
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
    return fig


@app.callback(
    Output("orbit-bar", "figure"),
    Input("site-dropdown", "value"),
    Input("payload-slider", "value"),
)
def update_orbit(site: str, payload_range: list[float]):
    df = _filter(site, payload_range)
    agg = (
        df.groupby("Orbit")
        .agg(launches=("Class", "size"), successes=("Class", "sum"))
        .reset_index()
    )
    agg["success_rate"] = (agg["successes"] / agg["launches"] * 100).round(1)
    fig = px.bar(
        agg.sort_values("success_rate", ascending=False),
        x="Orbit",
        y="success_rate",
        text="success_rate",
        title="Success rate by orbit (%)",
        color="success_rate",
        color_continuous_scale="Viridis",
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", yaxis_range=[0, 110])
    return fig


@app.callback(
    Output("pred-result", "children"),
    Input("pred-btn", "n_clicks"),
    State("pred-payload", "value"),
    State("pred-flights", "value"),
    State("pred-block", "value"),
    State("pred-reusedcount", "value"),
    State("pred-orbit", "value"),
    State("pred-site", "value"),
    State("pred-gridfins", "value"),
    State("pred-reused", "value"),
    State("pred-legs", "value"),
    prevent_initial_call=True,
)
def predict_landing(n_clicks, payload, flights, block, reused_count, orbit, site, gridfins, reused, legs):
    if model is None:
        return dbc.Alert("Model not loaded.", color="danger")
    if any(v is None for v in (payload, flights, block, reused_count, orbit, site)):
        return dbc.Alert("Please fill in all fields before predicting.", color="warning")

    row = pd.DataFrame(
        [
            {
                "PayloadMass": float(payload),
                "Flights": int(flights),
                "Block": float(block),
                "ReusedCount": int(reused_count),
                "Orbit": orbit,
                "LaunchSite": site,
                "GridFins": int(bool(gridfins)),
                "Reused": int(bool(reused)),
                "Legs": int(bool(legs)),
            }
        ]
    )[FEATURES]

    try:
        proba = float(model.predict_proba(row)[0, 1])
    except (AttributeError, NotImplementedError):
        # Fallback for estimators without predict_proba
        proba = float(model.decision_function(row)[0])
        proba = 1 / (1 + pow(2.71828, -proba))

    pred = proba >= 0.5
    color = "success" if pred else "danger"
    label = "LIKELY LAND" if pred else "LIKELY FAIL"
    icon = "bi-check-circle-fill" if pred else "bi-x-circle-fill"

    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.I(className=f"bi {icon} me-2"),
                        html.Span(label, className="fs-4 fw-bold"),
                    ],
                    className=f"text-{color}",
                ),
                dbc.Progress(
                    value=proba * 100,
                    label=f"{proba:.1%}",
                    color=color,
                    striped=True,
                    animated=True,
                    className="mt-3",
                    style={"height": "30px"},
                ),
                html.Small(
                    f"Predicted probability of successful first-stage landing.",
                    className="text-muted mt-2 d-block",
                ),
            ]
        ),
        className=f"border-{color}",
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
