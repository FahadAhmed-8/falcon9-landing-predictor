"""Falcon 9 Landing Predictor — Launch Records Dashboard.

A Dash + Plotly dashboard exploring SpaceX launch outcomes by site,
payload, booster version and orbit. Data is bundled locally so the app
boots quickly and works offline.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import Input, Output, dcc, html

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("falcon9")

DATA_PATH = Path(__file__).parent / "data" / "spacex_launch_data.csv"
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


spacex_df = load_data()
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
            ],
            id="tabs",
            active_tab="pie",
            className="mb-4",
        ),
        html.Footer(
            html.Small(
                [
                    "Built with Dash & Plotly · ",
                    html.A("source", href="https://github.com/", className="text-decoration-none"),
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
