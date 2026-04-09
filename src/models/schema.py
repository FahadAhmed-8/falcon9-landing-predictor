"""Feature schema for the Falcon 9 landing classifier.

Single source of truth — imported by both training and the Dash app
so the form, the joblib pipeline, and the training script can never
drift apart.
"""

from __future__ import annotations

NUMERIC_FEATURES: list[str] = [
    "PayloadMass",
    "Flights",
    "Block",
    "ReusedCount",
]

CATEGORICAL_FEATURES: list[str] = [
    "Orbit",
    "LaunchSite",
]

BOOLEAN_FEATURES: list[str] = [
    "GridFins",
    "Reused",
    "Legs",
]

FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES
TARGET: str = "Class"

# Choice menus for the Dash form (kept here so app + training agree)
ORBITS: list[str] = [
    "ES-L1", "GEO", "GTO", "HEO", "ISS", "LEO", "MEO", "PO", "SO", "SSO", "VLEO",
]
LAUNCH_SITES: list[str] = ["CCAFS SLC 40", "KSC LC 39A", "VAFB SLC 4E"]
BLOCKS: list[float] = [1.0, 2.0, 3.0, 4.0, 5.0]
