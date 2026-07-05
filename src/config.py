"""Project configuration and shared constants."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_PATH = RAW_DATA_DIR / "AIML Dataset.csv"

MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"
REPORT_DIR = OUTPUT_DIR / "reports"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
RISK_SCORE_DIR = OUTPUT_DIR / "risk_scores"

TARGET_COLUMN = "isFraud"
LEGACY_FLAG_COLUMN = "isFlaggedFraud"
RANDOM_STATE = 42

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

APPROVED_PALETTE = {
    "light_background": "#D6E4F0",
    "card_background": "#F7FAFC",
    "secondary_surface": "#E2E2E2",
    "primary_text": "#172033",
    "secondary_text": "#52616B",
    "primary_navy": "#0B1F33",
    "accent_blue": "#2563EB",
    "accent_teal": "#1F7A8C",
    "positive_green": "#2E7D32",
    "warning_amber": "#B7791F",
    "risk_red": "#C62828",
    "critical_red": "#8B1E1E",
    "dark_background": "#0B1F33",
    "dark_surface": "#132F4C",
    "dark_primary_text": "#F8FAFC",
    "dark_secondary_text": "#CBD5E1",
}


def ensure_project_directories() -> None:
    """Create all expected output directories without touching raw data."""

    for path in [
        PROCESSED_DATA_DIR,
        MODEL_DIR,
        CHART_DIR,
        REPORT_DIR,
        PREDICTION_DIR,
        RISK_SCORE_DIR,
        OUTPUT_DIR / "dashboard_screenshots",
    ]:
        path.mkdir(parents=True, exist_ok=True)