"""Score new transactions with the trained fraud model."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.config import MODEL_DIR
from src.evaluation.metrics import assign_risk_segment
from src.features.feature_engineering import align_feature_columns, build_features


def load_model_bundle(model_path: str | Path = MODEL_DIR / "fraud_detection_model.joblib") -> dict:
    """Load the trained model bundle exported by the pipeline."""

    return joblib.load(model_path)


def score_frame(frame: pd.DataFrame, model_bundle: dict) -> pd.DataFrame:
    """Return a scored transaction frame with probability, score, and segment."""

    features, _ = build_features(
        frame,
        params=model_bundle["feature_params"],
        include_flagged=False,
    )
    matrix = align_feature_columns(features, model_bundle["feature_columns"])
    probabilities = model_bundle["model"].predict_proba(matrix)[:, 1]
    scored = frame.copy()
    scored["predicted_fraud_probability"] = probabilities
    scored["fraud_risk_score"] = (probabilities * 1000).round(2)
    scored["model_risk_segment"] = assign_risk_segment(probabilities)
    scored["investigation_priority_rank"] = scored["predicted_fraud_probability"].rank(
        method="first", ascending=False
    ).astype(int)
    return scored.sort_values("predicted_fraud_probability", ascending=False)


def score_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    model_path: str | Path = MODEL_DIR / "fraud_detection_model.joblib",
) -> Path:
    """Score a CSV file and save the ranked investigation queue."""

    bundle = load_model_bundle(model_path)
    frame = pd.read_csv(input_csv)
    scored = score_frame(frame, bundle)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_csv, index=False)
    return output_csv