"""Model interpretation with native importances and SHAP-ready fallback notes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def extract_model_importance(model, feature_columns: list[str]) -> pd.DataFrame:
    """Extract model-native importance where available."""

    estimator = model
    if hasattr(model, "named_steps"):
        estimator = list(model.named_steps.values())[-1]

    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        importance = np.abs(np.asarray(estimator.coef_)).ravel()
    else:
        importance = np.zeros(len(feature_columns), dtype=float)

    total = importance.sum()
    normalized = importance / total if total else importance
    return (
        pd.DataFrame({"feature": feature_columns, "importance": normalized})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def compute_permutation_importance(
    model,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
    feature_columns: list[str],
    output_dir: str | Path,
    random_state: int = 42,
    sample_size: int = 5_000,
) -> pd.DataFrame:
    """Compute PR-AUC based permutation importance on a validation sample."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(x_valid) > sample_size:
        sample = x_valid.sample(sample_size, random_state=random_state)
        y_sample = y_valid[sample.index.to_numpy()]
    else:
        sample = x_valid
        y_sample = y_valid

    result = permutation_importance(
        model,
        sample,
        y_sample,
        scoring="average_precision",
        n_repeats=2,
        random_state=random_state,
        n_jobs=-1,
    )
    importance = (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "permutation_importance_mean": result.importances_mean,
                "permutation_importance_std": result.importances_std,
            }
        )
        .sort_values("permutation_importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(output_dir / "permutation_importance.csv", index=False)
    (output_dir / "interpretability_notes.json").write_text(
        json.dumps(
            {
                "method": "Permutation importance using average precision on validation data",
                "shap_status": "Optional. Install shap and rerun the notebook/script to add SHAP plots.",
                "reason": "SHAP is not required for the reproducible baseline environment; permutation importance is used as an equivalent model explanation fallback.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return importance