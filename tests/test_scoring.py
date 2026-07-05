from unittest.mock import Mock

import numpy as np

from src.features.feature_engineering import build_features, fit_feature_params
from src.scoring.score_transactions import score_frame


def test_score_frame_returns_ranked_risk_columns(sample_transactions):
    params = fit_feature_params(sample_transactions)
    _, features = build_features(sample_transactions, params=params, include_flagged=False)
    model = Mock()
    model.predict_proba.return_value = np.array(
        [
            [0.05, 0.95],
            [0.10, 0.90],
            [0.80, 0.20],
            [0.99, 0.01],
        ]
    )
    bundle = {
        "model": model,
        "feature_params": params,
        "feature_columns": features,
    }

    scored = score_frame(sample_transactions, bundle)

    assert scored.iloc[0]["predicted_fraud_probability"] == 0.95
    assert scored.iloc[0]["investigation_priority_rank"] == 1
    assert {"fraud_risk_score", "model_risk_segment"}.issubset(scored.columns)