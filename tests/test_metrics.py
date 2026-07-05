import numpy as np

from src.evaluation.metrics import assign_risk_segment, classification_metrics, threshold_optimization_table, top_k_capture_table


def test_threshold_optimization_returns_review_metrics():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.01, 0.25, 0.80, 0.95])

    table = threshold_optimization_table(y_true, scores)

    assert {"precision", "recall", "f2", "review_rate"}.issubset(table.columns)
    assert table.iloc[0]["f2"] >= 0
    assert table.iloc[0]["pr_auc"] == 1.0


def test_top_k_capture_table_has_lift():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.01, 0.25, 0.80, 0.95])

    table = top_k_capture_table(y_true, scores, review_rates=(0.5,))

    assert table.loc[0, "fraud_captured"] == 2
    assert table.loc[0, "lift_vs_random"] >= 1


def test_classification_metrics_confusion_counts():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.10, 0.20, 0.70, 0.95])

    metrics = classification_metrics(y_true, scores, threshold=0.50)

    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_assign_risk_segment_boundaries():
    segments = list(assign_risk_segment(np.array([0.01, 0.10, 0.30, 0.90])))

    assert segments == ["Low", "Medium", "High", "Critical"]