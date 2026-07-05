"""Fraud-focused evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_metric(func, *args, default: float = 0.0, **kwargs) -> float:
    try:
        return float(func(*args, **kwargs))
    except ValueError:
        return default


def classification_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    """Return metrics designed for rare-fraud classification."""

    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "pr_auc": _safe_metric(average_precision_score, y_true, scores),
        "roc_auc": _safe_metric(roc_auc_score, y_true, scores),
        "precision": _safe_metric(precision_score, y_true, predictions, zero_division=0),
        "recall": _safe_metric(recall_score, y_true, predictions, zero_division=0),
        "f1": _safe_metric(f1_score, y_true, predictions, zero_division=0),
        "f2": _safe_metric(fbeta_score, y_true, predictions, beta=2, zero_division=0),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
    }


def threshold_optimization_table(
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build an auditable threshold table and optimize for F2."""

    if thresholds is None:
        thresholds = np.unique(
            np.concatenate(
                [
                    np.linspace(0.01, 0.99, 99),
                    np.quantile(scores, np.linspace(0.50, 0.999, 60)),
                ]
            )
        )
    pr_auc = _safe_metric(average_precision_score, y_true, scores)
    roc_auc = _safe_metric(roc_auc_score, y_true, scores)
    rows = []
    total = len(y_true)
    y_true_bool = y_true.astype(bool)
    positives = int(y_true_bool.sum())
    negatives = int(total - positives)
    for threshold in thresholds:
        prediction_bool = scores >= float(threshold)
        tp = int(np.count_nonzero(prediction_bool & y_true_bool))
        fp = int(np.count_nonzero(prediction_bool & ~y_true_bool))
        fn = int(positives - tp)
        tn = int(negatives - fp)
        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        beta_squared = 4.0
        f2 = (
            float((1 + beta_squared) * precision * recall / ((beta_squared * precision) + recall))
            if ((beta_squared * precision) + recall)
            else 0.0
        )
        reviewed = int(tp + fp)
        rows.append(
            {
                "threshold": float(threshold),
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "f2": f2,
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
                "reviewed_transactions": reviewed,
                "review_rate": float(reviewed / total) if total else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["f2", "recall", "precision"], ascending=False)


def top_k_capture_table(
    y_true: np.ndarray,
    scores: np.ndarray,
    review_rates: tuple[float, ...] = (0.001, 0.005, 0.01, 0.05),
) -> pd.DataFrame:
    """Measure fraud capture when investigators review only top-ranked cases."""

    order = np.argsort(-scores)
    y_sorted = y_true[order]
    total_fraud = int(y_true.sum())
    base_rate = float(total_fraud / len(y_true)) if len(y_true) else 0.0
    rows = []
    for review_rate in review_rates:
        reviewed = max(1, int(np.ceil(len(y_true) * review_rate)))
        captured = int(y_sorted[:reviewed].sum())
        precision = captured / reviewed
        recall = captured / total_fraud if total_fraud else 0.0
        rows.append(
            {
                "review_rate": review_rate,
                "reviewed_transactions": reviewed,
                "fraud_captured": captured,
                "precision_at_k": precision,
                "recall_at_k": recall,
                "lift_vs_random": precision / base_rate if base_rate else 0.0,
            }
        )
    return pd.DataFrame(rows)


def assign_risk_segment(scores: np.ndarray) -> np.ndarray:
    """Convert probability scores into business-friendly risk bands."""

    return pd.cut(
        scores,
        bins=[-0.001, 0.05, 0.20, 0.50, 1.001],
        labels=["Low", "Medium", "High", "Critical"],
    ).astype(str)