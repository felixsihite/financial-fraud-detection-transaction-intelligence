"""Train and evaluate fraud-detection models with time-aware validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src.config import (
    CHART_DIR,
    LEGACY_FLAG_COLUMN,
    MODEL_DIR,
    PREDICTION_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RAW_DATA_PATH,
    REPORT_DIR,
    RISK_SCORE_DIR,
    TARGET_COLUMN,
)
from src.evaluation.metrics import (
    assign_risk_segment,
    classification_metrics,
    threshold_optimization_table,
    top_k_capture_table,
)
from src.features.feature_engineering import align_feature_columns, build_features, fit_feature_params
from src.interpretation.explain import compute_permutation_importance, extract_model_importance
from src.visualization.theme import APPROVED_PALETTE, set_plot_theme


def _predict_scores(model, matrix: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(matrix)[:, 1]
    decision = model.decision_function(matrix)
    return (decision - decision.min()) / max(decision.max() - decision.min(), 1e-9)


def _step_cutoff(input_path: Path, split_quantile: float, chunksize: int) -> int:
    step_counts: dict[int, int] = {}
    for chunk in pd.read_csv(input_path, usecols=["step"], chunksize=chunksize):
        counts = chunk["step"].value_counts().to_dict()
        for step, count in counts.items():
            step_counts[int(step)] = step_counts.get(int(step), 0) + int(count)
    if not step_counts:
        raise ValueError("Unable to infer step range from dataset.")

    target_position = int(np.floor(sum(step_counts.values()) * split_quantile))
    cumulative = 0
    for step in sorted(step_counts):
        cumulative += step_counts[step]
        if cumulative >= target_position:
            return int(step)
    return int(max(step_counts))


def build_time_aware_modeling_frames(
    input_path: str | Path = RAW_DATA_PATH,
    split_quantile: float = 0.75,
    max_train_nonfraud: int = 220_000,
    chunksize: int = 500_000,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build a fraud-enriched training sample and a chronological validation set."""

    input_path = Path(input_path)
    cutoff = _step_cutoff(input_path, split_quantile, chunksize)
    train_fraud_parts: list[pd.DataFrame] = []
    train_nonfraud_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []
    sampled_nonfraud = 0
    chunk_index = 0

    for chunk in pd.read_csv(input_path, chunksize=chunksize):
        train_chunk = chunk[chunk["step"] <= cutoff]
        validation_chunk = chunk[chunk["step"] > cutoff]

        if not validation_chunk.empty:
            validation_parts.append(validation_chunk)

        if not train_chunk.empty:
            fraud = train_chunk[train_chunk[TARGET_COLUMN] == 1]
            nonfraud = train_chunk[train_chunk[TARGET_COLUMN] == 0]
            if not fraud.empty:
                train_fraud_parts.append(fraud)

            remaining = max_train_nonfraud - sampled_nonfraud
            if remaining > 0 and not nonfraud.empty:
                draw = min(remaining, len(nonfraud))
                sampled = nonfraud.sample(draw, random_state=random_state + chunk_index)
                train_nonfraud_parts.append(sampled)
                sampled_nonfraud += len(sampled)

        chunk_index += 1

    if not train_fraud_parts or not train_nonfraud_parts or not validation_parts:
        raise ValueError("Modeling split failed. Check the raw dataset and step values.")

    train_frame = (
        pd.concat(train_fraud_parts + train_nonfraud_parts, ignore_index=True)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )
    validation_frame = pd.concat(validation_parts, ignore_index=True)

    metadata = {
        "split_strategy": "time-aware validation using step",
        "step_cutoff": cutoff,
        "train_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
        "train_fraud_rows": int(train_frame[TARGET_COLUMN].sum()),
        "validation_fraud_rows": int(validation_frame[TARGET_COLUMN].sum()),
        "max_train_nonfraud": int(max_train_nonfraud),
    }
    return train_frame, validation_frame, metadata


def _model_candidates(random_state: int = RANDOM_STATE) -> dict[str, Any]:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1_000,
                        solver="lbfgs",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=40,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=50,
            max_depth=12,
            min_samples_leaf=25,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=120,
            max_leaf_nodes=31,
            l2_regularization=0.03,
            random_state=random_state,
        ),
    }


def _fit_model(model_name: str, model, x_train: pd.DataFrame, y_train: np.ndarray):
    if model_name == "HistGradientBoosting":
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        return model.fit(x_train, y_train, sample_weight=sample_weight)
    return model.fit(x_train, y_train)


def _save_confusion_matrix(y_true: np.ndarray, scores: np.ndarray, threshold: float, path: Path) -> None:
    predictions = (scores >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    plt.figure(figsize=(6.5, 5.2))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=",d",
        cmap="Blues",
        xticklabels=["Predicted Non-Fraud", "Predicted Fraud"],
        yticklabels=["Actual Non-Fraud", "Actual Fraud"],
        cbar=False,
    )
    plt.title("Confusion Matrix at Optimized Threshold")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_model_comparison(comparison: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(9.2, 5.4))
    plot_frame = comparison[comparison["feature_set"] == "final_no_legacy_flag"].copy()
    sns.barplot(data=plot_frame, x="pr_auc", y="model_name", color=APPROVED_PALETTE["accent_blue"])
    plt.xlabel("PR-AUC")
    plt.ylabel("")
    plt.title("Model Comparison by PR-AUC")
    plt.xlim(0, max(1.0, float(plot_frame["pr_auc"].max()) * 1.08))
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_threshold_tradeoff(thresholds: pd.DataFrame, path: Path) -> None:
    top = thresholds.sort_values("threshold").copy()
    plt.figure(figsize=(9.5, 5.4))
    plt.plot(top["threshold"], top["precision"], label="Precision", color=APPROVED_PALETTE["accent_blue"])
    plt.plot(top["threshold"], top["recall"], label="Recall", color=APPROVED_PALETTE["risk_red"])
    plt.plot(top["threshold"], top["f2"], label="F2", color=APPROVED_PALETTE["accent_teal"])
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision, Recall, and F2 Threshold Tradeoff")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_feature_importance(importance: pd.DataFrame, value_column: str, path: Path) -> None:
    plot_frame = importance.head(15).sort_values(value_column, ascending=True)
    plt.figure(figsize=(9.2, 6.4))
    sns.barplot(data=plot_frame, x=value_column, y="feature", color=APPROVED_PALETTE["accent_teal"])
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.title("Top Fraud Prediction Drivers")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _save_curves(y_true: np.ndarray, scores: np.ndarray) -> None:
    PrecisionRecallDisplay.from_predictions(y_true, scores)
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "precision_recall_curve.png", dpi=180)
    plt.close()

    RocCurveDisplay.from_predictions(y_true, scores)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "roc_curve.png", dpi=180)
    plt.close()


def _render_model_markdown(summary: dict[str, Any]) -> str:
    best = summary["best_model"]
    metrics = summary["best_model_metrics"]
    return "\n".join(
        [
            "# Model Performance Summary",
            "",
            f"- Validation strategy: {summary['split_metadata']['split_strategy']}",
            f"- Step cutoff: {summary['split_metadata']['step_cutoff']}",
            f"- Best final model: {best}",
            f"- Optimized threshold: {metrics['threshold']:.6f}",
            f"- PR-AUC: {metrics['pr_auc']:.6f}",
            f"- ROC-AUC: {metrics['roc_auc']:.6f}",
            f"- Fraud precision: {metrics['precision']:.6f}",
            f"- Fraud recall: {metrics['recall']:.6f}",
            f"- F2 score: {metrics['f2']:.6f}",
            f"- False positives: {metrics['false_positives']:,}",
            f"- False negatives: {metrics['false_negatives']:,}",
            "",
            "The selected model excludes `isFlaggedFraud`. A separate benchmark with the legacy flag is generated only to quantify its effect.",
        ]
    )


def train_and_evaluate(
    input_path: str | Path = RAW_DATA_PATH,
    max_train_nonfraud: int = 220_000,
    split_quantile: float = 0.75,
    chunksize: int = 500_000,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Train models, evaluate with PR-AUC, tune threshold, and export artifacts."""

    set_plot_theme()
    for path in [PROCESSED_DATA_DIR, MODEL_DIR, CHART_DIR, REPORT_DIR, PREDICTION_DIR, RISK_SCORE_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    train_frame, validation_frame, split_metadata = build_time_aware_modeling_frames(
        input_path=input_path,
        split_quantile=split_quantile,
        max_train_nonfraud=max_train_nonfraud,
        chunksize=chunksize,
        random_state=random_state,
    )

    train_frame.to_csv(PROCESSED_DATA_DIR / "modeling_train_sample.csv", index=False)

    feature_params = fit_feature_params(train_frame)
    train_features, feature_columns = build_features(train_frame, feature_params, include_flagged=False)
    valid_features, _ = build_features(validation_frame, feature_params, include_flagged=False)
    x_train = align_feature_columns(train_features, feature_columns).reset_index(drop=True)
    x_valid = align_feature_columns(valid_features, feature_columns).reset_index(drop=True)
    y_train = train_frame[TARGET_COLUMN].astype(int).to_numpy()
    y_valid = validation_frame[TARGET_COLUMN].astype(int).to_numpy()

    comparison_rows = []
    fitted_models = {}
    model_scores = {}

    for model_name, model in _model_candidates(random_state).items():
        fitted = _fit_model(model_name, model, x_train, y_train)
        scores = _predict_scores(fitted, x_valid)
        threshold_table = threshold_optimization_table(y_valid, scores)
        best_threshold = float(threshold_table.iloc[0]["threshold"])
        metrics = classification_metrics(y_valid, scores, best_threshold)
        comparison_rows.append(
            {
                "model_name": model_name,
                "feature_set": "final_no_legacy_flag",
                **metrics,
            }
        )
        fitted_models[model_name] = fitted
        model_scores[model_name] = scores

    flagged_train, flagged_columns = build_features(train_frame, feature_params, include_flagged=True)
    flagged_valid, _ = build_features(validation_frame, feature_params, include_flagged=True)
    x_train_flagged = align_feature_columns(flagged_train, flagged_columns).reset_index(drop=True)
    x_valid_flagged = align_feature_columns(flagged_valid, flagged_columns).reset_index(drop=True)
    flagged_model = _fit_model(
        "HistGradientBoosting",
        _model_candidates(random_state)["HistGradientBoosting"],
        x_train_flagged,
        y_train,
    )
    flagged_scores = _predict_scores(flagged_model, x_valid_flagged)
    flagged_thresholds = threshold_optimization_table(y_valid, flagged_scores)
    flagged_metrics = classification_metrics(y_valid, flagged_scores, float(flagged_thresholds.iloc[0]["threshold"]))
    comparison_rows.append(
        {
            "model_name": "HistGradientBoosting + isFlaggedFraud benchmark",
            "feature_set": "benchmark_with_legacy_flag",
            **flagged_metrics,
        }
    )

    comparison = pd.DataFrame(comparison_rows).sort_values("pr_auc", ascending=False)
    final_comparison = comparison[comparison["feature_set"] == "final_no_legacy_flag"].sort_values(
        "pr_auc", ascending=False
    )
    best_model_name = str(final_comparison.iloc[0]["model_name"])
    best_model = fitted_models[best_model_name]
    best_scores = model_scores[best_model_name]

    threshold_table = threshold_optimization_table(y_valid, best_scores)
    best_threshold = float(threshold_table.iloc[0]["threshold"])
    best_metrics = classification_metrics(y_valid, best_scores, best_threshold)
    top_k = top_k_capture_table(y_valid, best_scores)

    risk_score = best_scores * 1000.0
    scored = validation_frame[
        [
            "step",
            "type",
            "amount",
            "nameOrig",
            "oldbalanceOrg",
            "newbalanceOrig",
            "nameDest",
            "oldbalanceDest",
            "newbalanceDest",
            TARGET_COLUMN,
            LEGACY_FLAG_COLUMN,
        ]
    ].copy()
    scored["predicted_fraud_probability"] = best_scores
    scored["fraud_risk_score"] = risk_score.round(2)
    scored["model_risk_segment"] = assign_risk_segment(best_scores)
    scored["investigation_priority_rank"] = scored["predicted_fraud_probability"].rank(
        method="first", ascending=False
    ).astype(int)
    scored = scored.sort_values("predicted_fraud_probability", ascending=False)

    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    threshold_table.head(50).to_csv(REPORT_DIR / "threshold_optimization_top50.csv", index=False)
    top_k.to_csv(REPORT_DIR / "top_k_fraud_capture.csv", index=False)
    scored.head(1_000).to_csv(RISK_SCORE_DIR / "investigation_priority_top_1000.csv", index=False)
    scored.head(20_000).to_csv(PREDICTION_DIR / "validation_predictions_top_20000.csv", index=False)
    scored.sample(min(20_000, len(scored)), random_state=random_state).to_csv(
        PROCESSED_DATA_DIR / "validation_scored_sample.csv", index=False
    )

    model_importance = extract_model_importance(best_model, feature_columns)
    model_importance.to_csv(REPORT_DIR / "model_native_feature_importance.csv", index=False)
    permutation = compute_permutation_importance(
        best_model,
        x_valid,
        y_valid,
        feature_columns,
        output_dir=REPORT_DIR,
        random_state=random_state,
    )

    _plot_model_comparison(comparison, CHART_DIR / "model_comparison_pr_auc.png")
    _plot_threshold_tradeoff(threshold_table, CHART_DIR / "threshold_tradeoff.png")
    _plot_feature_importance(
        permutation,
        "permutation_importance_mean",
        CHART_DIR / "feature_importance.png",
    )
    _save_confusion_matrix(y_valid, best_scores, best_threshold, CHART_DIR / "confusion_matrix.png")
    _save_curves(y_valid, best_scores)

    model_bundle = {
        "model": best_model,
        "model_name": best_model_name,
        "feature_columns": feature_columns,
        "feature_params": feature_params,
        "threshold": best_threshold,
        "metrics": best_metrics,
        "split_metadata": split_metadata,
        "excludes_legacy_flag": True,
    }
    joblib.dump(model_bundle, MODEL_DIR / "fraud_detection_model.joblib")

    summary = {
        "best_model": best_model_name,
        "best_model_metrics": best_metrics,
        "split_metadata": split_metadata,
        "feature_columns": feature_columns,
        "top_k_capture": top_k.to_dict(orient="records"),
        "benchmark_note": "Final model excludes isFlaggedFraud; benchmark with legacy flag is reported separately.",
    }
    (REPORT_DIR / "model_performance_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (REPORT_DIR / "model_performance_summary.md").write_text(
        _render_model_markdown(summary),
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    train_and_evaluate()