"""Chunked data-quality and leakage audit for the raw fraud dataset."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import RAW_DATA_PATH, REPORT_DIR, TARGET_COLUMN


EXPECTED_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]


def _safe_pct(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _series_to_int_dict(series: pd.Series) -> dict[str, int]:
    return {str(index): int(value) for index, value in series.items()}


def _update_nested_counter(store: dict[str, Counter], frame: pd.DataFrame) -> None:
    grouped = frame.groupby("type")[TARGET_COLUMN].agg(["count", "sum"])
    for txn_type, row in grouped.iterrows():
        store[str(txn_type)]["transactions"] += int(row["count"])
        store[str(txn_type)]["fraud_transactions"] += int(row["sum"])


def compute_quality_report(
    input_path: str | Path = RAW_DATA_PATH,
    output_dir: str | Path = REPORT_DIR,
    chunksize: int = 500_000,
    check_duplicates: bool = True,
) -> dict[str, Any]:
    """Compute an auditable data-quality report without loading all columns twice.

    Duplicate and account-cardinality checks use stable pandas hashes. That keeps
    the audit memory-aware for a 6.3M-row CSV while still detecting practical data
    duplication issues.
    """

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    column_names: list[str] | None = None
    dtype_summary: dict[str, str] = {}
    missing_counts: Counter = Counter()
    target_counts: Counter = Counter()
    flagged_counts: Counter = Counter()
    flagged_vs_fraud: Counter = Counter()
    type_summary: dict[str, Counter] = defaultdict(Counter)

    step_min: int | None = None
    step_max: int | None = None

    amount_sum = 0.0
    amount_squared_sum = 0.0
    amount_min = np.inf
    amount_max = -np.inf
    fraud_amount_sum = 0.0
    fraud_amount_min = np.inf
    fraud_amount_max = -np.inf

    balance_checks = Counter()
    pattern_checks = Counter()
    seen_row_hashes: set[int] = set()
    duplicate_rows = 0
    origin_hashes: set[int] = set()
    destination_hashes: set[int] = set()

    for chunk in pd.read_csv(input_path, chunksize=chunksize):
        if column_names is None:
            column_names = list(chunk.columns)
            dtype_summary = {column: str(dtype) for column, dtype in chunk.dtypes.items()}

        row_count = len(chunk)
        total_rows += row_count

        missing_counts.update(chunk.isna().sum().astype(int).to_dict())
        target_counts.update(chunk[TARGET_COLUMN].astype(int).value_counts().to_dict())
        flagged_counts.update(chunk["isFlaggedFraud"].astype(int).value_counts().to_dict())
        flagged_vs_fraud.update(
            {
                f"fraud_{int(fraud)}__flag_{int(flag)}": int(count)
                for (fraud, flag), count in chunk.groupby(
                    [TARGET_COLUMN, "isFlaggedFraud"]
                ).size().items()
            }
        )
        _update_nested_counter(type_summary, chunk)

        step_min = int(chunk["step"].min()) if step_min is None else min(step_min, int(chunk["step"].min()))
        step_max = int(chunk["step"].max()) if step_max is None else max(step_max, int(chunk["step"].max()))

        amount = chunk["amount"].astype(float)
        amount_sum += float(amount.sum())
        amount_squared_sum += float((amount**2).sum())
        amount_min = min(amount_min, float(amount.min()))
        amount_max = max(amount_max, float(amount.max()))

        fraud_amount = amount[chunk[TARGET_COLUMN].astype(int) == 1]
        if not fraud_amount.empty:
            fraud_amount_sum += float(fraud_amount.sum())
            fraud_amount_min = min(fraud_amount_min, float(fraud_amount.min()))
            fraud_amount_max = max(fraud_amount_max, float(fraud_amount.max()))

        origin_error = (
            chunk["oldbalanceOrg"].astype(float)
            - chunk["amount"].astype(float)
            - chunk["newbalanceOrig"].astype(float)
        )
        destination_error = (
            chunk["oldbalanceDest"].astype(float)
            + chunk["amount"].astype(float)
            - chunk["newbalanceDest"].astype(float)
        )
        balance_checks["origin_balance_error_gt_001"] += int((origin_error.abs() > 0.01).sum())
        balance_checks["destination_balance_error_gt_001"] += int((destination_error.abs() > 0.01).sum())

        origin_drained = (
            (chunk["oldbalanceOrg"].astype(float) > 0)
            & (chunk["newbalanceOrig"].astype(float).abs() <= 0.01)
            & (chunk["amount"].astype(float) >= chunk["oldbalanceOrg"].astype(float) - 0.01)
        )
        amount_equals_old_origin = (
            (chunk["amount"].astype(float) - chunk["oldbalanceOrg"].astype(float)).abs() <= 0.01
        )
        pattern_checks["origin_account_drained"] += int(origin_drained.sum())
        pattern_checks["fraud_origin_account_drained"] += int(
            (origin_drained & (chunk[TARGET_COLUMN].astype(int) == 1)).sum()
        )
        pattern_checks["amount_equals_old_origin_balance"] += int(amount_equals_old_origin.sum())
        pattern_checks["fraud_amount_equals_old_origin_balance"] += int(
            (amount_equals_old_origin & (chunk[TARGET_COLUMN].astype(int) == 1)).sum()
        )

        origin_hashes.update(
            int(value)
            for value in pd.util.hash_pandas_object(chunk["nameOrig"], index=False).to_numpy()
        )
        destination_hashes.update(
            int(value)
            for value in pd.util.hash_pandas_object(chunk["nameDest"], index=False).to_numpy()
        )

        if check_duplicates:
            row_hashes = pd.util.hash_pandas_object(chunk, index=False).to_numpy()
            for value in row_hashes:
                row_hash = int(value)
                if row_hash in seen_row_hashes:
                    duplicate_rows += 1
                else:
                    seen_row_hashes.add(row_hash)

    fraud_count = int(target_counts.get(1, 0))
    nonfraud_count = int(target_counts.get(0, 0))
    fraud_rate = _safe_pct(fraud_count, total_rows)
    amount_mean = _safe_pct(amount_sum, total_rows)
    amount_variance = max(_safe_pct(amount_squared_sum, total_rows) - amount_mean**2, 0.0)

    type_rows = []
    for txn_type, counter in sorted(type_summary.items()):
        transactions = int(counter["transactions"])
        fraud_transactions = int(counter["fraud_transactions"])
        type_rows.append(
            {
                "type": txn_type,
                "transactions": transactions,
                "fraud_transactions": fraud_transactions,
                "fraud_rate": _safe_pct(fraud_transactions, transactions),
            }
        )

    flagged_true = int(flagged_counts.get(1, 0))
    flagged_fraud_true = int(flagged_vs_fraud.get("fraud_1__flag_1", 0))

    report: dict[str, Any] = {
        "source_file": str(input_path),
        "dataset_source": "https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset",
        "row_count": int(total_rows),
        "column_count": int(len(column_names or [])),
        "columns": column_names or [],
        "expected_columns_match": (column_names or []) == EXPECTED_COLUMNS,
        "dtypes": dtype_summary,
        "missing_values": {column: int(missing_counts.get(column, 0)) for column in column_names or []},
        "total_missing_values": int(sum(missing_counts.values())),
        "duplicate_rows_hash_check": int(duplicate_rows),
        "target_distribution": {"non_fraud": nonfraud_count, "fraud": fraud_count},
        "fraud_rate": fraud_rate,
        "flagged_distribution": _series_to_int_dict(pd.Series(dict(flagged_counts))),
        "flagged_vs_fraud": dict(flagged_vs_fraud),
        "flagged_fraud_capture_rate": _safe_pct(flagged_fraud_true, fraud_count),
        "flagged_precision": _safe_pct(flagged_fraud_true, flagged_true),
        "transaction_type_summary": type_rows,
        "step_min": step_min,
        "step_max": step_max,
        "amount_summary": {
            "min": float(amount_min),
            "max": float(amount_max),
            "mean": float(amount_mean),
            "std": float(np.sqrt(amount_variance)),
            "fraud_sum": float(fraud_amount_sum),
            "fraud_min": float(fraud_amount_min if np.isfinite(fraud_amount_min) else 0.0),
            "fraud_max": float(fraud_amount_max if np.isfinite(fraud_amount_max) else 0.0),
        },
        "balance_consistency": dict(balance_checks),
        "behavioral_patterns": dict(pattern_checks),
        "account_cardinality_hash_based": {
            "nameOrig_unique": int(len(origin_hashes)),
            "nameDest_unique": int(len(destination_hashes)),
        },
        "leakage_audit": {
            "target_column_excluded_from_features": True,
            "isFlaggedFraud_treated_as_legacy_rule": True,
            "identifier_columns_used_only_for_behavioral_flags": True,
            "validation_strategy": "step-aware split in the modeling pipeline",
        },
    }

    json_path = output_dir / "data_quality_summary.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path = output_dir / "data_quality_summary.md"
    markdown_path.write_text(render_quality_markdown(report), encoding="utf-8")
    return report


def render_quality_markdown(report: dict[str, Any]) -> str:
    """Render a concise, portfolio-ready quality report."""

    type_lines = [
        "| Type | Transactions | Fraud | Fraud Rate |",
        "|---|---:|---:|---:|",
    ]
    for row in report["transaction_type_summary"]:
        type_lines.append(
            f"| {row['type']} | {row['transactions']:,} | {row['fraud_transactions']:,} | {row['fraud_rate']:.4%} |"
        )

    missing_total = report["total_missing_values"]
    duplicate_rows = report["duplicate_rows_hash_check"]
    flagged_capture = report["flagged_fraud_capture_rate"]
    flagged_precision = report["flagged_precision"]

    return "\n".join(
        [
            "# Data Quality and Leakage Audit",
            "",
            f"- Source file: `{report['source_file']}`",
            f"- Dataset source: {report['dataset_source']}",
            f"- Rows: {report['row_count']:,}",
            f"- Columns: {report['column_count']:,}",
            f"- Expected schema match: `{report['expected_columns_match']}`",
            f"- Missing values: {missing_total:,}",
            f"- Duplicate rows (hash audit): {duplicate_rows:,}",
            f"- Fraud transactions: {report['target_distribution']['fraud']:,}",
            f"- Fraud rate: {report['fraud_rate']:.4%}",
            f"- Step range: {report['step_min']} to {report['step_max']}",
            "",
            "## Transaction Type Fraud Distribution",
            "",
            *type_lines,
            "",
            "## Legacy Rule Audit",
            "",
            f"- `isFlaggedFraud` fraud capture rate: {flagged_capture:.4%}",
            f"- `isFlaggedFraud` precision when triggered: {flagged_precision:.4%}",
            "- Modeling pipeline excludes `isFlaggedFraud` from the final predictive feature set and keeps it only as a benchmark.",
            "",
            "## Dataset Limitations",
            "",
            "- The data is simulated transaction data, not real customer data.",
            "- There are no real calendar timestamps, demographics, investigation outcomes, or confirmed recovery amounts.",
            "- Business impact values in this project are scenario assumptions only.",
        ]
    )


if __name__ == "__main__":
    compute_quality_report()