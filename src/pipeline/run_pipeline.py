"""Run the full fraud detection portfolio pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import CHART_DIR, REPORT_DIR, RAW_DATA_PATH, ensure_project_directories
from src.data.data_quality import compute_quality_report
from src.models.train_model import train_and_evaluate
from src.visualization.theme import APPROVED_PALETTE, set_plot_theme


def _plot_data_quality_charts() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    report = json.loads((REPORT_DIR / "data_quality_summary.json").read_text(encoding="utf-8"))
    type_summary = pd.DataFrame(report["transaction_type_summary"])
    type_summary["fraud_rate_pct"] = type_summary["fraud_rate"] * 100

    plt.figure(figsize=(9.2, 5.4))
    sns.barplot(data=type_summary, x="type", y="transactions", color=APPROVED_PALETTE["accent_blue"])
    plt.title("Transaction Volume by Type")
    plt.xlabel("")
    plt.ylabel("Transactions")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "transaction_volume_by_type.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9.2, 5.4))
    sns.barplot(data=type_summary, x="type", y="fraud_rate_pct", color=APPROVED_PALETTE["risk_red"])
    plt.title("Fraud Rate by Transaction Type")
    plt.xlabel("")
    plt.ylabel("Fraud Rate (%)")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "fraud_rate_by_type.png", dpi=180)
    plt.close()


def run_pipeline(
    raw_data_path: str | Path = RAW_DATA_PATH,
    max_train_nonfraud: int = 220_000,
    chunksize: int = 500_000,
) -> dict:
    """Execute audit, modeling, reporting, and chart generation."""

    ensure_project_directories()
    set_plot_theme()
    quality = compute_quality_report(raw_data_path, REPORT_DIR, chunksize=chunksize)
    _plot_data_quality_charts()
    model_summary = train_and_evaluate(
        input_path=raw_data_path,
        max_train_nonfraud=max_train_nonfraud,
        chunksize=chunksize,
    )
    return {"quality": quality, "model": model_summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fraud detection portfolio pipeline.")
    parser.add_argument("--raw-data-path", default=str(RAW_DATA_PATH))
    parser.add_argument("--max-train-nonfraud", type=int, default=220_000)
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()
    summary = run_pipeline(
        raw_data_path=args.raw_data_path,
        max_train_nonfraud=args.max_train_nonfraud,
        chunksize=args.chunksize,
    )
    print(
        json.dumps(
            {
                "rows": summary["quality"]["row_count"],
                "fraud_rate": summary["quality"]["fraud_rate"],
                "best_model": summary["model"]["best_model"],
                "pr_auc": summary["model"]["best_model_metrics"]["pr_auc"],
                "recall": summary["model"]["best_model_metrics"]["recall"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()