"""Executive dashboard for financial fraud detection and transaction intelligence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.theme import APPROVED_PALETTE, STREAMLIT_CSS  # noqa: E402


REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
CHART_DIR = PROJECT_ROOT / "outputs" / "charts"
RISK_DIR = PROJECT_ROOT / "outputs" / "risk_scores"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


st.set_page_config(
    page_title="Fraud Detection & Transaction Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(STREAMLIT_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def require_outputs() -> tuple[dict, dict]:
    quality_path = REPORT_DIR / "data_quality_summary.json"
    model_path = REPORT_DIR / "model_performance_summary.json"
    missing = [str(path) for path in [quality_path, model_path] if not path.exists()]
    if missing:
        st.error("Pipeline output is missing. Run `python run_pipeline.py` once from the project root.")
        st.stop()
    return load_json(quality_path), load_json(model_path)


def money_short(value: float) -> str:
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:,.0f}"


def count_fmt(value: float | int) -> str:
    return f"{int(value):,}"


def pct_fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}%}"


def page_header(kicker: str, title: str, summary: str) -> None:
    st.markdown(f"<div class='page-kicker'>{kicker}</div>", unsafe_allow_html=True)
    st.title(title)
    st.markdown(f"<div class='status-band'>{summary}</div>", unsafe_allow_html=True)


def section_title(text: str) -> None:
    st.markdown(f"<div class='section-title'>{text}</div>", unsafe_allow_html=True)


def metric_card(label: str, value: str, note: str = "") -> None:
    note_html = f"<div class='metric-note'>{note}</div>" if note else ""
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_plot(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor=APPROVED_PALETTE["card_background"],
        plot_bgcolor=APPROVED_PALETTE["card_background"],
        font=dict(color=APPROVED_PALETTE["primary_text"], size=12),
        title=dict(font=dict(size=15, color=APPROVED_PALETTE["primary_text"])),
        margin=dict(l=42, r=22, t=58, b=42),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor="#A9BFD3",
        tickfont=dict(color=APPROVED_PALETTE["secondary_text"]),
        title_font=dict(color=APPROVED_PALETTE["secondary_text"]),
    )
    fig.update_yaxes(
        gridcolor="#D8E5EF",
        zerolinecolor="#C9D8E8",
        tickfont=dict(color=APPROVED_PALETTE["secondary_text"]),
        title_font=dict(color=APPROVED_PALETTE["secondary_text"]),
    )
    return fig


def display_table(frame: pd.DataFrame, height: int | None = None) -> None:
    options = {"width": "stretch", "hide_index": True}
    if height is not None:
        options["height"] = height
    st.dataframe(frame, **options)


def type_summary_frame(quality: dict) -> pd.DataFrame:
    frame = pd.DataFrame(quality["transaction_type_summary"])
    frame["fraud_rate_pct"] = frame["fraud_rate"] * 100
    return frame


def page_executive(quality: dict, model: dict) -> None:
    metrics = model["best_model_metrics"]
    top_k = pd.DataFrame(model["top_k_capture"])
    review_queue = metrics["true_positives"] + metrics["false_positives"]

    page_header(
        "Executive Overview",
        "Financial Fraud Detection & Transaction Intelligence",
        "Rare-event fraud scoring case study with leakage controls, time-aware validation, risk ranking, and investigator-ready outputs.",
    )

    cols = st.columns(2)
    with cols[0]:
        metric_card("Total Transactions", count_fmt(quality["row_count"]), "Raw dataset rows")
    with cols[1]:
        metric_card("Fraud Transactions", count_fmt(quality["target_distribution"]["fraud"]), "Confirmed labels")
    cols = st.columns(2)
    with cols[0]:
        metric_card("Fraud Rate", pct_fmt(quality["fraud_rate"], 3), "Severe class imbalance")
    with cols[1]:
        metric_card("Simulated Fraud Amount", money_short(quality["amount_summary"]["fraud_sum"]), "No real recovery claim")

    cols = st.columns(2)
    with cols[0]:
        metric_card("Final Model", model["best_model"], "Legacy rule excluded")
    with cols[1]:
        metric_card("PR-AUC", f"{metrics['pr_auc']:.4f}", "Primary rare-event metric")
    cols = st.columns(2)
    with cols[0]:
        metric_card("Fraud Recall", pct_fmt(metrics["recall"], 2), "Optimized threshold")
    with cols[1]:
        metric_card("Review Queue", count_fmt(review_queue), f"Threshold {metrics['threshold']:.2f}")

    section_title("What Matters Operationally")
    cols = st.columns(1)
    with cols[0]:
        st.markdown(
            "<div class='insight-box'><b>Fraud concentration:</b><br>Fraud appears only in TRANSFER and CASH_OUT transaction types.</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        f"<div class='insight-box'><b>Legacy rule gap:</b><br><code>isFlaggedFraud</code> captures only {pct_fmt(quality['flagged_fraud_capture_rate'], 3)} of fraud cases.</div>",
        unsafe_allow_html=True,
    )
    top_05 = top_k.loc[top_k["review_rate"].round(3) == 0.005].iloc[0]
    st.markdown(
        f"<div class='insight-box'><b>Investigation leverage:</b><br>Top 0.5% ranked transactions capture {pct_fmt(top_05['recall_at_k'], 2)} of validation fraud.</div>",
        unsafe_allow_html=True,
    )

    section_title("Investigation Capacity Curve")
    cap = top_k.copy()
    cap["review_capacity"] = cap["review_rate"].map(lambda value: f"Top {value:.1%}")
    fig = go.Figure()
    fig.add_bar(
        x=cap["review_capacity"],
        y=cap["recall_at_k"] * 100,
        name="Fraud Recall",
        marker_color=APPROVED_PALETTE["accent_teal"],
        text=(cap["recall_at_k"] * 100).round(1).astype(str) + "%",
        textposition="outside",
    )
    fig.update_yaxes(title="Fraud Recall (%)", range=[0, 110])
    fig.update_xaxes(title="")
    fig.update_layout(title="Fraud Captured by Review Capacity")
    st.plotly_chart(style_plot(fig, height=350), width="stretch", config={"displayModeBar": False})


def page_transaction_intelligence(quality: dict) -> None:
    page_header(
        "Transaction Intelligence",
        "Transaction Behavior & Fraud Concentration",
        "Type-level volumes, fraud rate concentration, and amount behavior using validation scoring outputs.",
    )
    summary = type_summary_frame(quality)

    fig = px.bar(
        summary,
        x="type",
        y="transactions",
        text="transactions",
        color_discrete_sequence=[APPROVED_PALETTE["accent_blue"]],
        title="Transaction Volume by Type",
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig.update_yaxes(title="Transactions")
    fig.update_xaxes(title="")
    st.plotly_chart(style_plot(fig, height=390), width="stretch", config={"displayModeBar": False})

    fig = px.bar(
        summary,
        x="type",
        y="fraud_rate_pct",
        text=summary["fraud_rate_pct"].map(lambda value: f"{value:.3f}%"),
        color_discrete_sequence=[APPROVED_PALETTE["risk_red"]],
        title="Fraud Rate by Transaction Type",
    )
    fig.update_traces(textposition="outside")
    fig.update_yaxes(title="Fraud Rate (%)")
    fig.update_xaxes(title="")
    st.plotly_chart(style_plot(fig, height=390), width="stretch", config={"displayModeBar": False})

    section_title("Type-Level Summary")
    table = summary[["type", "transactions", "fraud_transactions", "fraud_rate"]].copy()
    table["fraud_rate"] = table["fraud_rate"].map(lambda value: f"{value:.4%}")
    table.columns = ["Transaction Type", "Transactions", "Fraud Transactions", "Fraud Rate"]
    display_table(table)

    sample_path = PROCESSED_DIR / "validation_scored_sample.csv"
    if sample_path.exists():
        sample = load_csv(sample_path)
        sample["amount_log10"] = np.log10(sample["amount"].clip(lower=0) + 1)
        section_title("Validation Amount Profile")
        fig = px.histogram(
            sample,
            x="amount_log10",
            color="type",
            nbins=48,
            title="Amount Distribution by Transaction Type",
            color_discrete_sequence=[
                APPROVED_PALETTE["accent_blue"],
                APPROVED_PALETTE["accent_teal"],
                APPROVED_PALETTE["warning_amber"],
                APPROVED_PALETTE["positive_green"],
                APPROVED_PALETTE["risk_red"],
            ],
        )
        fig.update_xaxes(title="log10(amount + 1)")
        fig.update_yaxes(title="Transactions")
        st.plotly_chart(style_plot(fig, height=420), width="stretch", config={"displayModeBar": False})
        amount_table = (
            sample.groupby("type")["amount"]
            .agg(["count", "median", "mean", "max"])
            .reset_index()
            .rename(
                columns={
                    "type": "Type",
                    "count": "Rows",
                    "median": "Median Amount",
                    "mean": "Mean Amount",
                    "max": "Max Amount",
                }
            )
        )
        display_table(amount_table.round(2), height=240)


def page_fraud_patterns(quality: dict) -> None:
    page_header(
        "Fraud Pattern Analysis",
        "Behavioral Signals Behind Fraud",
        "Balance movement and account-drain patterns explain why the simulated fraud label is separable.",
    )
    patterns = quality["behavioral_patterns"]
    balance = quality["balance_consistency"]

    cols = st.columns(2)
    with cols[0]:
        metric_card("Origin Drained", count_fmt(patterns["origin_account_drained"]), "All transactions")
    with cols[1]:
        metric_card("Fraud Origin Drained", count_fmt(patterns["fraud_origin_account_drained"]), "Fraud subset")
    cols = st.columns(2)
    with cols[0]:
        metric_card("Amount = Old Origin", count_fmt(patterns["amount_equals_old_origin_balance"]), "All transactions")
    with cols[1]:
        metric_card("Fraud Match", count_fmt(patterns["fraud_amount_equals_old_origin_balance"]), "Fraud subset")

    audit_frame = pd.DataFrame(
        [
            {"Check": "Origin balance error > 0.01", "Transactions": balance["origin_balance_error_gt_001"]},
            {
                "Check": "Destination balance error > 0.01",
                "Transactions": balance["destination_balance_error_gt_001"],
            },
            {"Check": "Origin account drained", "Transactions": patterns["origin_account_drained"]},
            {
                "Check": "Amount equals old origin balance",
                "Transactions": patterns["amount_equals_old_origin_balance"],
            },
        ]
    )
    fig = px.bar(
        audit_frame,
        x="Transactions",
        y="Check",
        orientation="h",
        text="Transactions",
        color_discrete_sequence=[APPROVED_PALETTE["warning_amber"]],
        title="Balance and Behavioral Pattern Counts",
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig.update_xaxes(title="Transactions")
    fig.update_yaxes(title="")
    st.plotly_chart(style_plot(fig, height=420), width="stretch", config={"displayModeBar": False})
    st.markdown(
        """
        <div class='insight-box'>
          <b>Interpretation</b><br>
          The strongest patterns are transaction-behavior signals, not customer identity signals. The model avoids customer-ID encoding and relies on balance movement, amount behavior, transaction type, and sequential step features.
        </div>
        """,
        unsafe_allow_html=True,
    )
    legacy = pd.DataFrame(
        [
            {"Metric": "Legacy flags triggered", "Value": quality["flagged_distribution"].get("1", 0)},
            {"Metric": "Fraud captured by legacy flag", "Value": quality["flagged_vs_fraud"].get("fraud_1__flag_1", 0)},
            {"Metric": "Fraud missed by legacy flag", "Value": quality["flagged_vs_fraud"].get("fraud_1__flag_0", 0)},
        ]
    )
    display_table(legacy, height=180)


def page_model_performance(model: dict) -> None:
    page_header(
        "Model Performance",
        "Rare-Event Model Evaluation",
        "Evaluation is driven by PR-AUC, fraud recall, precision, F2, threshold tuning, and top-ranked investigation capture.",
    )
    metrics = model["best_model_metrics"]
    comparison = load_csv(REPORT_DIR / "model_comparison.csv")
    threshold = load_csv(REPORT_DIR / "threshold_optimization_top50.csv")
    top_k = load_csv(REPORT_DIR / "top_k_fraud_capture.csv")

    cols = st.columns(2)
    with cols[0]:
        metric_card("PR-AUC", f"{metrics['pr_auc']:.4f}", "Primary metric")
    with cols[1]:
        metric_card("ROC-AUC", f"{metrics['roc_auc']:.4f}", "Secondary")
    cols = st.columns(2)
    with cols[0]:
        metric_card("Precision", pct_fmt(metrics["precision"], 2), "Fraud class")
    with cols[1]:
        metric_card("Recall", pct_fmt(metrics["recall"], 2), "Fraud class")
    cols = st.columns(2)
    with cols[0]:
        metric_card("False Negatives", count_fmt(metrics["false_negatives"]), "Validation")
    with cols[1]:
        metric_card("False Positives", count_fmt(metrics["false_positives"]), "Validation")

    section_title("Model Comparison")
    compare_plot = comparison[comparison["feature_set"] == "final_no_legacy_flag"].copy()
    compare_plot["Model"] = compare_plot["model_name"]
    fig = px.bar(
        compare_plot.sort_values("pr_auc"),
        x="pr_auc",
        y="Model",
        orientation="h",
        text=compare_plot.sort_values("pr_auc")["pr_auc"].map(lambda value: f"{value:.4f}"),
        color_discrete_sequence=[APPROVED_PALETTE["accent_blue"]],
        title="PR-AUC by Candidate Model",
    )
    fig.update_traces(textposition="outside")
    fig.update_xaxes(title="PR-AUC", range=[0, 1.05])
    fig.update_yaxes(title="")
    st.plotly_chart(style_plot(fig, height=360), width="stretch", config={"displayModeBar": False})

    sorted_threshold = threshold.sort_values("threshold")
    fig = go.Figure()
    fig.add_scatter(x=sorted_threshold["threshold"], y=sorted_threshold["precision"], mode="lines", name="Precision", line=dict(color=APPROVED_PALETTE["accent_blue"], width=3))
    fig.add_scatter(x=sorted_threshold["threshold"], y=sorted_threshold["recall"], mode="lines", name="Recall", line=dict(color=APPROVED_PALETTE["risk_red"], width=3))
    fig.add_scatter(x=sorted_threshold["threshold"], y=sorted_threshold["f2"], mode="lines", name="F2", line=dict(color=APPROVED_PALETTE["accent_teal"], width=3))
    fig.update_layout(title="Threshold Tradeoff")
    fig.update_xaxes(title="Threshold")
    fig.update_yaxes(title="Score", range=[0, 1.05])
    st.plotly_chart(style_plot(fig, height=380), width="stretch", config={"displayModeBar": False})

    top_k_plot = top_k.copy()
    top_k_plot["Capacity"] = top_k_plot["review_rate"].map(lambda value: f"Top {value:.1%}")
    fig = px.bar(
        top_k_plot,
        x="Capacity",
        y="recall_at_k",
        text=top_k_plot["recall_at_k"].map(lambda value: f"{value:.1%}"),
        color_discrete_sequence=[APPROVED_PALETTE["accent_teal"]],
        title="Recall Captured by Review Capacity",
    )
    fig.update_traces(textposition="outside")
    fig.update_yaxes(title="Recall", tickformat=".0%", range=[0, 1.1])
    fig.update_xaxes(title="")
    st.plotly_chart(style_plot(fig, height=380), width="stretch", config={"displayModeBar": False})

    section_title("Threshold Table")
    threshold_display = threshold[
        [
            "threshold",
            "precision",
            "recall",
            "f2",
            "false_positives",
            "false_negatives",
            "reviewed_transactions",
            "review_rate",
        ]
    ].head(12)
    display_table(threshold_display.round(6), height=360)


def page_risk_scoring() -> None:
    page_header(
        "Fraud Risk Scoring",
        "Investigation Priority Queue",
        "Model probabilities are converted into risk scores and ranked for analyst review capacity.",
    )
    queue_path = RISK_DIR / "investigation_priority_top_1000.csv"
    if not queue_path.exists():
        st.error("Risk-score output is missing. Run `python run_pipeline.py` once from the project root.")
        st.stop()

    queue = load_csv(queue_path)
    segment_counts = (
        queue["model_risk_segment"].value_counts().rename_axis("Risk Segment").reset_index(name="Transactions")
    )
    cols = st.columns(2)
    with cols[0]:
        metric_card("Top Queue Size", count_fmt(len(queue)), "Exported transactions")
    with cols[1]:
        metric_card("Max Risk Score", f"{queue['fraud_risk_score'].max():,.1f}", "Scale 0-1000")
    cols = st.columns(2)
    with cols[0]:
        metric_card("Median Amount", money_short(queue["amount"].median()), "Top queue")
    with cols[1]:
        metric_card("Legacy Flagged", count_fmt(queue["isFlaggedFraud"].sum()), "Within top queue")

    st.download_button(
        "Download Top 1,000 Investigation Queue",
        data=queue.to_csv(index=False).encode("utf-8"),
        file_name="investigation_priority_top_1000.csv",
        mime="text/csv",
        width="stretch",
    )

    fig = px.bar(
        segment_counts,
        x="Risk Segment",
        y="Transactions",
        text="Transactions",
        color="Risk Segment",
        color_discrete_map={
            "Critical": APPROVED_PALETTE["critical_red"],
            "High": APPROVED_PALETTE["risk_red"],
            "Medium": APPROVED_PALETTE["warning_amber"],
            "Low": APPROVED_PALETTE["positive_green"],
        },
        title="Risk Segment Mix",
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig.update_yaxes(title="Transactions")
    fig.update_xaxes(title="")
    st.plotly_chart(style_plot(fig, height=390), width="stretch", config={"displayModeBar": False})

    display_cols = [
        "investigation_priority_rank",
        "step",
        "type",
        "amount",
        "predicted_fraud_probability",
        "fraud_risk_score",
        "model_risk_segment",
        "isFlaggedFraud",
        "isFraud",
    ]
    table = queue[display_cols].head(100).copy()
    table.columns = [
        "Rank",
        "Step",
        "Type",
        "Amount",
        "Fraud Probability",
        "Risk Score",
        "Risk Segment",
        "Legacy Flag",
        "Validation Label",
    ]
    display_table(table, height=420)


def page_explainable_ai() -> None:
    page_header(
        "Explainable AI",
        "Model Drivers & Investigation Trust",
        "Final model explanations are based on primitive transaction features. The composite rule score is excluded from the ML feature set.",
    )
    importance = load_csv(REPORT_DIR / "permutation_importance.csv")
    native = load_csv(REPORT_DIR / "model_native_feature_importance.csv")
    notes = load_json(REPORT_DIR / "interpretability_notes.json")

    plot_frame = importance.head(12).sort_values("permutation_importance_mean")
    fig = px.bar(
        plot_frame,
        x="permutation_importance_mean",
        y="feature",
        orientation="h",
        color_discrete_sequence=[APPROVED_PALETTE["accent_teal"]],
        title="Permutation Importance",
    )
    fig.update_xaxes(title="Mean PR-AUC Impact")
    fig.update_yaxes(title="")
    st.plotly_chart(style_plot(fig, height=430), width="stretch", config={"displayModeBar": False})

    plot_frame = native.head(12).sort_values("importance")
    fig = px.bar(
        plot_frame,
        x="importance",
        y="feature",
        orientation="h",
        color_discrete_sequence=[APPROVED_PALETTE["accent_blue"]],
        title="Model-Native Importance",
    )
    fig.update_xaxes(title="Normalized Importance")
    fig.update_yaxes(title="")
    st.plotly_chart(style_plot(fig, height=430), width="stretch", config={"displayModeBar": False})

    cols = st.columns(2)
    with cols[0]:
        section_title("Top Permutation Drivers")
        display_table(importance.head(12).round(8), height=360)
    with cols[1]:
        section_title("Top Native Drivers")
        display_table(native.head(12).round(8), height=360)

    st.markdown(
        f"<div class='insight-box'><b>Method:</b> {notes['method']}<br><b>Optional extension:</b> {notes['shap_status']}</div>",
        unsafe_allow_html=True,
    )


quality_report, model_report = require_outputs()

PAGE_LABELS = [
    "Executive Fraud Overview",
    "Transaction Intelligence",
    "Fraud Pattern Analysis",
    "Model Performance",
    "Fraud Risk Scoring",
    "Explainable AI",
]
PAGE_SLUGS = {
    "Executive Fraud Overview": "executive-fraud-overview",
    "Transaction Intelligence": "transaction-intelligence",
    "Fraud Pattern Analysis": "fraud-pattern-analysis",
    "Model Performance": "model-performance",
    "Fraud Risk Scoring": "fraud-risk-scoring",
    "Explainable AI": "explainable-ai",
}
SLUG_TO_PAGE = {slug: label for label, slug in PAGE_SLUGS.items()}
requested_slug = st.query_params.get("page", PAGE_SLUGS["Executive Fraud Overview"])
default_page = SLUG_TO_PAGE.get(requested_slug, "Executive Fraud Overview")

with st.sidebar:
    st.title("Fraud Intelligence")
    st.caption("Transaction risk analytics")
    selected_page = st.radio(
        "Navigation",
        PAGE_LABELS,
        index=PAGE_LABELS.index(default_page),
    )
    st.markdown("---")
    st.caption("Dataset: Kaggle simulated financial transactions")

if selected_page == "Executive Fraud Overview":
    page_executive(quality_report, model_report)
elif selected_page == "Transaction Intelligence":
    page_transaction_intelligence(quality_report)
elif selected_page == "Fraud Pattern Analysis":
    page_fraud_patterns(quality_report)
elif selected_page == "Model Performance":
    page_model_performance(model_report)
elif selected_page == "Fraud Risk Scoring":
    page_risk_scoring()
else:
    page_explainable_ai()