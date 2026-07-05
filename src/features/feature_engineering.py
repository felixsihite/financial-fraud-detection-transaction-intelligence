"""Leakage-aware transaction feature engineering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import LEGACY_FLAG_COLUMN, TRANSACTION_TYPES


BASE_FEATURES = [
    "step",
    "amount",
    "transaction_type_encoded",
    "amount_log",
    "amount_percentile",
    "origin_balance_delta",
    "destination_balance_delta",
    "origin_balance_error",
    "destination_balance_error",
    "is_origin_account_drained",
    "is_destination_zero_before",
    "is_destination_zero_after",
    "amount_equals_old_origin_balance",
    "origin_balance_ratio",
    "destination_balance_ratio",
    "high_amount_flag",
    "transfer_or_cashout_flag",
    "customer_to_customer_flag",
    "customer_to_merchant_flag",
    "step_hour_bucket",
    "step_day_simulation",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
]


TYPE_ENCODING = {name: index for index, name in enumerate(TRANSACTION_TYPES)}


@dataclass(frozen=True)
class FeatureParams:
    """Training-set parameters used to transform future data consistently."""

    amount_high_threshold: float
    amount_quantile_edges: tuple[float, ...]


def fit_feature_params(frame: pd.DataFrame) -> FeatureParams:
    """Fit unsupervised transformation parameters on the training frame only."""

    amount = frame["amount"].astype(float)
    quantile_grid = np.linspace(0.0, 1.0, 101)
    edges = np.quantile(amount, quantile_grid)
    edges = np.maximum.accumulate(edges)
    return FeatureParams(
        amount_high_threshold=float(amount.quantile(0.95)),
        amount_quantile_edges=tuple(float(value) for value in edges),
    )


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.astype(float)
    ratio = np.divide(
        numerator.astype(float),
        denominator,
        out=np.zeros(len(numerator), dtype=float),
        where=denominator.abs().to_numpy() > 1e-9,
    )
    return pd.Series(np.clip(ratio, -10.0, 10.0), index=numerator.index)


def _amount_percentile(amount: pd.Series, params: FeatureParams) -> pd.Series:
    edges = np.asarray(params.amount_quantile_edges, dtype=float)
    bins = np.searchsorted(edges, amount.astype(float).to_numpy(), side="right") - 1
    return pd.Series(np.clip(bins / 100.0, 0.0, 1.0), index=amount.index)


def build_features(
    frame: pd.DataFrame,
    params: FeatureParams | None = None,
    include_flagged: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Create professional fraud features while excluding target leakage.

    `nameOrig` and `nameDest` are not used as high-cardinality categoricals. They
    only support coarse behavioral flags such as customer-to-merchant movement.
    """

    engineered = frame.copy()
    if params is None:
        params = fit_feature_params(engineered)

    amount = engineered["amount"].astype(float)
    old_origin = engineered["oldbalanceOrg"].astype(float)
    new_origin = engineered["newbalanceOrig"].astype(float)
    old_destination = engineered["oldbalanceDest"].astype(float)
    new_destination = engineered["newbalanceDest"].astype(float)

    engineered["transaction_type_encoded"] = (
        engineered["type"].map(TYPE_ENCODING).fillna(-1).astype(int)
    )
    engineered["amount_log"] = np.log1p(amount)
    engineered["amount_percentile"] = _amount_percentile(amount, params)
    engineered["origin_balance_delta"] = old_origin - new_origin
    engineered["destination_balance_delta"] = new_destination - old_destination
    engineered["origin_balance_error"] = (old_origin - amount - new_origin).abs()
    engineered["destination_balance_error"] = (old_destination + amount - new_destination).abs()
    engineered["is_origin_account_drained"] = (
        (old_origin > 0) & (new_origin.abs() <= 0.01) & (amount >= old_origin - 0.01)
    ).astype(int)
    engineered["is_destination_zero_before"] = (old_destination.abs() <= 0.01).astype(int)
    engineered["is_destination_zero_after"] = (new_destination.abs() <= 0.01).astype(int)
    engineered["amount_equals_old_origin_balance"] = ((amount - old_origin).abs() <= 0.01).astype(int)
    engineered["origin_balance_ratio"] = _safe_ratio(amount, old_origin)
    engineered["destination_balance_ratio"] = _safe_ratio(amount, old_destination)
    engineered["high_amount_flag"] = (amount >= params.amount_high_threshold).astype(int)
    engineered["transfer_or_cashout_flag"] = engineered["type"].isin(["TRANSFER", "CASH_OUT"]).astype(int)
    engineered["customer_to_customer_flag"] = (
        engineered["nameOrig"].astype(str).str.startswith("C")
        & engineered["nameDest"].astype(str).str.startswith("C")
    ).astype(int)
    engineered["customer_to_merchant_flag"] = (
        engineered["nameOrig"].astype(str).str.startswith("C")
        & engineered["nameDest"].astype(str).str.startswith("M")
    ).astype(int)
    engineered["step_hour_bucket"] = (engineered["step"].astype(int) % 24).astype(int)
    engineered["step_day_simulation"] = (engineered["step"].astype(int) // 24).astype(int)

    type_dummies = pd.get_dummies(engineered["type"], prefix="type", dtype=int)
    for txn_type in TRANSACTION_TYPES:
        column = f"type_{txn_type}"
        engineered[column] = type_dummies[column] if column in type_dummies else 0

    engineered["risk_rule_score"] = (
        28 * engineered["transfer_or_cashout_flag"]
        + 24 * engineered["is_origin_account_drained"]
        + 18 * engineered["amount_equals_old_origin_balance"]
        + 12 * engineered["high_amount_flag"]
        + 10 * engineered["is_destination_zero_before"]
        + 8 * (engineered["origin_balance_error"] > 0.01).astype(int)
    ).clip(0, 100)
    engineered["fraud_risk_segment"] = pd.cut(
        engineered["risk_rule_score"],
        bins=[-0.1, 24, 49, 74, 100],
        labels=["Low", "Medium", "High", "Critical"],
    ).astype(str)

    feature_columns = list(BASE_FEATURES)
    if include_flagged:
        engineered[LEGACY_FLAG_COLUMN] = engineered[LEGACY_FLAG_COLUMN].astype(int)
        feature_columns.append(LEGACY_FLAG_COLUMN)

    engineered[feature_columns] = engineered[feature_columns].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return engineered, feature_columns


def align_feature_columns(frame: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    """Return a numeric matrix with stable feature order."""

    matrix = frame.copy()
    for column in feature_columns:
        if column not in matrix.columns:
            matrix[column] = 0
    return matrix[list(feature_columns)].astype(float)