from src.config import TARGET_COLUMN
from src.features.feature_engineering import BASE_FEATURES, build_features, fit_feature_params


def test_feature_engineering_excludes_target_identifiers_and_legacy_flag(sample_transactions):
    params = fit_feature_params(sample_transactions)
    engineered, features = build_features(sample_transactions, params=params, include_flagged=False)

    assert TARGET_COLUMN not in features
    assert "nameOrig" not in features
    assert "nameDest" not in features
    assert "isFlaggedFraud" not in features
    assert "risk_rule_score" not in features
    assert engineered[features].isna().sum().sum() == 0


def test_fraud_behavior_flags_and_business_rule_score_are_created(sample_transactions):
    params = fit_feature_params(sample_transactions)
    engineered, _ = build_features(sample_transactions, params=params, include_flagged=False)

    assert engineered.loc[0, "is_origin_account_drained"] == 1
    assert engineered.loc[0, "amount_equals_old_origin_balance"] == 1
    assert engineered.loc[0, "transfer_or_cashout_flag"] == 1
    assert engineered.loc[0, "risk_rule_score"] > 0


def test_optional_legacy_flag_benchmark_can_be_requested(sample_transactions):
    params = fit_feature_params(sample_transactions)
    _, features = build_features(sample_transactions, params=params, include_flagged=True)

    assert "isFlaggedFraud" in features
    assert BASE_FEATURES == [feature for feature in features if feature != "isFlaggedFraud"]