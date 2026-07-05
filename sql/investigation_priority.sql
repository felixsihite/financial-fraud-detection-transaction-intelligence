/*
Investigation priority queue for fraud operations.
This query is intended for scored validation or production-like demo data.
*/

SELECT
    investigation_priority_rank,
    step,
    type AS transaction_type,
    amount,
    nameOrig AS origin_account,
    oldbalanceOrg AS origin_balance_before,
    newbalanceOrig AS origin_balance_after,
    nameDest AS destination_account,
    oldbalanceDest AS destination_balance_before,
    newbalanceDest AS destination_balance_after,
    predicted_fraud_probability,
    fraud_risk_score,
    model_risk_segment,
    isFlaggedFraud AS legacy_rule_flag,
    isFraud AS validation_label
FROM scored_transactions
WHERE model_risk_segment IN ('Critical', 'High')
ORDER BY investigation_priority_rank
FETCH FIRST 1000 ROWS ONLY;

-- High-risk transactions not captured by the legacy rule.
SELECT
    investigation_priority_rank,
    step,
    type AS transaction_type,
    amount,
    predicted_fraud_probability,
    fraud_risk_score,
    model_risk_segment,
    isFlaggedFraud,
    isFraud
FROM scored_transactions
WHERE isFlaggedFraud = 0
  AND model_risk_segment IN ('Critical', 'High')
ORDER BY predicted_fraud_probability DESC
FETCH FIRST 1000 ROWS ONLY;