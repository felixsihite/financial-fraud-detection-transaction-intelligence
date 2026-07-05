/*
Risk segment reporting after model scoring.

Assumed scored table: scored_transactions
Required scoring columns:
predicted_fraud_probability, fraud_risk_score, model_risk_segment,
investigation_priority_rank
*/

SELECT
    model_risk_segment,
    COUNT(*) AS transaction_count,
    MIN(predicted_fraud_probability) AS min_probability,
    AVG(predicted_fraud_probability) AS average_probability,
    MAX(predicted_fraud_probability) AS max_probability,
    SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) AS confirmed_fraud_count,
    1.0 * SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) / COUNT(*) AS observed_fraud_rate,
    SUM(amount) AS total_amount,
    SUM(CASE WHEN isFraud = 1 THEN amount ELSE 0 END) AS observed_fraud_amount
FROM scored_transactions
GROUP BY model_risk_segment
ORDER BY
    CASE model_risk_segment
        WHEN 'Critical' THEN 1
        WHEN 'High' THEN 2
        WHEN 'Medium' THEN 3
        WHEN 'Low' THEN 4
        ELSE 5
    END;

-- Review-capacity reporting for top-ranked transactions.
WITH ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (ORDER BY predicted_fraud_probability DESC) AS review_rank,
        COUNT(*) OVER () AS total_transactions,
        SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) OVER () AS total_fraud
    FROM scored_transactions
)
SELECT
    CASE
        WHEN review_rank <= total_transactions * 0.001 THEN 'Top 0.1%'
        WHEN review_rank <= total_transactions * 0.005 THEN 'Top 0.5%'
        WHEN review_rank <= total_transactions * 0.010 THEN 'Top 1.0%'
        WHEN review_rank <= total_transactions * 0.050 THEN 'Top 5.0%'
        ELSE 'Outside Top 5%'
    END AS review_band,
    COUNT(*) AS reviewed_transactions,
    SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) AS fraud_captured,
    1.0 * SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) / COUNT(*) AS precision_in_band,
    1.0 * SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) / MAX(total_fraud) AS recall_share
FROM ranked
GROUP BY review_band
ORDER BY MIN(review_rank);