/*
Financial Fraud Detection & Transaction Intelligence
Transaction summary queries

Assumed source table: transactions
Columns follow the raw dataset schema:
step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
*/

-- Transaction volume, value, and fraud rate by transaction type.
SELECT
    type AS transaction_type,
    COUNT(*) AS transaction_count,
    SUM(amount) AS total_amount,
    AVG(amount) AS average_amount,
    SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) AS fraud_transactions,
    SUM(CASE WHEN isFraud = 1 THEN amount ELSE 0 END) AS fraud_amount,
    1.0 * SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) / COUNT(*) AS fraud_rate
FROM transactions
GROUP BY type
ORDER BY fraud_rate DESC, transaction_count DESC;

-- Sequential step-level monitoring summary.
SELECT
    step,
    COUNT(*) AS transaction_count,
    SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) AS fraud_transactions,
    SUM(CASE WHEN isFraud = 1 THEN amount ELSE 0 END) AS fraud_amount,
    1.0 * SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) / COUNT(*) AS fraud_rate
FROM transactions
GROUP BY step
ORDER BY step;