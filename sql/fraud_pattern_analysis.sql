/*
Fraud pattern analysis queries.
Use these queries to validate transaction behaviors before model scoring.
*/

-- Fraud occurs in this dataset only for TRANSFER and CASH_OUT.
SELECT
    type AS transaction_type,
    isFraud,
    COUNT(*) AS transaction_count,
    SUM(amount) AS total_amount,
    AVG(amount) AS average_amount
FROM transactions
GROUP BY type, isFraud
ORDER BY isFraud DESC, transaction_count DESC;

-- Existing system rule flag effectiveness.
SELECT
    isFlaggedFraud,
    isFraud,
    COUNT(*) AS transaction_count,
    SUM(amount) AS total_amount
FROM transactions
GROUP BY isFlaggedFraud, isFraud
ORDER BY isFlaggedFraud DESC, isFraud DESC;

-- Origin account drained pattern.
SELECT
    type AS transaction_type,
    COUNT(*) AS transactions,
    SUM(CASE WHEN oldbalanceOrg > 0 AND newbalanceOrig = 0 AND amount >= oldbalanceOrg THEN 1 ELSE 0 END)
        AS origin_account_drained_count,
    SUM(CASE WHEN isFraud = 1 AND oldbalanceOrg > 0 AND newbalanceOrig = 0 AND amount >= oldbalanceOrg THEN 1 ELSE 0 END)
        AS fraud_origin_account_drained_count
FROM transactions
GROUP BY type
ORDER BY fraud_origin_account_drained_count DESC;

-- Amount equals old origin balance pattern.
SELECT
    type AS transaction_type,
    COUNT(*) AS transactions,
    SUM(CASE WHEN ABS(amount - oldbalanceOrg) <= 0.01 THEN 1 ELSE 0 END)
        AS amount_equals_old_origin_count,
    SUM(CASE WHEN isFraud = 1 AND ABS(amount - oldbalanceOrg) <= 0.01 THEN 1 ELSE 0 END)
        AS fraud_amount_equals_old_origin_count
FROM transactions
GROUP BY type
ORDER BY fraud_amount_equals_old_origin_count DESC;

-- Balance movement error audit.
SELECT
    type AS transaction_type,
    COUNT(*) AS transactions,
    SUM(CASE WHEN ABS(oldbalanceOrg - amount - newbalanceOrig) > 0.01 THEN 1 ELSE 0 END)
        AS origin_balance_error_count,
    SUM(CASE WHEN ABS(oldbalanceDest + amount - newbalanceDest) > 0.01 THEN 1 ELSE 0 END)
        AS destination_balance_error_count
FROM transactions
GROUP BY type
ORDER BY origin_balance_error_count DESC;