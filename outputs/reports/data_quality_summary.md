# Data Quality and Leakage Audit

- Source file: `D:\Project\Data Science\financial_fraud_detection_&_transaction_intelligence\data\raw\AIML Dataset.csv`
- Dataset source: https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset
- Rows: 6,362,620
- Columns: 11
- Expected schema match: `True`
- Missing values: 0
- Duplicate rows (hash audit): 0
- Fraud transactions: 8,213
- Fraud rate: 0.1291%
- Step range: 1 to 743

## Transaction Type Fraud Distribution

| Type | Transactions | Fraud | Fraud Rate |
|---|---:|---:|---:|
| CASH_IN | 1,399,284 | 0 | 0.0000% |
| CASH_OUT | 2,237,500 | 4,116 | 0.1840% |
| DEBIT | 41,432 | 0 | 0.0000% |
| PAYMENT | 2,151,495 | 0 | 0.0000% |
| TRANSFER | 532,909 | 4,097 | 0.7688% |

## Legacy Rule Audit

- `isFlaggedFraud` fraud capture rate: 0.1948%
- `isFlaggedFraud` precision when triggered: 100.0000%
- Modeling pipeline excludes `isFlaggedFraud` from the final predictive feature set and keeps it only as a benchmark.

## Dataset Limitations

- The data is simulated transaction data, not real customer data.
- There are no real calendar timestamps, demographics, investigation outcomes, or confirmed recovery amounts.
- Business impact values in this project are scenario assumptions only.