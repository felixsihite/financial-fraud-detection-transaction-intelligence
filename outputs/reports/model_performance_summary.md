# Model Performance Summary

- Validation strategy: time-aware validation using step
- Step cutoff: 335
- Best final model: Random Forest
- Optimized threshold: 0.750000
- PR-AUC: 0.999792
- ROC-AUC: 0.999992
- Fraud precision: 1.000000
- Fraud recall: 0.999328
- F2 score: 0.999462
- False positives: 0
- False negatives: 3

The selected model excludes `isFlaggedFraud`. A separate benchmark with the legacy flag is generated only to quantify its effect.