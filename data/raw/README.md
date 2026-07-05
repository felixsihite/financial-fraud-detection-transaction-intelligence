# Raw Data

This folder is reserved for the original Kaggle dataset used by the project.

## Dataset Source

- Kaggle: [Fraud Detection Dataset](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset)
- Expected local file name: `AIML Dataset.csv`
- Expected local path: `data/raw/AIML Dataset.csv`

## Storage Policy

The raw CSV is not committed to Git because the local file is approximately 470 MB, which exceeds GitHub's normal file-size limit. The project keeps the raw-data folder visible while preserving a clean repository size.

To reproduce the full pipeline, download the dataset from Kaggle and place the CSV in this folder:

```text
data/raw/AIML Dataset.csv
```
