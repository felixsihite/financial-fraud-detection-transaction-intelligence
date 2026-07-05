"""Financial Fraud Detection & Transaction Intelligence.

Reusable package for data-quality auditing, leakage-aware feature engineering,
fraud model evaluation, model interpretation, and transaction risk scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

__version__ = "1.0.0"
__project_name__ = "Financial Fraud Detection & Transaction Intelligence"


@dataclass(frozen=True)
class ProjectMetadata:
    """Human-readable project metadata used by notebooks and documentation."""

    name: str
    version: str
    python_version: str
    dataset_source: str
    target_column: str


PROJECT_METADATA = ProjectMetadata(
    name=__project_name__,
    version=__version__,
    python_version="3.13.1",
    dataset_source="https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset",
    target_column="isFraud",
)

__all__ = [
    "PROJECT_METADATA",
    "ProjectMetadata",
    "__project_name__",
    "__version__",
    "config",
    "data",
    "evaluation",
    "features",
    "interpretation",
    "models",
    "pipeline",
    "scoring",
    "visualization",
]