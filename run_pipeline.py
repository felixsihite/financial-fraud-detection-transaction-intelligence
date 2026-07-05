"""Command-line entry point for the fraud analytics pipeline.

This wrapper keeps the project root runnable with:

    python run_pipeline.py

The implementation lives in `src.pipeline.run_pipeline` so notebooks, tests,
and automation can reuse the same pipeline without shell-specific behavior.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

from src.config import PROJECT_ROOT, RAW_DATA_PATH
from src.pipeline.run_pipeline import main as pipeline_main


SUPPORTED_MAJOR_MINOR = (3, 13)


def validate_runtime() -> None:
    """Fail fast on missing inputs and document the local Python runtime."""

    current = sys.version_info[:2]
    if current != SUPPORTED_MAJOR_MINOR:
        print(
            "Warning: this portfolio was validated with "
            f"Python {SUPPORTED_MAJOR_MINOR[0]}.{SUPPORTED_MAJOR_MINOR[1]}. "
            f"Current runtime is {platform.python_version()}."
        )

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            "Raw dataset not found. Expected file: "
            f"{RAW_DATA_PATH}. Place `AIML Dataset.csv` in data/raw before running."
        )


def main() -> None:
    """Validate project context, then delegate to the package pipeline."""

    validate_runtime()
    print(f"Project root: {Path(PROJECT_ROOT)}")
    print(f"Raw dataset: {Path(RAW_DATA_PATH)}")
    pipeline_main()


if __name__ == "__main__":
    main()