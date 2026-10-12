"""Paths shared by the nodes. Model weights live in task-01/ai/models (git-ignored)."""

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[3]  # server/src/tdml/config.py -> task-01/ai
DEFAULT_MODELS_DIR = PROJECT_DIR / "models"


def models_dir() -> Path:
    return Path(os.environ.get("TDML_MODELS_DIR", DEFAULT_MODELS_DIR))
