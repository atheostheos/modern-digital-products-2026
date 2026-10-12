"""Real-model tests read weights from task-01/ai/models (offline, never downloading)."""

import os
from pathlib import Path

import pytest

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"


@pytest.fixture(scope="session", autouse=True)
def offline_model_cache():
    os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))
    os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / "torch"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return MODELS_DIR
