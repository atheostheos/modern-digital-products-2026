"""Typed, immutable project configuration loaded from YAML."""

from datetime import date
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

# HF dataset category names look like "Musical_Instruments"; the pattern also
# keeps the value safe to embed in download paths.
CATEGORY_PATTERN = r"^[A-Za-z][A-Za-z_]*$"


class ConfigError(ValueError):
    """Raised when a config file is missing or invalid."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DataConfig(_Frozen):
    hf_repo: str = Field(min_length=1)
    category: str = Field(pattern=CATEGORY_PATTERN)
    download_dir: Path  # HF local_dir: files land at <download_dir>/raw/... (repo layout)
    interim_dir: Path
    processed_dir: Path

    def resolve(self, root: Path) -> Self:
        """Return a copy with relative paths anchored at ``root``."""
        return self.model_copy(
            update={
                "download_dir": root / self.download_dir,
                "interim_dir": root / self.interim_dir,
                "processed_dir": root / self.processed_dir,
            }
        )


class SplitConfig(_Frozen):
    """Global temporal split: train < val_start <= val < test_start <= test < test_end."""

    val_start: date
    test_start: date
    test_end: date

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if not self.val_start < self.test_start < self.test_end:
            raise ValueError("split dates must satisfy val_start < test_start < test_end")
        return self


class Config(_Frozen):
    seed: int = 42
    results_dir: Path
    data: DataConfig
    split: SplitConfig

    def resolve(self, root: Path) -> Self:
        """Return a copy with every relative path anchored at ``root``."""
        return self.model_copy(
            update={"results_dir": root / self.results_dir, "data": self.data.resolve(root)}
        )


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load and validate a YAML config file."""
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"Malformed YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Config root must be a mapping, got {type(raw).__name__}: {path}")
    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid config {path}:\n{exc}") from exc
