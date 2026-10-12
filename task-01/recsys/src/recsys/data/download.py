"""Download raw Amazon Reviews 2023 files for one category from Hugging Face."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

from recsys.config import DataConfig

logger = logging.getLogger(__name__)

Fetch = Callable[..., str]


class DownloadError(RuntimeError):
    """Raised when a dataset file cannot be downloaded."""


@dataclass(frozen=True)
class DatasetFiles:
    reviews: str
    meta: str


@dataclass(frozen=True)
class DatasetPaths:
    reviews: Path
    meta: Path


def dataset_files(category: str) -> DatasetFiles:
    """Repo-relative file names of the reviews and metadata JSONL for ``category``."""
    return DatasetFiles(
        reviews=f"raw/review_categories/{category}.jsonl",
        meta=f"raw/meta_categories/meta_{category}.jsonl",
    )


def download_dataset(data: DataConfig, fetch: Fetch = hf_hub_download) -> DatasetPaths:
    """Download (or reuse cached) reviews + meta files into ``data.download_dir``."""
    files = dataset_files(data.category)
    return DatasetPaths(
        reviews=_fetch_one(data, files.reviews, fetch),
        meta=_fetch_one(data, files.meta, fetch),
    )


def _fetch_one(data: DataConfig, filename: str, fetch: Fetch) -> Path:
    logger.info("Fetching %s/%s into %s", data.hf_repo, filename, data.download_dir)
    try:
        local = fetch(
            repo_id=data.hf_repo,
            filename=filename,
            repo_type="dataset",
            local_dir=data.download_dir,
        )
    except Exception as exc:
        raise DownloadError(f"Failed to download {data.hf_repo}/{filename}: {exc}") from exc
    return Path(local)
