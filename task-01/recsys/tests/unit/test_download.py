from pathlib import Path

import pytest

from recsys.config import DataConfig
from recsys.data.download import DownloadError, dataset_files, download_dataset

DATA = DataConfig(
    hf_repo="McAuley-Lab/Amazon-Reviews-2023",
    category="Musical_Instruments",
    download_dir=Path("data"),
    interim_dir=Path("data/interim"),
    processed_dir=Path("data/processed"),
)


def test_dataset_files_point_to_reviews_and_meta_for_category():
    files = dataset_files("Musical_Instruments")

    assert files.reviews == "raw/review_categories/Musical_Instruments.jsonl"
    assert files.meta == "raw/meta_categories/meta_Musical_Instruments.jsonl"


def test_download_fetches_both_files_into_raw_dir(tmp_path):
    calls = []

    def fake_fetch(*, repo_id: str, filename: str, repo_type: str, local_dir: Path) -> str:
        calls.append((repo_id, filename, repo_type, local_dir))
        return str(local_dir / filename)

    paths = download_dataset(DATA.resolve(tmp_path), fetch=fake_fetch)

    assert [c[1] for c in calls] == [
        "raw/review_categories/Musical_Instruments.jsonl",
        "raw/meta_categories/meta_Musical_Instruments.jsonl",
    ]
    assert all(c[0] == "McAuley-Lab/Amazon-Reviews-2023" and c[2] == "dataset" for c in calls)
    assert all(c[3] == tmp_path / "data" for c in calls)
    assert paths.reviews == tmp_path / "data/raw/review_categories/Musical_Instruments.jsonl"
    assert paths.meta.name == "meta_Musical_Instruments.jsonl"


def test_download_wraps_fetch_failures_with_context(tmp_path):
    def failing_fetch(**_: object) -> str:
        raise OSError("connection reset")

    with pytest.raises(DownloadError, match=r"Musical_Instruments\.jsonl.*connection reset"):
        download_dataset(DATA.resolve(tmp_path), fetch=failing_fetch)
