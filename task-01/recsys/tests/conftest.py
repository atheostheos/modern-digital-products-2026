import json
from pathlib import Path

import pytest
import yaml

from recsys.config import SparkConfig
from recsys.data.download import dataset_files
from recsys.spark import spark_session
from tests.toy_data import META, REVIEWS, TOY_CATEGORY

TESTS_ROOT = Path(__file__).parent
LEVELS = ("unit", "integration", "e2e")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark each test with its level, taken from its top-level tests/ subdirectory."""
    for item in items:
        level = Path(item.path).relative_to(TESTS_ROOT).parts[0]
        if level in LEVELS:
            item.add_marker(getattr(pytest.mark, level))


@pytest.fixture
def toy_project(tmp_path: Path) -> Path:
    """Project root with tiny raw JSONL files and a config.yaml using relative paths."""
    files = dataset_files(TOY_CATEGORY)
    reviews = tmp_path / "data" / files.reviews
    meta = tmp_path / "data" / files.meta
    reviews.parent.mkdir(parents=True)
    meta.parent.mkdir(parents=True)
    reviews.write_text(
        "".join(
            json.dumps(
                {
                    "user_id": u,
                    "parent_asin": p,
                    "timestamp": t,
                    "rating": 5.0,
                    "verified_purchase": True,
                    "text": "ok",
                }
            )
            + "\n"
            for u, p, t in REVIEWS
        )
    )
    meta.write_text("".join(json.dumps(m) + "\n" for m in META))
    config = {
        "seed": 1,
        "results_dir": "results",
        "data": {
            "hf_repo": "x/y",
            "category": TOY_CATEGORY,
            "download_dir": "data",
            "interim_dir": "data/interim",
            "processed_dir": "data/processed",
        },
        "split": {"val_start": "2022-10-01", "test_start": "2023-01-01", "test_end": "2023-04-01"},
        "evaluation": {"ks": [1, 2], "history_bins": [0, 1, 2]},
        "spark": {"driver_memory_gb": 1, "shuffle_partitions": 2, "local_dir": "data/spark-tmp"},
        "models": {
            "popular": {"kind": "popularity"},
            "item_knn": {"kind": "item_knn", "num_neighbours": 5},
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    return tmp_path


@pytest.fixture(scope="session")
def spark(tmp_path_factory: pytest.TempPathFactory):
    """One local SparkSession for all Spark-backed tests (startup is ~10-30 s)."""

    local_dir = tmp_path_factory.mktemp("spark")
    session = spark_session(
        SparkConfig(driver_memory_gb=1, shuffle_partitions=2, local_dir=local_dir, cores=2)
    )
    yield session
    session.stop()
