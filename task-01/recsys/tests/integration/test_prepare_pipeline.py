import json

import polars as pl
import pytest

from recsys.config import load_config
from recsys.data.download import dataset_files
from recsys.data.ingest import IngestError
from recsys.data.pipeline import prepare
from tests.toy_data import TOY_CATEGORY


def test_prepare_writes_splits_items_and_stats(toy_project):
    cfg = load_config(toy_project / "config.yaml").resolve(toy_project)

    result = prepare(cfg)

    processed = toy_project / "data/processed"
    train = pl.read_parquet(processed / "train.parquet")
    test = pl.read_parquet(processed / "test.parquet")
    items = pl.read_parquet(processed / "items.parquet")
    assert train.height == 2  # u1-p1, u2-p1 (deduplicated)
    assert pl.read_parquet(processed / "val.parquet").height == 1
    assert sorted(test["user_id"].to_list()) == ["u1", "u3"]
    assert items.height == 3
    assert items.filter(pl.col("item_id") == "p2")["price"].item() == 5.0

    stats = json.loads((toy_project / "results/data/dataset_stats.json").read_text())
    assert stats == result.stats
    assert stats["raw_reviews"] == 7
    assert stats["dropped_after_test_end"] == 1
    assert stats["interactions"]["interactions"] == 6
    assert stats["splits"]["test"]["users_with_history_1plus_share"] == 0.5
    assert stats["item_features"]["brand"] == pytest.approx(2 / 3)
    assert stats["distinct_brands"] == 2  # null brand is not a brand
    assert stats["interaction_items_missing_from_meta"] == 0


def test_prepare_fails_with_hint_when_raw_data_missing(toy_project):
    cfg = load_config(toy_project / "config.yaml").resolve(toy_project)
    (toy_project / "data" / dataset_files(TOY_CATEGORY).reviews).unlink()

    with pytest.raises(IngestError, match="not found"):
        prepare(cfg)
