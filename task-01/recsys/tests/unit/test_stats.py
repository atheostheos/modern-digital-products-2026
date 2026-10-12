from datetime import datetime

import polars as pl
import pytest

from recsys.data.split import Splits
from recsys.data.stats import history_coverage, interaction_stats, item_feature_coverage


def frame(rows: list[tuple[str, str]], ts: datetime = datetime(2022, 1, 1)) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "user_id": [u for u, _ in rows],
            "item_id": [i for _, i in rows],
            "timestamp": [ts] * len(rows),
        }
    )


def test_interaction_stats_counts_sparsity_and_activity():
    df = frame([("u1", "a"), ("u1", "b"), ("u2", "a"), ("u3", "a")])

    stats = interaction_stats(df)

    assert stats["interactions"] == 4
    assert stats["users"] == 3
    assert stats["items"] == 2
    assert stats["density"] == pytest.approx(4 / 6)
    assert stats["users_with_1_interaction_share"] == pytest.approx(2 / 3)
    assert stats["users_with_5plus_interactions_share"] == 0.0
    assert stats["top_20pct_items_interaction_share"] == pytest.approx(3 / 4)


def test_item_feature_coverage_reports_non_null_shares():
    items = pl.DataFrame(
        {
            "item_id": ["a", "b"],
            "brand": ["X", None],
            "category": ["c", "c"],
            "price": [None, None],
            "bought_together": [None, None],
        },
        schema_overrides={"price": pl.Float64, "bought_together": pl.List(pl.String)},
    )

    cov = item_feature_coverage(items)

    assert cov == {"brand": 0.5, "category": 1.0, "price": 0.0, "bought_together": 0.0}


def test_history_coverage_measures_warm_users_and_cold_items():
    splits = Splits(
        train=frame([("u1", "a"), ("u1", "b"), ("u2", "a")]),
        val=frame([("u2", "b")]),
        test=frame([("u1", "c"), ("u2", "a"), ("u3", "a")]),
    )

    cov = history_coverage(splits)

    # test history = train + val: u1 has 2, u2 has 2, u3 has 0
    assert cov["test"]["users"] == 3
    assert cov["test"]["users_with_history_1plus_share"] == pytest.approx(2 / 3)
    assert cov["test"]["users_with_history_2plus_share"] == pytest.approx(2 / 3)
    assert cov["test"]["interactions_on_unseen_items_share"] == pytest.approx(1 / 3)
    # val history = train only: u2 has 1
    assert cov["val"]["users_with_history_1plus_share"] == 1.0
    assert cov["val"]["users_with_history_2plus_share"] == 0.0
