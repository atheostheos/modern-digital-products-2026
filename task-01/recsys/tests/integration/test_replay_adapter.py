from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
from replay.models import AssociationRulesItemRec, ItemKNN, PopRec

from recsys.config import SparkConfig
from recsys.models.base import TrainData
from recsys.models.replay_models import ReplayModel
from recsys.spark import spark_session

pytestmark = pytest.mark.spark

# u1-u3 co-interact on a/b, so item-KNN links them; c is popular among others
INTERACTIONS = pl.DataFrame(
    {
        "user_id": ["u1", "u1", "u2", "u2", "u2", "u3", "u3", "u4", "u4"],
        "item_id": ["a", "b", "a", "b", "c", "a", "c", "c", "d"],
        "timestamp": [datetime(2022, 1, d) for d in range(1, 10)],
        "rating": [5.0] * 9,
    }
)
ITEMS = pl.DataFrame({"item_id": ["a", "b", "c", "d"], "category": ["x", "x", "y", "y"]})
TRAIN = TrainData(interactions=INTERACTIONS, items=ITEMS)


def test_session_uses_utc(spark):
    assert spark.conf.get("spark.sql.session.timeZone") == "UTC"


def test_session_uses_the_configured_number_of_cores(spark):
    assert spark.sparkContext.master == "local[2]"


def test_item_knn_recommends_unseen_items_for_known_users(spark):
    model = ReplayModel("item_knn", lambda: ItemKNN(num_neighbours=3), spark).fit(TRAIN)

    recs = model.recommend(pl.Series(["u1", "u3", "cold_user"]), k=2)

    assert recs.columns == ["user_id", "item_id", "score"]
    assert set(recs["user_id"]) <= {"u1", "u3"}  # unknown users get nothing
    seen = INTERACTIONS.select("user_id", "item_id")
    assert recs.join(seen, on=["user_id", "item_id"], how="semi").is_empty()
    assert recs.group_by("user_id").len()["len"].max() <= 2
    assert recs.filter(pl.col("user_id") == "u1")["item_id"].to_list()[0] == "c"


def test_pop_rec_ranks_by_popularity(spark):
    model = ReplayModel("pop", PopRec, spark).fit(TRAIN)

    recs = model.recommend(pl.Series(["u1"]), k=2)

    assert recs["item_id"].to_list() == ["c", "d"]  # a, b already seen by u1
    assert recs["score"].is_sorted(descending=True)


def test_recommend_before_fit_raises(spark):
    with pytest.raises(RuntimeError, match="fit"):
        ReplayModel("pop", PopRec, spark).recommend(pl.Series(["u1"]), k=1)


def test_recommend_for_only_unknown_users_returns_empty_typed_frame(spark):
    model = ReplayModel("pop", PopRec, spark).fit(TRAIN)

    recs = model.recommend(pl.Series(["nobody", "stranger"]), k=3)

    assert recs.is_empty()
    assert recs.schema == {"user_id": pl.String, "item_id": pl.String, "score": pl.Float64}


def test_session_reapplies_utc_when_reused(spark):
    spark.conf.set("spark.sql.session.timeZone", "Europe/Moscow")

    reused = spark_session(
        SparkConfig(
            driver_memory_gb=1,
            shuffle_partitions=2,
            local_dir=Path(spark.conf.get("spark.local.dir")),
        )
    )

    assert reused.conf.get("spark.sql.session.timeZone") == "UTC"


def test_association_rules_recommend_items_co_reviewed_by_other_users(spark):
    # a & b are co-reviewed by u1, u2, u3; c with a by u2, u3. u4 reviewed only a.
    rows = [
        ("u1", "a"),
        ("u1", "b"),
        ("u2", "a"),
        ("u2", "b"),
        ("u2", "c"),
        ("u3", "a"),
        ("u3", "b"),
        ("u3", "c"),
        ("u4", "a"),
        ("u5", "d"),
    ]
    interactions = pl.DataFrame(
        {
            "user_id": [u for u, _ in rows],
            "item_id": [i for _, i in rows],
            "timestamp": [datetime(2022, 1, 1)] * len(rows),
            "rating": [5.0] * len(rows),
        }
    )
    model = ReplayModel(
        "also_bought",
        lambda: AssociationRulesItemRec(
            session_column="user_id", min_item_count=1, min_pair_count=2
        ),
        spark,
    ).fit(TrainData(interactions=interactions, items=ITEMS))

    recs = model.recommend(pl.Series(["u4"]), k=2)

    assert recs["item_id"].to_list() == ["b", "c"]  # b co-occurs with a more often than c
