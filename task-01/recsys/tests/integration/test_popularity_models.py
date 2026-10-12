from datetime import datetime

import polars as pl
import pytest

from recsys.models.base import TrainData
from recsys.models.popularity import KeyedPopularity

pytestmark = pytest.mark.spark

# guitars: g1 popular (3 users), g2 (1); drums: d1 popular (2), d2 (1); recent: n1 (2, last week)
INTERACTIONS = pl.DataFrame(
    {
        "user_id": ["a", "b", "c", "a", "d", "e", "e", "f", "g", "h"],
        "item_id": ["g1", "g1", "g1", "g2", "d1", "d1", "d2", "n1", "n1", "g2"],
        "timestamp": [datetime(2022, 1, 1)] * 8 + [datetime(2022, 6, 28), datetime(2022, 6, 30)],
        "rating": [5.0] * 10,
    }
)
ITEMS = pl.DataFrame(
    {
        "item_id": ["g1", "g2", "d1", "d2", "n1"],
        "category": ["Guitars", "Guitars", "Drums", "Drums", "Drums"],
        "brand": ["Fender", "Gibson", "Pearl", "Pearl", None],
    }
)
TRAIN = TrainData(interactions=INTERACTIONS, items=ITEMS)


def items_of(recs: pl.DataFrame, user: str) -> list[str]:
    return recs.filter(pl.col("user_id") == user)["item_id"].to_list()


def test_global_popularity_serves_cold_users_and_hides_seen_items(spark):
    model = KeyedPopularity("pop", spark).fit(TRAIN)

    recs = model.recommend(pl.Series(["a", "cold"]), k=2)

    assert items_of(recs, "cold") == ["g1", "d1"]
    assert items_of(recs, "a") == ["d1", "n1"]  # g1, g2 seen; n1 vs g2/d2 by popularity then id
    assert recs.columns == ["user_id", "item_id", "score"]


def test_trending_counts_only_the_recent_window(spark):
    model = KeyedPopularity("trending", spark, window_days=7).fit(TRAIN)

    recs = model.recommend(pl.Series(["cold"]), k=3)

    assert items_of(recs, "cold") == ["g2", "n1"]  # one recent user each; tie -> item id


def test_category_popularity_uses_the_users_main_category(spark):
    model = KeyedPopularity("cat_pop", spark, key="category").fit(TRAIN)

    recs = model.recommend(pl.Series(["e", "b", "cold"]), k=2)

    assert items_of(recs, "e") == ["n1"]  # Drums fan; d1, d2 seen
    assert items_of(recs, "b") == ["g2"]  # Guitars fan; g1 seen
    assert items_of(recs, "cold") == []  # no history -> no category


def test_brand_popularity_recommends_within_the_users_brand(spark):
    model = KeyedPopularity("brand_pop", spark, key="brand").fit(TRAIN)

    recs = model.recommend(pl.Series(["d"]), k=3)

    assert items_of(recs, "d") == ["d2"]  # Pearl: d1 seen, d2 left


def test_recommend_before_fit_raises(spark):
    with pytest.raises(RuntimeError, match="fit"):
        KeyedPopularity("pop", spark).recommend(pl.Series(["a"]), k=1)


def test_keyed_model_returns_empty_frame_when_no_user_has_a_key(spark):
    model = KeyedPopularity("cat_pop", spark, key="category").fit(TRAIN)

    recs = model.recommend(pl.Series(["cold1", "cold2"]), k=2)

    assert recs.is_empty()
    assert recs.columns == ["user_id", "item_id", "score"]


def test_user_who_saw_the_whole_candidate_list_gets_fewer_recs_and_a_warning(spark, caplog):
    model = KeyedPopularity("pop", spark, candidates=2).fit(TRAIN)  # candidates: g1, d1

    with caplog.at_level("WARNING"):
        recs = model.recommend(pl.Series(["a", "d"]), k=2)

    assert items_of(recs, "a") == ["d1"]  # g1 seen; list exhausted
    assert items_of(recs, "d") == ["g1"]
    assert "2 of 2 served users got fewer than k=2" in caplog.text


def test_keyed_candidates_break_popularity_ties_by_item_id(spark):
    # Drums: d1 and n1 both have 2 users; with 1 candidate per key, d1 must win
    model = KeyedPopularity("cat_pop", spark, key="category", candidates=1).fit(TRAIN)

    recs = model.recommend(pl.Series(["c"]), k=1)  # Guitars fan; check Drums via another user
    drum_fan = model.recommend(pl.Series(["x_d"]), k=1)

    assert items_of(recs, "c") == []  # Guitars top-1 is g1, already seen by c
    assert drum_fan.is_empty()  # unknown user has no key
    assert model.candidate_lists().filter(pl.col("key") == "Drums")["item_id"].to_list() == ["d1"]


def test_candidate_lists_before_fit_raises(spark):
    with pytest.raises(RuntimeError, match="fit"):
        KeyedPopularity("pop", spark).candidate_lists()
