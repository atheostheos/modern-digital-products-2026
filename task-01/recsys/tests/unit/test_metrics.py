import math

import polars as pl
import pytest

from recsys.evaluation.metrics import (
    average_recommendation_popularity,
    compute_metrics,
    gini_index,
    long_tail_share,
    top_k,
)


def recs(rows: list[tuple[str, str, float]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=["user_id", "item_id", "score"], orient="row")


def pairs(rows: list[tuple[str, str]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=["user_id", "item_id"], orient="row")


ITEMS = pl.DataFrame({"item_id": ["a", "b", "c", "d"], "category": ["x", "x", "y", None]})
# history popularity: a=3, b=2, c=1, d=0 (d never interacted)
HISTORY = pairs([("h1", "a"), ("h2", "a"), ("h3", "a"), ("h1", "b"), ("h2", "b"), ("u1", "c")])


def test_accuracy_metrics_match_hand_computed_values():
    rec = recs([("u1", "a", 2.0), ("u1", "b", 1.0), ("u2", "c", 2.0), ("u2", "a", 1.0)])
    truth = pairs([("u1", "b"), ("u1", "d"), ("u2", "a")])

    m = compute_metrics(rec, truth, HISTORY, ITEMS, ks=[2])

    # u1: hit at rank 2 of 2 relevant; u2: hit at rank 2 of 1 relevant
    assert m["Precision@2"] == pytest.approx((1 / 2 + 1 / 2) / 2)
    assert m["Recall@2"] == pytest.approx((1 / 2 + 1) / 2)
    assert m["HitRate@2"] == 1.0
    assert m["MRR@2"] == pytest.approx(0.5)


def test_users_without_recommendations_count_as_misses():
    rec = recs([("u1", "b", 1.0)])
    truth = pairs([("u1", "b"), ("u2", "a")])

    m = compute_metrics(rec, truth, HISTORY, ITEMS, ks=[1])

    assert m["HitRate@1"] == 0.5


def test_beyond_accuracy_metrics_are_reported_per_k():
    rec = recs([("u1", "a", 2.0), ("u1", "c", 1.0), ("u2", "a", 2.0), ("u2", "b", 1.0)])
    truth = pairs([("u1", "b"), ("u2", "c")])

    m = compute_metrics(rec, truth, HISTORY, ITEMS, ks=[1, 2])

    assert m["Coverage@1"] == pytest.approx(1 / 3)  # {a} of history items {a, b, c}
    assert m["Coverage@2"] == pytest.approx(1.0)
    assert m["CategoricalDiversity@2"] == pytest.approx((2 / 2 + 1 / 2) / 2)
    assert {"Novelty@2", "Surprisal@2", "Gini@2", "ARP@2", "LongTailShare@2"} <= m.keys()


def test_gini_is_zero_for_uniform_exposure_and_high_for_concentrated():
    catalog = pl.Series(["a", "b", "c", "d"])
    uniform = recs([("u1", "a", 1.0), ("u2", "b", 1.0), ("u3", "c", 1.0), ("u4", "d", 1.0)])
    concentrated = recs([("u1", "a", 1.0), ("u2", "a", 1.0), ("u3", "a", 1.0), ("u4", "a", 1.0)])

    assert gini_index(uniform, catalog) == pytest.approx(0.0)
    assert gini_index(concentrated, catalog) == pytest.approx(0.75)  # (n-1)/n for n=4


def test_average_recommendation_popularity_uses_history_counts():
    rec = recs([("u1", "a", 2.0), ("u1", "d", 1.0), ("u2", "b", 1.0)])

    # u1: (3 + 0) / 2 = 1.5 ; u2: 2 -> mean 1.75
    assert average_recommendation_popularity(rec, HISTORY) == pytest.approx(1.75)


def test_long_tail_share_counts_slots_outside_popularity_head():
    rec = recs([("u1", "a", 2.0), ("u1", "c", 1.0), ("u2", "d", 1.0), ("u2", "b", 1.0)])

    # head = top 1/3 of the 3 history items -> {a}; tail slots: c, d, b
    assert long_tail_share(rec, HISTORY, head_share=1 / 3) == pytest.approx(3 / 4)


def test_popularity_measures_handle_empty_recommendations():
    empty = recs([]).cast({"user_id": pl.String, "item_id": pl.String, "score": pl.Float64})

    assert long_tail_share(empty, HISTORY) == 0.0
    assert gini_index(empty, pl.Series(["a", "b"])) == 0.0
    assert average_recommendation_popularity(empty, HISTORY) == 0.0


def test_no_recommendations_score_zero_accuracy_and_nan_beyond_accuracy():
    empty = recs([]).cast({"user_id": pl.String, "item_id": pl.String, "score": pl.Float64})
    truth = pairs([("u1", "a")])

    m = compute_metrics(empty, truth, HISTORY, ITEMS, ks=[2])

    assert m["HitRate@2"] == 0.0
    assert m["NDCG@2"] == 0.0
    assert math.isnan(m["Coverage@2"])
    assert math.isnan(m["Gini@2"])


def test_top_k_breaks_score_ties_by_item_id_and_truncates():
    rec = recs([("u1", "c", 1.0), ("u1", "a", 1.0), ("u1", "b", 2.0), ("u2", "z", 1.0)])

    top = top_k(rec, 2)

    assert top.filter(pl.col("user_id") == "u1")["item_id"].to_list() == ["b", "a"]
    assert top.filter(pl.col("user_id") == "u2")["item_id"].to_list() == ["z"]


def test_tied_scores_rank_identically_for_replay_and_own_metrics():
    # all scores tied: both accuracy (RePlay) and exposure (own) metrics must see items a, b @2
    rec = recs([("u1", "c", 1.0), ("u1", "b", 1.0), ("u1", "a", 1.0)])
    truth = pairs([("u1", "c")])

    m = compute_metrics(rec, truth, HISTORY, ITEMS, ks=[2])

    assert m["HitRate@2"] == 0.0  # c is ranked third by the item_id tie-break
    assert m["ARP@2"] == pytest.approx((3 + 2) / 2)


def test_novelty_counts_share_of_items_unseen_by_the_user():
    rec = recs([("u1", "c", 2.0), ("u1", "a", 1.0)])  # u1 has seen c in HISTORY
    truth = pairs([("u1", "b")])

    m = compute_metrics(rec, truth, HISTORY, ITEMS, ks=[2])

    assert m["Novelty@2"] == pytest.approx(1 / 2)


def test_gini_uses_full_item_catalog_including_items_without_history():
    rec = recs([("u1", "d", 1.0), ("u2", "d", 1.0)])  # d has no history interactions
    truth = pairs([("u1", "a")])

    m = compute_metrics(rec, truth, HISTORY, ITEMS, ks=[1])

    assert m["Gini@1"] == pytest.approx(0.75)  # all exposure on one of 4 catalog items


def test_novelty_is_one_when_no_served_user_has_history():
    rec = recs([("new_user", "a", 1.0)])

    m = compute_metrics(rec, pairs([("new_user", "a")]), HISTORY, ITEMS, ks=[1])

    assert m["Novelty@1"] == 1.0
