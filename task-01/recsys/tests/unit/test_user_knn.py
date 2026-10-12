import polars as pl
import pytest

from recsys.models.base import TrainData
from recsys.models.user_knn import UserKnn

ROWS = [
    ("u1", "a"),
    ("u1", "b"),
    ("u2", "a"),
    ("u2", "b"),
    ("u2", "c"),
    ("u3", "a"),
    ("u3", "c"),
    ("u4", "d"),
]
INTERACTIONS = pl.DataFrame({"user_id": [u for u, _ in ROWS], "item_id": [i for _, i in ROWS]})
TRAIN = TrainData(interactions=INTERACTIONS, items=pl.DataFrame({"item_id": ["a", "b", "c", "d"]}))


@pytest.mark.parametrize("similarity", ["cosine", "bm25"])
def test_recommends_items_of_similar_users_only(similarity):
    model = UserKnn("user_knn", num_neighbours=3, similarity=similarity).fit(TRAIN)

    recs = model.recommend(pl.Series(["u1", "cold"]), k=2)

    assert recs.columns == ["user_id", "item_id", "score"]
    assert recs["user_id"].to_list() == ["u1"]  # cold user unserved
    assert recs["item_id"].to_list() == ["c"]  # d is held only by a dissimilar user; a, b seen


def test_scores_are_sorted_and_seen_items_excluded():
    model = UserKnn("user_knn", num_neighbours=3).fit(TRAIN)

    recs = model.recommend(pl.Series(["u3"]), k=5)

    assert "a" not in recs["item_id"].to_list() and "c" not in recs["item_id"].to_list()
    assert recs["item_id"].to_list() == ["b"]
    assert recs["score"].is_sorted(descending=True)


def test_recommend_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        UserKnn("user_knn").recommend(pl.Series(["u1"]), k=1)


def test_only_unknown_users_give_empty_typed_frame():
    recs = UserKnn("user_knn").fit(TRAIN).recommend(pl.Series(["x"]), k=1)

    assert recs.is_empty() and recs.columns == ["user_id", "item_id", "score"]


def test_num_neighbours_counts_real_neighbours_not_the_user_itself():
    # with one neighbour, u1 must still get u2's unseen item (self is not a neighbour)
    model = UserKnn("user_knn", num_neighbours=1).fit(TRAIN)

    recs = model.recommend(pl.Series(["u1"]), k=2)

    assert recs["item_id"].to_list() == ["c"]


def test_cosine_scores_equal_summed_neighbour_similarity():
    model = UserKnn("user_knn", num_neighbours=3).fit(TRAIN)

    recs = model.recommend(pl.Series(["u1"]), k=1)

    # u1={a,b}; u2={a,b,c}: cos=2/sqrt(2*3); u3={a,c}: cos=1/sqrt(2*2); c held by u2 and u3
    expected = 2 / (2 * 3) ** 0.5 + 1 / (2 * 2) ** 0.5
    assert recs["score"].item() == pytest.approx(expected, rel=1e-5)


def test_thread_count_does_not_change_results():
    single = UserKnn("u", num_neighbours=3, num_threads=1).fit(TRAIN)
    multi = UserKnn("u", num_neighbours=3, num_threads=2).fit(TRAIN)

    users = pl.Series(["u1", "u3"])
    assert single.recommend(users, k=3).equals(multi.recommend(users, k=3))
