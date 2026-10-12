import polars as pl
import pytest

from recsys.models.als import ImplicitAls
from recsys.models.base import TrainData

# two taste clusters: guitar fans (g*) and drum fans (d*); "gf" lacks g3, "df" lacks d3
GUITAR = [f"g{i}" for i in range(1, 4)]
DRUMS = [f"d{i}" for i in range(1, 4)]
ROWS = (
    [(f"guitar_fan{u}", item) for u in range(6) for item in GUITAR]
    + [(f"drum_fan{u}", item) for u in range(6) for item in DRUMS]
    + [("gf", "g1"), ("gf", "g2"), ("df", "d1"), ("df", "d2")]
)
TRAIN = TrainData(
    interactions=pl.DataFrame({"user_id": [u for u, _ in ROWS], "item_id": [i for _, i in ROWS]}),
    items=pl.DataFrame({"item_id": GUITAR + DRUMS}),
)


def model() -> ImplicitAls:
    return ImplicitAls(
        "als", factors=4, iterations=30, regularization=0.01, alpha=10.0, seed=0, threads=1
    )


def test_latent_factors_recommend_the_missing_item_of_the_users_cluster():
    recs = model().fit(TRAIN).recommend(pl.Series(["gf", "df"]), k=1)

    assert recs.filter(pl.col("user_id") == "gf")["item_id"].to_list() == ["g3"]
    assert recs.filter(pl.col("user_id") == "df")["item_id"].to_list() == ["d3"]


def test_seen_items_are_excluded_and_scores_sorted():
    recs = model().fit(TRAIN).recommend(pl.Series(["gf"]), k=4)

    assert not {"g1", "g2"} & set(recs["item_id"].to_list())
    assert recs.height == 4
    assert recs["score"].is_sorted(descending=True)
    assert recs.columns == ["user_id", "item_id", "score"]


def test_fixed_seed_gives_identical_recommendations():
    users = pl.Series(["gf", "df", "guitar_fan0"])

    first = model().fit(TRAIN).recommend(users, k=3)
    second = model().fit(TRAIN).recommend(users, k=3)

    assert first.equals(second)


def test_cold_users_are_not_served():
    recs = model().fit(TRAIN).recommend(pl.Series(["gf", "cold"]), k=2)

    assert set(recs["user_id"]) == {"gf"}
    assert model().fit(TRAIN).recommend(pl.Series(["cold"]), k=2).is_empty()


def test_recommend_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        model().recommend(pl.Series(["gf"]), k=1)
