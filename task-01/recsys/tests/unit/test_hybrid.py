from typing import Self

import polars as pl
import pytest

from recsys.models.base import TrainData
from recsys.models.hybrid import Hybrid

TRAIN = TrainData(interactions=pl.DataFrame(), items=pl.DataFrame())


class Stub:
    """Recommender returning fixed ranked lists; records what it was asked."""

    def __init__(self, name: str, lists: dict[str, list[str]]) -> None:
        self.name = name
        self._lists = lists
        self.fitted = False
        self.depths: list[int] = []

    def fit(self, data: TrainData) -> Self:
        self.fitted = True
        return self

    def recommend(self, users: pl.Series, k: int) -> pl.DataFrame:
        self.depths.append(k)
        rows = []
        for user in users.to_list():
            ranked = self._lists.get(user, [])[:k]
            rows += [(user, item, float(len(ranked) - pos)) for pos, item in enumerate(ranked)]
        return pl.DataFrame(
            rows,
            schema={"user_id": pl.String, "item_id": pl.String, "score": pl.Float64},
            orient="row",
        )


def test_fit_fits_every_component():
    a, b = Stub("a", {}), Stub("b", {})

    Hybrid("h", [(a, 1.0), (b, 1.0)], depth=5).fit(TRAIN)

    assert a.fitted and b.fitted


def test_reciprocal_rank_fusion_with_weights():
    cf = Stub("cf", {"u": ["x", "y"]})
    pop = Stub("pop", {"u": ["y", "z"]})
    hybrid = Hybrid("h", [(cf, 2.0), (pop, 1.0)], depth=10, rrf_k=60).fit(TRAIN)

    recs = hybrid.recommend(pl.Series(["u"]), k=3)

    # x: 2/61 ; y: 2/62 + 1/61 ; z: 1/62
    assert recs["item_id"].to_list() == ["y", "x", "z"]
    assert recs["score"].to_list() == pytest.approx([2 / 62 + 1 / 61, 2 / 61, 1 / 62])
    assert cf.depths == [10] and pop.depths == [10]


def test_users_served_by_only_one_component_get_that_components_order():
    cf = Stub("cf", {"warm": ["x"]})
    trending = Stub("trending", {"warm": ["t1"], "cold": ["t1", "t2"]})
    hybrid = Hybrid("h", [(cf, 1.0), (trending, 0.1)], depth=5).fit(TRAIN)

    recs = hybrid.recommend(pl.Series(["warm", "cold"]), k=2)

    assert recs.filter(pl.col("user_id") == "cold")["item_id"].to_list() == ["t1", "t2"]
    assert recs.filter(pl.col("user_id") == "warm")["item_id"].to_list() == ["x", "t1"]


def test_requires_components_and_positive_weights():
    with pytest.raises(ValueError, match="component"):
        Hybrid("h", [], depth=5)
    with pytest.raises(ValueError, match="weight"):
        Hybrid("h", [(Stub("a", {}), 0.0)], depth=5)
