"""Hybrid recommender: weighted Reciprocal Rank Fusion (RRF) of component models.

score(u, i) = sum_m  w_m / (rrf_k + rank_m(u, i))   over components m that rank i for u.

RRF (Cormack et al., 2009) needs no score calibration across models. Including a
non-personalized component (e.g. trending) makes the hybrid serve cold users with that
component's order and pads short personalized lists - the cold-start fallback.
"""

import logging
from typing import Self

import polars as pl

from recsys.evaluation.metrics import top_k
from recsys.models.base import REC_COLUMNS, Recommender, TrainData

logger = logging.getLogger(__name__)

DEFAULT_RRF_K = 60  # standard RRF constant: damps the advantage of the very top ranks


class Hybrid:
    def __init__(
        self,
        name: str,
        components: list[tuple[Recommender, float]],
        depth: int,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        if not components:
            raise ValueError(f"{name}: a hybrid needs at least one component")
        if any(weight <= 0 for _, weight in components):
            raise ValueError(f"{name}: component weights must be positive")
        self.name = name
        self._components = components
        self._depth = depth
        self._rrf_k = rrf_k

    def fit(self, data: TrainData) -> Self:
        for model, _ in self._components:
            model.fit(data)
        return self

    def recommend(self, users: pl.Series, k: int) -> pl.DataFrame:
        contributions = [
            self._rrf(model.recommend(users, max(k, self._depth)), weight)
            for model, weight in self._components
        ]
        fused = pl.concat(contributions).group_by("user_id", "item_id").agg(pl.col("score").sum())
        logger.info(
            "%s: served %s of %s users", self.name, fused["user_id"].n_unique(), users.n_unique()
        )
        return top_k(fused.select(REC_COLUMNS), k)

    def _rrf(self, recs: pl.DataFrame, weight: float) -> pl.DataFrame:
        ranked = top_k(recs, max(self._depth, 1))
        rank = pl.int_range(1, pl.len() + 1).over("user_id")
        return ranked.with_columns(score=weight / (self._rrf_k + rank)).select(REC_COLUMNS)
