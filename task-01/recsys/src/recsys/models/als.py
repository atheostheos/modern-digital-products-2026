"""Matrix factorization for implicit feedback: ALS from the ``implicit`` library.

Each user and item gets a latent vector; ALS (Hu, Koren & Volinsky, 2008) alternates
closed-form least-squares solves for users and items, treating every unobserved pair
as a weak negative weighted by ``alpha``. Replaces RePlay's Spark ALSWrap, whose
predict step (all users x all items in Spark) does not fit a laptop's memory.
"""

import logging
from typing import Self

import numpy as np
import polars as pl
from implicit.als import AlternatingLeastSquares
from threadpoolctl import threadpool_limits

from recsys.models.base import REC_SCHEMA, TrainData, empty_recs
from recsys.models.sparse import InteractionMatrix

logger = logging.getLogger(__name__)


class ImplicitAls:
    def __init__(
        self,
        name: str,
        *,
        factors: int,
        iterations: int,
        regularization: float,
        alpha: float,
        seed: int,
        threads: int,
    ) -> None:
        self.name = name
        self._params = {
            "factors": factors,
            "iterations": iterations,
            "regularization": regularization,
            "alpha": alpha,
            "random_state": seed,
            "num_threads": threads,
        }
        self._data: InteractionMatrix | None = None
        self._model: AlternatingLeastSquares | None = None

    def fit(self, data: TrainData) -> Self:
        matrix = InteractionMatrix.from_interactions(data.interactions)
        model = AlternatingLeastSquares(**self._params)
        with threadpool_limits(limits=1, user_api="blas"):  # implicit parallelizes itself
            model.fit(matrix.matrix, show_progress=False)
        self._data, self._model = matrix, model
        return self

    def recommend(self, users: pl.Series, k: int) -> pl.DataFrame:
        if self._data is None or self._model is None:
            raise RuntimeError(f"Model '{self.name}' must be fit before recommend")
        known = self._data.user_indices(users)
        logger.info("%s: %s of %s users known", self.name, known.height, users.n_unique())
        if known.is_empty():
            return empty_recs()
        rows = known["row"].to_numpy()
        with threadpool_limits(limits=1, user_api="blas"):
            ids, scores = self._model.recommend(
                rows, self._data.matrix[rows], N=k, filter_already_liked_items=True
            )
        valid = (ids >= 0) & np.isfinite(scores)  # fewer than k unseen items -> padding
        user_ids = np.repeat(known["user_id"].to_numpy(), ids.shape[1]).reshape(ids.shape)
        return pl.DataFrame(
            {
                "user_id": user_ids[valid],
                "item_id": self._data.items.to_numpy()[ids[valid]],
                "score": scores[valid].astype(np.float64),
            },
            schema=REC_SCHEMA,
        ).sort(["user_id", "score", "item_id"], descending=[False, True, False])
