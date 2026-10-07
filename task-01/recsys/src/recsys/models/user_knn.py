"""User-based collaborative filtering on ``implicit`` nearest-neighbour models.

``implicit`` computes item-item neighbours; fitting it on the transposed matrix
(items x users) yields user-user neighbours. A user's score for an item is the
summed similarity of their neighbours who interacted with it; seen items are removed.
"""

import logging
from typing import Literal, Self

import polars as pl
from implicit.nearest_neighbours import BM25Recommender, CosineRecommender
from scipy import sparse

from recsys.models.base import TrainData, empty_recs
from recsys.models.sparse import InteractionMatrix, top_k_from_scores

logger = logging.getLogger(__name__)

Similarity = Literal["cosine", "bm25"]


class UserKnn:
    def __init__(
        self,
        name: str,
        num_neighbours: int = 50,
        similarity: Similarity = "cosine",
        num_threads: int = 1,
    ) -> None:
        self.name = name
        self._num_threads = num_threads
        self._num_neighbours = num_neighbours
        self._similarity_kind = similarity
        self._data: InteractionMatrix | None = None
        self._similarity: sparse.csr_matrix | None = None  # users x users, K neighbours/row

    def fit(self, data: TrainData) -> Self:
        matrix = InteractionMatrix.from_interactions(data.interactions)
        slots = self._num_neighbours + 1  # implicit keeps the user itself among the K
        knn = (
            BM25Recommender(K=slots, num_threads=self._num_threads)
            if self._similarity_kind == "bm25"
            else CosineRecommender(K=slots, num_threads=self._num_threads)
        )
        knn.fit(matrix.matrix.T.tocsr(), show_progress=False)  # "items" = users
        similarity = knn.similarity.tocsr()
        similarity.setdiag(0)
        similarity.eliminate_zeros()
        self._data, self._similarity = matrix, similarity
        return self

    def recommend(self, users: pl.Series, k: int) -> pl.DataFrame:
        if self._data is None or self._similarity is None:
            raise RuntimeError(f"Model '{self.name}' must be fit before recommend")
        known = self._data.user_indices(users)
        logger.info("%s: %s of %s users known", self.name, known.height, users.n_unique())
        if known.is_empty():
            return empty_recs()
        rows = known["row"].to_numpy()
        seen = self._data.matrix[rows]
        scores = self._similarity[rows] @ self._data.matrix
        unseen = (scores - scores.multiply(seen)).tocsr()
        unseen.eliminate_zeros()
        return top_k_from_scores(unseen, known["user_id"], self._data.items, k)
