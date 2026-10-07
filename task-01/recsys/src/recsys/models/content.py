"""Content-based recommender: LSA vectors of item text + brand, user profile = mean vector.

TF-IDF (sklearn) -> TruncatedSVD -> L2-normalized dense item vectors. A user's profile is
the normalized mean of the vectors of their (most recent ``profile_items``) items; the
recommendations are the nearest items by cosine similarity (sklearn NearestNeighbors),
excluding seen ones. Items without interactions are reachable through their text.
"""

import logging
from typing import Self

import numpy as np
import polars as pl
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from recsys.evaluation.metrics import top_k
from recsys.models.base import REC_COLUMNS, REC_SCHEMA, TrainData, empty_recs

logger = logging.getLogger(__name__)

MIN_VOCABULARY = 2  # TruncatedSVD needs at least one component below the vocabulary size


class ContentBased:
    def __init__(
        self,
        name: str,
        *,
        max_features: int,
        min_df: int,
        svd_components: int,
        profile_items: int | None,
        seed: int,
        threads: int = 1,
    ) -> None:
        self.name = name
        self._max_features = max_features
        self._min_df = min_df
        self._svd_components = svd_components
        self._profile_items = profile_items
        self._seed = seed
        self._threads = threads
        self._item_ids: pl.Series | None = None
        self._vectors: np.ndarray | None = None
        self._history: pl.DataFrame | None = None  # user_id, item_id, timestamp, col
        self._index: NearestNeighbors | None = None

    def fit(self, data: TrainData) -> Self:
        items = data.items.sort("item_id")
        with threadpool_limits(limits=self._threads):  # caps BLAS threads in TruncatedSVD
            vectors = self._item_vectors(items)
        columns = pl.DataFrame({"item_id": items["item_id"], "col": np.arange(items.height)})
        pairs = (
            data.interactions.select("user_id", "item_id", "timestamp")
            .sort("timestamp", descending=True, maintain_order=True)
            .unique(["user_id", "item_id"], keep="first", maintain_order=True)
        )
        history = pairs.join(columns, on="item_id")
        if dropped := pairs.height - history.height:
            logger.warning(
                "%s: %s interactions on items without metadata ignored", self.name, dropped
            )
        self._item_ids, self._vectors, self._history = items["item_id"], vectors, history
        self._index = NearestNeighbors(
            metric="cosine", algorithm="brute", n_jobs=self._threads
        ).fit(vectors)
        return self

    def recommend(self, users: pl.Series, k: int) -> pl.DataFrame:
        if self._history is None or self._index is None or self._vectors is None:
            raise RuntimeError(f"Model '{self.name}' must be fit before recommend")
        history = self._history.join(users.unique().to_frame("user_id"), on="user_id")
        if history.is_empty():
            return empty_recs()
        profile_users, profiles = self._profiles(history)
        logger.info(
            "%s: %s of %s users have a profile", self.name, profile_users.len(), users.n_unique()
        )
        if profile_users.is_empty():
            return empty_recs()
        max_seen = history.group_by("user_id").len()["len"].max()
        n_neighbours = min(self._vectors.shape[0], k + max_seen)
        with threadpool_limits(limits=self._threads):
            distances, cols = self._index.kneighbors(profiles, n_neighbors=n_neighbours)
        candidates = pl.DataFrame(
            {
                "user_id": np.repeat(profile_users.to_numpy(), n_neighbours),
                "item_id": self._item_ids.to_numpy()[cols.ravel()],
                "score": 1.0 - distances.ravel().astype(np.float64),
            },
            schema=REC_SCHEMA,
        )
        unseen = candidates.join(
            history.select("user_id", "item_id"), on=["user_id", "item_id"], how="anti"
        )
        return top_k(unseen.select(REC_COLUMNS), k)

    def _item_vectors(self, items: pl.DataFrame) -> np.ndarray:
        texts = (
            items.select(
                pl.concat_str(
                    [pl.col("text").fill_null(""), pl.col("brand").fill_null("")], separator=" "
                )
            )
            .to_series()
            .to_list()
        )
        tfidf = TfidfVectorizer(
            max_features=self._max_features,
            min_df=self._min_df,
            stop_words="english",
            sublinear_tf=True,
            dtype=np.float32,
        ).fit_transform(texts)
        if tfidf.shape[1] < MIN_VOCABULARY:
            raise ValueError(
                f"{self.name}: vocabulary of {tfidf.shape[1]} term(s) is too small for SVD; "
                "lower min_df or check the item text"
            )
        components = min(self._svd_components, tfidf.shape[1] - 1)
        reduced = TruncatedSVD(n_components=components, random_state=self._seed).fit_transform(
            tfidf
        )
        logger.info(
            "%s: %s items, vocabulary %s, %s components",
            self.name,
            items.height,
            tfidf.shape[1],
            components,
        )
        return normalize(reduced).astype(np.float32)

    def _profiles(self, history: pl.DataFrame) -> tuple[pl.Series, np.ndarray]:
        """Normalized mean item vector per user over their (most recent) items.

        Ties in timestamp are broken by item_id (descending) so windows are deterministic;
        users whose items all have zero vectors (no usable text) get no profile.
        """
        window = history.sort(["user_id", "timestamp", "item_id"], descending=[False, True, True])
        if self._profile_items is not None:
            window = window.group_by("user_id", maintain_order=True).head(self._profile_items)
        users = window["user_id"].unique(maintain_order=True)
        rows = window.join(
            pl.DataFrame({"user_id": users, "row": np.arange(users.len())}), on="user_id"
        )
        weights = sparse.csr_matrix(
            (
                np.ones(rows.height, dtype=np.float32),
                (rows["row"].to_numpy(), rows["col"].to_numpy()),
            ),
            shape=(users.len(), self._vectors.shape[0]),
        )
        profiles = weights @ self._vectors
        has_signal = np.linalg.norm(profiles, axis=1) > 0
        if not has_signal.any():
            return users.clear(), profiles[has_signal]
        return users.filter(pl.Series(has_signal)), normalize(profiles[has_signal])
