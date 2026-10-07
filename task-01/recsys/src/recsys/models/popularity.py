"""Popularity-family heuristics: global, trending, same-category and same-brand popular.

RePlay computes the rankings (``PopRec`` globally, ``CatPopRec`` per category or brand);
this module assigns each user a key (their main category / brand, or one global key),
then serves the key's candidate list minus the items the user has already seen.
"""

import logging
from datetime import timedelta
from typing import Self

import polars as pl
from pyspark.sql import SparkSession
from replay.models import CatPopRec, PopRec

from recsys.evaluation.metrics import top_k
from recsys.models.base import REC_COLUMNS, TrainData, empty_recs
from recsys.models.replay_data import build_dataset, decode_items

logger = logging.getLogger(__name__)

GLOBAL_KEY = "__global__"
DEFAULT_CANDIDATES = 1000  # per key; shortfalls (k > candidates - seen) are logged


def main_key_per_user(interactions: pl.DataFrame, items: pl.DataFrame, key: str) -> pl.DataFrame:
    """Each user's most frequent ``key`` value (ties -> most recent); users without one omitted."""
    return (
        interactions.join(items.select("item_id", key), on="item_id", how="inner")
        .drop_nulls(key)
        .group_by("user_id", key)
        .agg(pl.len().alias("n"), pl.col("timestamp").max().alias("last"))
        .sort(["user_id", "n", "last", key], descending=[False, True, True, False])
        .group_by("user_id", maintain_order=True)
        .first()
        .select("user_id", pl.col(key).alias("key"))
    )


class KeyedPopularity:
    """Most popular unseen items, globally (``key=None``) or within the user's main ``key``.

    Keyed variants serve only users with a key (category/brand from their history); cold
    users get nothing here by design - falling back to global popularity is the hybrid's job.
    """

    def __init__(
        self,
        name: str,
        spark: SparkSession,
        key: str | None = None,
        window_days: int | None = None,
        candidates: int = DEFAULT_CANDIDATES,
    ) -> None:
        self.name = name
        self._spark = spark
        self._key = key
        self._window_days = window_days
        self._candidates = candidates
        self._ranking: pl.DataFrame | None = None  # key, item_id, score
        self._user_keys: pl.DataFrame | None = None  # user_id, key
        self._seen: pl.DataFrame | None = None  # user_id, item_id

    def fit(self, data: TrainData) -> Self:
        window = self._recent(data.interactions)
        if self._key is None:
            self._ranking, self._user_keys = self._global_ranking(window), None
        else:
            self._ranking = self._keyed_ranking(window, data.items)
            self._user_keys = main_key_per_user(data.interactions, data.items, self._key)
        self._seen = data.interactions.select("user_id", "item_id").unique()
        return self

    def recommend(self, users: pl.Series, k: int) -> pl.DataFrame:
        if self._ranking is None or self._seen is None:
            raise RuntimeError(f"Model '{self.name}' must be fit before recommend")
        requested = users.unique().to_frame("user_id")
        keyed = (
            requested.with_columns(key=pl.lit(GLOBAL_KEY))
            if self._user_keys is None
            else requested.join(self._user_keys, on="user_id", how="inner")
        )
        if keyed.is_empty():
            logger.info("%s: 0 of %s users have a key", self.name, requested.height)
            return empty_recs()
        candidates = keyed.join(self._ranking, on="key").join(
            self._seen, on=["user_id", "item_id"], how="anti"
        )
        recs = top_k(candidates.select(REC_COLUMNS), k)
        self._log_coverage(recs, requested.height, k)
        return recs

    def candidate_lists(self) -> pl.DataFrame:
        """The fitted per-key candidate lists (key, item_id, score)."""
        if self._ranking is None:
            raise RuntimeError(f"Model '{self.name}' must be fit first")
        return self._ranking

    def _log_coverage(self, recs: pl.DataFrame, requested: int, k: int) -> None:
        per_user = recs.group_by("user_id").len()
        short = per_user.filter(pl.col("len") < k).height
        logger.info("%s: served %s of %s users", self.name, per_user.height, requested)
        if short:
            logger.warning(
                "%s: %s of %s served users got fewer than k=%s recs (candidate list exhausted)",
                self.name,
                short,
                per_user.height,
                k,
            )

    def _recent(self, interactions: pl.DataFrame) -> pl.DataFrame:
        if self._window_days is None:
            return interactions
        start = interactions["timestamp"].max() - timedelta(days=self._window_days)
        return interactions.filter(pl.col("timestamp") >= start)

    def _global_ranking(self, interactions: pl.DataFrame) -> pl.DataFrame:
        encoded = build_dataset(self._spark, interactions)
        model = PopRec()
        model.fit(encoded.dataset)
        popularity = pl.from_pandas(model.item_popularity.toPandas()).rename({"rating": "score"})
        ranking = decode_items(encoded.encoder, popularity).with_columns(key=pl.lit(GLOBAL_KEY))
        return _truncate(ranking, self._candidates)

    def _keyed_ranking(self, interactions: pl.DataFrame, items: pl.DataFrame) -> pl.DataFrame:
        with_key = (
            interactions.join(items.select("item_id", self._key), on="item_id", how="inner")
            .drop_nulls(self._key)
            .rename({self._key: "category"})
        )
        encoded = build_dataset(self._spark, with_key, extra_columns=("category",))
        model = CatPopRec()
        model.fit(encoded.dataset)  # fit only aggregates (category, item) popularity
        popularity = pl.from_pandas(model.cat_item_popularity.toPandas()).rename(
            {"category": "key", "rating": "score"}
        )
        return _truncate(decode_items(encoded.encoder, popularity), self._candidates)


def _truncate(ranking: pl.DataFrame, candidates: int) -> pl.DataFrame:
    """Top ``candidates`` items per key; popularity ties broken by item_id (deterministic)."""
    return (
        ranking.sort(["key", "score", "item_id"], descending=[False, True, False])
        .group_by("key", maintain_order=True)
        .head(candidates)
        .select("key", "item_id", "score")
    )
