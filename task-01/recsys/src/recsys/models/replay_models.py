"""Adapter that runs any personalized RePlay (PySpark) recommender behind ``Recommender``."""

import logging
from collections.abc import Callable
from typing import Self

import polars as pl
from pyspark.sql import SparkSession
from replay.models.base_rec import BaseRecommender

from recsys.models.base import REC_COLUMNS, TrainData, empty_recs
from recsys.models.replay_data import EncodedDataset, build_dataset, encode_users

logger = logging.getLogger(__name__)


class ReplayModel:
    """Encodes string ids, fits a RePlay model in Spark and decodes its top-k back to polars."""

    def __init__(
        self, name: str, factory: Callable[[], BaseRecommender], spark: SparkSession
    ) -> None:
        self.name = name
        self._factory = factory
        self._spark = spark
        self._data: EncodedDataset | None = None
        self._model: BaseRecommender | None = None

    def fit(self, data: TrainData) -> Self:
        encoded = build_dataset(self._spark, data.interactions)
        model = self._factory()
        model.fit(encoded.dataset)
        self._data, self._model = encoded, model
        return self

    def recommend(self, users: pl.Series, k: int) -> pl.DataFrame:
        if self._model is None or self._data is None:
            raise RuntimeError(f"Model '{self.name}' must be fit before recommend")
        known = encode_users(self._data.encoder, users)
        logger.info(
            "%s: %s of %s requested users are known to the model",
            self.name,
            f"{known.len():,}",
            f"{users.n_unique():,}",
        )
        if known.is_empty():
            return empty_recs()
        queries = self._spark.createDataFrame(known.to_frame().to_pandas())
        predicted = self._model.predict(
            self._data.dataset, k=k, queries=queries, filter_seen_items=True
        )
        recs = pl.from_pandas(predicted.toPandas()).rename({"rating": "score"})
        return (
            self._data.encoder.inverse_transform(recs)
            .select(REC_COLUMNS)
            .sort(["user_id", "score"], descending=[False, True])
        )
