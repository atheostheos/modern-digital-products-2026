"""Adapter that runs any RePlay (PySpark) recommender behind the ``Recommender`` contract."""

import logging
from collections.abc import Callable
from typing import Self

import polars as pl
from pyspark.sql import SparkSession
from replay.data import Dataset, FeatureHint, FeatureInfo, FeatureSchema, FeatureType
from replay.models.base_rec import BaseRecommender
from replay.preprocessing import LabelEncoder, LabelEncodingRule

from recsys.models.base import REC_COLUMNS, TrainData

logger = logging.getLogger(__name__)

ID_COLUMNS = ("user_id", "item_id")
SCHEMA = FeatureSchema(
    [
        FeatureInfo("user_id", FeatureType.CATEGORICAL, FeatureHint.QUERY_ID),
        FeatureInfo("item_id", FeatureType.CATEGORICAL, FeatureHint.ITEM_ID),
        FeatureInfo("rating", FeatureType.NUMERICAL, FeatureHint.RATING),
        FeatureInfo("timestamp", FeatureType.NUMERICAL, FeatureHint.TIMESTAMP),
    ]
)


class ReplayModel:
    """Encodes string ids, fits a RePlay model in Spark and decodes its top-k back to polars."""

    def __init__(
        self, name: str, factory: Callable[[], BaseRecommender], spark: SparkSession
    ) -> None:
        self.name = name
        self._factory = factory
        self._spark = spark
        self._encoder: LabelEncoder | None = None
        self._model: BaseRecommender | None = None
        self._dataset: Dataset | None = None

    def fit(self, data: TrainData) -> Self:
        interactions = data.interactions.select(*ID_COLUMNS, "rating", "timestamp")
        encoder = LabelEncoder(
            [LabelEncodingRule(c, handle_unknown="drop") for c in ID_COLUMNS]
        ).fit(interactions)
        encoded = encoder.transform(interactions)
        dataset = Dataset(
            feature_schema=SCHEMA,
            interactions=self._spark.createDataFrame(encoded.to_pandas()),
        )
        model = self._factory()
        model.fit(dataset)
        self._encoder, self._model, self._dataset = encoder, model, dataset
        return self

    def recommend(self, users: pl.Series, k: int) -> pl.DataFrame:
        if self._model is None or self._encoder is None or self._dataset is None:
            raise RuntimeError(f"Model '{self.name}' must be fit before recommend")
        user_codes = self._encoder.mapping["user_id"]
        known = (
            users.unique()
            .replace_strict(user_codes, default=None, return_dtype=pl.Int64)
            .drop_nulls()
            .alias("user_id")
        )
        logger.info(
            "%s: %s of %s requested users are known to the model",
            self.name,
            f"{known.len():,}",
            f"{users.n_unique():,}",
        )
        if known.is_empty():
            return pl.DataFrame(
                schema={"user_id": pl.String, "item_id": pl.String, "score": pl.Float64}
            )
        queries = self._spark.createDataFrame(known.to_frame().to_pandas())
        predicted = self._model.predict(self._dataset, k=k, queries=queries, filter_seen_items=True)
        recs = pl.from_pandas(predicted.toPandas()).rename({"rating": "score"})
        return (
            self._encoder.inverse_transform(recs)
            .select(REC_COLUMNS)
            .sort(["user_id", "score"], descending=[False, True])
        )
