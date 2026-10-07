"""Shared plumbing between polars frames with string ids and RePlay's Spark ``Dataset``."""

from dataclasses import dataclass

import polars as pl
from pyspark.sql import SparkSession
from replay.data import Dataset, FeatureHint, FeatureInfo, FeatureSchema, FeatureType
from replay.preprocessing import LabelEncoder, LabelEncodingRule

ID_COLUMNS = ("user_id", "item_id")
BASE_FEATURES = [
    FeatureInfo("user_id", FeatureType.CATEGORICAL, FeatureHint.QUERY_ID),
    FeatureInfo("item_id", FeatureType.CATEGORICAL, FeatureHint.ITEM_ID),
    FeatureInfo("rating", FeatureType.NUMERICAL, FeatureHint.RATING),
    FeatureInfo("timestamp", FeatureType.NUMERICAL, FeatureHint.TIMESTAMP),
]


@dataclass(frozen=True)
class EncodedDataset:
    encoder: LabelEncoder  # string ids <-> contiguous ints used inside RePlay
    dataset: Dataset


def build_dataset(
    spark: SparkSession, interactions: pl.DataFrame, extra_columns: tuple[str, ...] = ()
) -> EncodedDataset:
    """Encode ids and build a RePlay Dataset; ``extra_columns`` become categorical features."""
    frame = interactions.select(*ID_COLUMNS, "rating", "timestamp", *extra_columns)
    encoder = LabelEncoder([LabelEncodingRule(c, handle_unknown="drop") for c in ID_COLUMNS]).fit(
        frame
    )
    schema = FeatureSchema(
        [*BASE_FEATURES, *(FeatureInfo(c, FeatureType.CATEGORICAL) for c in extra_columns)]
    )
    dataset = Dataset(
        feature_schema=schema,
        interactions=spark.createDataFrame(encoder.transform(frame).to_pandas()),
    )
    return EncodedDataset(encoder=encoder, dataset=dataset)


def encode_users(encoder: LabelEncoder, users: pl.Series) -> pl.Series:
    """Integer codes of the users known to ``encoder`` (unknown users are dropped)."""
    return (
        users.unique()
        .replace_strict(encoder.mapping["user_id"], default=None, return_dtype=pl.Int64)
        .drop_nulls()
        .alias("user_id")
    )


def decode_items(encoder: LabelEncoder, frame: pl.DataFrame) -> pl.DataFrame:
    """Replace integer ``item_id`` codes in ``frame`` with the original string ids."""
    inverse = encoder.inverse_mapping["item_id"]
    decoded = frame.with_columns(
        pl.col("item_id").replace_strict(inverse, default=None, return_dtype=pl.String)
    )
    unknown = decoded["item_id"].null_count() - frame["item_id"].null_count()
    if unknown:
        raise ValueError(f"{unknown} unknown item codes cannot be decoded")
    return decoded
