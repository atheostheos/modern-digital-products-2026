"""Common model contract: every recommender (library-backed or own) is evaluated the same way."""

from dataclasses import dataclass
from typing import Protocol, Self

import polars as pl

REC_COLUMNS = ["user_id", "item_id", "score"]


@dataclass(frozen=True)
class TrainData:
    interactions: pl.DataFrame  # user_id, item_id, timestamp, rating
    items: pl.DataFrame  # item_id + content features (category, brand, text, ...)


class Recommender(Protocol):
    name: str

    def fit(self, data: TrainData) -> Self:
        """Fit on training data (inputs are never modified) and return the fitted model."""
        ...

    def recommend(self, users: pl.Series, k: int) -> pl.DataFrame:
        """Top-k unseen items per user as ``REC_COLUMNS``; users it cannot serve are omitted."""
        ...
