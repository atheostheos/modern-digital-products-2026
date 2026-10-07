"""Global temporal train / val / test split (RePlay ``TimeSplitter``)."""

from dataclasses import dataclass
from datetime import date, datetime, time

import polars as pl
from replay.splitters import TimeSplitter

from recsys.config import SplitConfig


class SplitError(ValueError):
    """Raised when a split window contains no interactions."""


@dataclass(frozen=True)
class Splits:
    train: pl.DataFrame  # timestamp < val_start
    val: pl.DataFrame  # val_start <= timestamp < test_start
    test: pl.DataFrame  # test_start <= timestamp < test_end


def split_by_time(interactions: pl.DataFrame, split: SplitConfig) -> Splits:
    """Split by global time thresholds; cold users/items stay in later windows."""
    in_range = interactions.filter(pl.col("timestamp") < _midnight(split.test_end))
    before_test, test = _cut(in_range, split.test_start)
    train, val = _cut(before_test, split.val_start)
    splits = Splits(train=train, val=val, test=test)
    _require_non_empty(splits, split)
    return splits


def _require_non_empty(splits: Splits, split: SplitConfig) -> None:
    empty = [name for name in ("train", "val", "test") if getattr(splits, name).is_empty()]
    if empty:
        raise SplitError(f"Empty split window(s) {empty} for {split.model_dump(mode='json')}")


def _cut(df: pl.DataFrame, at: date) -> tuple[pl.DataFrame, pl.DataFrame]:
    splitter = TimeSplitter(
        time_threshold=_midnight(at),
        query_column="user_id",
        item_column="item_id",
        timestamp_column="timestamp",
        drop_cold_users=False,
        drop_cold_items=False,
    )
    return splitter.split(df)


def _midnight(day: date) -> datetime:
    return datetime.combine(day, time.min)
