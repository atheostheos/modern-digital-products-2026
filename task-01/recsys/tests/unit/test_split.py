from datetime import date, datetime

import polars as pl
import pytest

from recsys.config import SplitConfig
from recsys.data.split import SplitError, split_by_time

SPLIT = SplitConfig(
    val_start=date(2022, 10, 1), test_start=date(2023, 1, 1), test_end=date(2023, 4, 1)
)


def interactions(timestamps: list[datetime]) -> pl.DataFrame:
    n = len(timestamps)
    return pl.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(n)],
            "item_id": [f"i{i}" for i in range(n)],
            "timestamp": timestamps,
            "rating": [5.0] * n,
        }
    )


def test_split_assigns_rows_by_half_open_windows():
    df = interactions(
        [
            datetime(2022, 9, 30, 23, 59),  # train
            datetime(2022, 10, 1),  # val (boundary belongs to the later window)
            datetime(2022, 12, 31, 23, 59),  # val
            datetime(2023, 1, 1),  # test
            datetime(2023, 3, 31, 23, 59),  # test
            datetime(2023, 4, 1),  # dropped (>= test_end)
        ]
    )

    splits = split_by_time(df, SPLIT)

    assert splits.train["user_id"].to_list() == ["u0"]
    assert sorted(splits.val["user_id"].to_list()) == ["u1", "u2"]
    assert sorted(splits.test["user_id"].to_list()) == ["u3", "u4"]


def test_split_keeps_cold_users_and_items_in_later_windows():
    df = pl.DataFrame(
        {
            "user_id": ["warm", "warm", "cold"],
            "item_id": ["old", "new", "old"],
            "timestamp": [datetime(2022, 1, 1), datetime(2023, 2, 1), datetime(2023, 2, 2)],
            "rating": [5.0, 5.0, 5.0],
        }
    ).vstack(interactions([datetime(2022, 11, 1)]))  # keeps the val window non-empty

    splits = split_by_time(df, SPLIT)

    assert splits.test.height == 2  # cold user and unseen item are not dropped


def test_split_does_not_mutate_input():
    df = interactions([datetime(2022, 1, 1), datetime(2022, 11, 1), datetime(2023, 2, 1)])
    before = df.clone()

    split_by_time(df, SPLIT)

    assert df.equals(before)


def test_split_raises_when_a_window_is_empty():
    df = interactions([datetime(2022, 1, 1), datetime(2023, 2, 1)])  # nothing in val

    with pytest.raises(SplitError, match="val"):
        split_by_time(df, SPLIT)
