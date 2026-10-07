from datetime import datetime

import polars as pl

from recsys.models.popularity import main_key_per_user


def test_main_key_is_most_frequent_then_most_recent():
    interactions = pl.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2"],
            "item_id": ["g1", "g2", "d1", "d1", "g1"],
            "timestamp": [datetime(2022, 1, d) for d in (1, 2, 3, 1, 5)],
        }
    )
    items = pl.DataFrame(
        {
            "item_id": ["g1", "g2", "d1"],
            "category": ["Guitars", "Guitars", "Drums"],
        }
    )

    keys = main_key_per_user(interactions, items, "category").sort("user_id")

    # u1: Guitars x2 beats Drums x1; u2: tie 1-1 -> most recent (g1, Guitars)
    assert keys.to_dicts() == [
        {"user_id": "u1", "key": "Guitars"},
        {"user_id": "u2", "key": "Guitars"},
    ]


def test_users_whose_items_lack_the_key_get_no_key():
    interactions = pl.DataFrame(
        {
            "user_id": ["u1"],
            "item_id": ["x"],
            "timestamp": [datetime(2022, 1, 1)],
        }
    )
    items = pl.DataFrame({"item_id": ["x"], "brand": [None]}, schema_overrides={"brand": pl.String})

    keys = main_key_per_user(interactions, items, "brand")

    assert keys.is_empty()
