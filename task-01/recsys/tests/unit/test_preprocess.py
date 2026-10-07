from datetime import UTC, datetime

import polars as pl

from recsys.data.preprocess import build_interactions, build_items

MS_2020 = int(datetime(2020, 1, 1, tzinfo=UTC).timestamp() * 1000)
DAY_MS = 86_400_000


def reviews(rows: list[tuple[str, str, float, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "user_id": pl.String,
            "parent_asin": pl.String,
            "rating": pl.Float64,
            "timestamp": pl.Int64,
        },
        orient="row",
    ).with_columns(verified_purchase=pl.lit(True))


def test_interactions_rename_item_and_convert_timestamp():
    df = build_interactions(reviews([("u1", "p1", 5.0, MS_2020)]))

    assert df.columns == ["user_id", "item_id", "timestamp", "rating", "verified_purchase"]
    assert df["timestamp"].to_list() == [datetime(2020, 1, 1)]


def test_interactions_keep_latest_review_per_user_item_pair():
    raw = reviews(
        [
            ("u1", "p1", 2.0, MS_2020),
            ("u1", "p1", 5.0, MS_2020 + DAY_MS),
            ("u2", "p1", 4.0, MS_2020),
        ]
    )

    df = build_interactions(raw).sort("user_id")

    assert df.height == 2
    assert df.filter(pl.col("user_id") == "u1")["rating"].item() == 5.0


def test_interactions_dedup_is_deterministic_on_timestamp_ties():
    raw = reviews([("u1", "p1", 2.0, MS_2020), ("u1", "p1", 5.0, MS_2020)])

    df = build_interactions(raw)

    assert df["rating"].to_list() == [5.0]  # last row in input order wins


def test_interactions_do_not_mutate_input():
    raw = reviews([("u1", "p1", 5.0, MS_2020)])
    before = raw.clone()

    build_interactions(raw)

    assert raw.equals(before)


META_COLUMNS = {
    "parent_asin": pl.String,
    "title": pl.String,
    "store": pl.String,
    "categories": pl.List(pl.String),
    "features": pl.List(pl.String),
    "description": pl.List(pl.String),
    "price": pl.Float64,
    "average_rating": pl.Float64,
    "rating_number": pl.Int64,
    "bought_together": pl.List(pl.String),
}


def meta(rows: list[dict]) -> pl.DataFrame:
    defaults = {
        "title": "t",
        "store": None,
        "categories": [],
        "features": [],
        "description": [],
        "price": None,
        "average_rating": None,
        "rating_number": None,
        "bought_together": None,
    }
    return pl.DataFrame([{**defaults, **r} for r in rows], schema=META_COLUMNS)


def test_items_extract_brand_category_levels_and_text():
    df = build_items(
        meta(
            [
                {
                    "parent_asin": "p1",
                    "title": "Drum Set",
                    "store": "  Pearl ",
                    "categories": ["Musical Instruments", "Drums & Percussion", "Drum Sets"],
                    "features": ["Maple shells"],
                    "description": ["Best seller"],
                    "price": 699.0,
                }
            ]
        )
    )

    row = df.row(0, named=True)
    assert row["item_id"] == "p1"
    assert row["brand"] == "Pearl"
    assert row["category"] == "Drum Sets"
    assert row["top_category"] == "Drums & Percussion"
    assert row["text"] == "Drum Set Maple shells Best seller Drums & Percussion Drum Sets"
    assert row["price"] == 699.0


def test_items_without_categories_or_brand_get_nulls():
    df = build_items(meta([{"parent_asin": "p1", "store": "  ", "categories": []}]))

    row = df.row(0, named=True)
    assert row["brand"] is None
    assert row["category"] is None
    assert row["top_category"] is None


def test_items_are_unique_by_item_id():
    df = build_items(meta([{"parent_asin": "p1"}, {"parent_asin": "p1", "title": "dup"}]))

    assert df.height == 1


def test_items_normalize_empty_lists_and_text_to_null():
    df = build_items(meta([{"parent_asin": "p1", "title": None, "bought_together": []}]))

    row = df.row(0, named=True)
    assert row["bought_together"] is None
    assert row["text"] is None
