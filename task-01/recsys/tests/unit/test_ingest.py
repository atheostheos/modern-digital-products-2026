import json
from pathlib import Path

import polars as pl
import pytest

from recsys.data.ingest import IngestError, ingest_meta, ingest_reviews


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


REVIEW = {
    "rating": 5.0,
    "title": "Five Stars",
    "text": "Great",
    "images": [],
    "asin": "A1",
    "parent_asin": "P1",
    "user_id": "U1",
    "timestamp": 1452650586000,
    "helpful_vote": 0,
    "verified_purchase": True,
}
META = {
    "main_category": "Musical Instruments",
    "title": "Drum Set",
    "average_rating": 4.2,
    "rating_number": 22,
    "features": ["f1", "f2"],
    "description": ["d1"],
    "price": None,
    "images": [],
    "videos": [],
    "store": "Pearl",
    "categories": ["Musical Instruments", "Drums & Percussion", "Drum Sets"],
    "details": {"Item Weight": "33 pounds"},
    "parent_asin": "P1",
    "bought_together": None,
}


def test_ingest_reviews_keeps_only_needed_columns(tmp_path):
    src = write_jsonl(tmp_path / "r.jsonl", [REVIEW, {**REVIEW, "user_id": "U2", "rating": 3.0}])

    out = ingest_reviews(src, tmp_path / "reviews.parquet")
    df = pl.read_parquet(out)

    assert df.columns == ["user_id", "parent_asin", "rating", "timestamp", "verified_purchase"]
    assert df.height == 2
    assert df.schema["timestamp"] == pl.Int64
    assert df["rating"].to_list() == [5.0, 3.0]


def test_ingest_meta_keeps_text_brand_category_price(tmp_path):
    priced = {**META, "parent_asin": "P2", "price": 19.99, "store": None}
    src = write_jsonl(tmp_path / "m.jsonl", [META, priced])

    df = pl.read_parquet(ingest_meta(src, tmp_path / "meta.parquet"))

    assert df.columns == [
        "parent_asin",
        "title",
        "store",
        "categories",
        "features",
        "description",
        "price",
        "average_rating",
        "rating_number",
        "bought_together",
    ]
    assert df["categories"].to_list()[0] == [
        "Musical Instruments",
        "Drums & Percussion",
        "Drum Sets",
    ]
    assert df["price"].to_list() == [None, 19.99]
    assert df["store"].to_list() == ["Pearl", None]


def test_ingest_meta_tolerates_string_price(tmp_path):
    src = write_jsonl(
        tmp_path / "m.jsonl", [{**META, "price": "from 12.50"}, {**META, "price": "7"}]
    )

    df = pl.read_parquet(ingest_meta(src, tmp_path / "meta.parquet"))

    assert df["price"].to_list() == [None, 7.0]


def test_ingest_fails_clearly_when_source_missing(tmp_path):
    with pytest.raises(IngestError, match="not found"):
        ingest_reviews(tmp_path / "missing.jsonl", tmp_path / "out.parquet")


def test_ingest_fails_clearly_when_required_column_missing(tmp_path):
    bad = {k: v for k, v in REVIEW.items() if k != "user_id"}
    src = write_jsonl(tmp_path / "r.jsonl", [bad])

    with pytest.raises(IngestError, match="user_id"):
        ingest_reviews(src, tmp_path / "out.parquet")


def test_ingest_rejects_null_ratings(tmp_path):
    src = write_jsonl(tmp_path / "r.jsonl", [{**REVIEW, "rating": None}])

    with pytest.raises(IngestError, match="rating"):
        ingest_reviews(src, tmp_path / "out.parquet")


def test_ingest_rejects_timestamps_in_seconds(tmp_path):
    src = write_jsonl(tmp_path / "r.jsonl", [{**REVIEW, "timestamp": 1452650586}])

    with pytest.raises(IngestError, match="timestamp"):
        ingest_reviews(src, tmp_path / "out.parquet")


def test_ingest_wraps_malformed_json_lines(tmp_path):
    src = tmp_path / "r.jsonl"
    src.write_text('{"user_id": "U1", "parent_asin": \n')

    with pytest.raises(IngestError, match="parse"):
        ingest_reviews(src, tmp_path / "out.parquet")
