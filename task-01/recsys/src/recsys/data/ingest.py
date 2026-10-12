"""Raw Amazon Reviews 2023 JSONL → typed parquet with only the columns we use."""

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

TextList = pl.List(pl.String)

REVIEWS_SCHEMA: dict[str, pl.DataType] = {
    "user_id": pl.String(),
    "parent_asin": pl.String(),
    "rating": pl.Float64(),
    "timestamp": pl.Int64(),  # epoch milliseconds
    "verified_purchase": pl.Boolean(),
}
REVIEWS_KEYS = ("user_id", "parent_asin", "timestamp", "rating")
# Amazon reviews start in the late 1990s; smaller values mean seconds, not milliseconds
MIN_PLAUSIBLE_TIMESTAMP_MS = 820_454_400_000  # 1996-01-01T00:00:00Z

# price is read as text because the raw field mixes numbers, null and strings
META_SCHEMA: dict[str, pl.DataType] = {
    "parent_asin": pl.String(),
    "title": pl.String(),
    "store": pl.String(),
    "categories": TextList,
    "features": TextList,
    "description": TextList,
    "price": pl.String(),
    "average_rating": pl.Float64(),
    "rating_number": pl.Int64(),
    "bought_together": TextList,
}
META_KEYS = ("parent_asin",)


class IngestError(RuntimeError):
    """Raised when a raw file is missing or violates the expected schema."""


def ingest_reviews(src: Path, dst: Path) -> Path:
    """Convert the reviews JSONL to parquet; fails if any key column has nulls."""
    df = _read(src, REVIEWS_SCHEMA)
    _require_non_null(df, REVIEWS_KEYS, src)
    _require_millisecond_timestamps(df, src)
    return _write(df, dst)


def ingest_meta(src: Path, dst: Path) -> Path:
    """Convert the metadata JSONL to parquet with a numeric ``price`` (null if unparsable)."""
    df = _read(src, META_SCHEMA).with_columns(
        pl.col("price").str.strip_chars().cast(pl.Float64, strict=False)
    )
    _require_non_null(df, META_KEYS, src)
    return _write(df, dst)


def _read(src: Path, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if not src.is_file():
        raise IngestError(f"Source file not found: {src}")
    logger.info("Reading %s", src)
    try:
        return pl.scan_ndjson(src, schema=schema).collect()
    except pl.exceptions.PolarsError as exc:
        raise IngestError(f"Failed to parse {src}: {exc}") from exc


def _require_non_null(df: pl.DataFrame, keys: tuple[str, ...], src: Path) -> None:
    nulls = {k: n for k in keys if (n := df[k].null_count()) > 0}
    if nulls:
        raise IngestError(f"Null values in required columns {nulls} in {src}")


def _require_millisecond_timestamps(df: pl.DataFrame, src: Path) -> None:
    earliest = df["timestamp"].min()
    if earliest is not None and earliest < MIN_PLAUSIBLE_TIMESTAMP_MS:
        raise IngestError(
            f"Implausible timestamp {earliest} in {src}: expected epoch milliseconds "
            f">= {MIN_PLAUSIBLE_TIMESTAMP_MS}"
        )


def _write(df: pl.DataFrame, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(dst)
    logger.info("Wrote %s rows to %s", f"{df.height:,}", dst)
    return dst
