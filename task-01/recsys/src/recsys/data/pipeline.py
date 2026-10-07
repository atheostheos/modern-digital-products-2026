"""``prepare`` pipeline: raw JSONL → interim parquet → processed splits + dataset stats."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from recsys.config import Config
from recsys.data.download import dataset_files
from recsys.data.ingest import ingest_meta, ingest_reviews
from recsys.data.preprocess import build_interactions, build_items
from recsys.data.split import Splits, split_by_time
from recsys.data.stats import history_coverage, interaction_stats, item_feature_coverage

logger = logging.getLogger(__name__)

STATS_FILE = Path("data") / "dataset_stats.json"


@dataclass(frozen=True)
class PrepareResult:
    splits: Splits
    items: pl.DataFrame
    stats: dict[str, Any]


def prepare(cfg: Config) -> PrepareResult:
    """Run the data pipeline for an already-resolved config (absolute paths)."""
    data = cfg.data
    files = dataset_files(data.category)
    reviews = pl.read_parquet(
        ingest_reviews(data.download_dir / files.reviews, data.interim_dir / "reviews.parquet")
    )
    meta = pl.read_parquet(
        ingest_meta(data.download_dir / files.meta, data.interim_dir / "meta.parquet")
    )
    interactions = build_interactions(reviews)
    items = build_items(meta)
    splits = split_by_time(interactions, cfg.split)

    _write_processed(splits, items, data.processed_dir)
    stats = _collect_stats(reviews, interactions, items, splits)
    _write_json(stats, cfg.results_dir / STATS_FILE)
    return PrepareResult(splits=splits, items=items, stats=stats)


def _write_processed(splits: Splits, items: pl.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = {"train": splits.train, "val": splits.val, "test": splits.test, "items": items}
    for name, df in frames.items():
        df.write_parquet(out_dir / f"{name}.parquet")
        logger.info("Wrote %s: %s rows", name, f"{df.height:,}")


def _collect_stats(
    reviews: pl.DataFrame, interactions: pl.DataFrame, items: pl.DataFrame, splits: Splits
) -> dict[str, Any]:
    kept = splits.train.height + splits.val.height + splits.test.height
    return {
        "raw_reviews": reviews.height,
        "dropped_after_test_end": interactions.height - kept,
        "interactions": interaction_stats(interactions),
        "train": interaction_stats(splits.train),
        "items_in_meta": items.height,
        "interaction_items_missing_from_meta": interactions.join(items, on="item_id", how="anti")[
            "item_id"
        ].n_unique(),
        "distinct_brands": items["brand"].drop_nulls().n_unique(),
        "distinct_categories": items["category"].drop_nulls().n_unique(),
        "distinct_top_categories": items["top_category"].drop_nulls().n_unique(),
        "item_features": item_feature_coverage(items),
        "splits": history_coverage(splits),
    }


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    logger.info("Wrote %s", path)
