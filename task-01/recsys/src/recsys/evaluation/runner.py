"""Experiment runner: fit configured models on a stage's train data and evaluate them.

Stages:
- ``val``:  fit on train,       evaluate on val  (model selection / tuning)
- ``test``: fit on train + val, evaluate on test (final numbers)

Timing note: Spark is lazy, so part of a model's fitting work can be executed during
its first ``recommend`` call; ``fit_seconds`` + ``recommend_seconds`` is the reliable total.
"""

import logging
import time as clock
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Literal

import polars as pl
from pyspark.sql import SparkSession

from recsys.config import Config
from recsys.evaluation.evaluator import EvalWindow, evaluate
from recsys.models.base import TrainData
from recsys.models.registry import BuildContext, build_model

logger = logging.getLogger(__name__)

Stage = Literal["val", "test"]
PROCESSED_FRAMES = ("train", "val", "test", "items")
# Novelty is omitted: every model filters seen items, so RePlay Novelty is always 1.0
SUMMARY_METRICS = (
    "HitRate",
    "Recall",
    "NDCG",
    "MAP",
    "Coverage",
    "Surprisal",
    "Gini",
    "LongTailShare",
)
SUMMARY_K = 10


class UnknownModelError(ValueError):
    """Raised when a requested model name is not configured."""


@dataclass(frozen=True)
class StageData:
    train: TrainData
    window: EvalWindow


@dataclass(frozen=True)
class ResultPaths:
    parquet: Path  # source of truth (types and NaN preserved)
    csv: Path
    markdown: Path


def validate_models(cfg: Config, model_names: list[str]) -> None:
    """Fail fast (before any Spark work) on names missing from the config."""
    unknown = [n for n in model_names if n not in cfg.models]
    if unknown:
        raise UnknownModelError(f"Unknown model(s) {unknown}; configured: {sorted(cfg.models)}")


@dataclass(frozen=True)
class TrialBudget:
    """Cheaper trials for sweeps: score a user sample and/or fit on recent history only.

    Winners should be confirmed with the full budget (the default).
    """

    sample_users: int | None = None  # seeded random sample of ground-truth users
    train_days: int | None = None  # history = last N days before the window start

    def __post_init__(self) -> None:
        for name in ("sample_users", "train_days"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")


FULL_BUDGET = TrialBudget()


def load_stage(cfg: Config, stage: Stage, budget: TrialBudget = FULL_BUDGET) -> StageData:
    """Train data and evaluation window for ``stage`` from the processed parquet files."""
    paths = {n: cfg.data.processed_dir / f"{n}.parquet" for n in PROCESSED_FRAMES}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Processed data missing: {missing}")
    train, val, test, items = (pl.read_parquet(paths[n]) for n in PROCESSED_FRAMES)
    history, window, window_start = (
        (train, val, cfg.split.val_start)
        if stage == "val"
        else (pl.concat([train, val]), test, cfg.split.test_start)
    )
    history = _recent_history(history, window_start, budget.train_days)
    truth = _sample_users(window.select("user_id", "item_id"), budget.sample_users, cfg.seed)
    return StageData(
        train=TrainData(interactions=history, items=items),
        window=EvalWindow(
            ground_truth=truth, history=history.select("user_id", "item_id"), items=items
        ),
    )


def _recent_history(history: pl.DataFrame, window_start: date, days: int | None) -> pl.DataFrame:
    if days is None:
        return history
    start = datetime.combine(window_start - timedelta(days=days), time.min)
    return history.filter(pl.col("timestamp") >= start)


def _sample_users(truth: pl.DataFrame, n: int | None, seed: int) -> pl.DataFrame:
    users = truth["user_id"].unique().sort()
    if n is None or n >= users.len():
        return truth
    sample = users.sample(n=n, seed=seed)
    return truth.filter(pl.col("user_id").is_in(sample.implode()))


def run_experiment(
    cfg: Config, data: StageData, model_names: list[str], spark: SparkSession
) -> pl.DataFrame:
    """Fit + evaluate each named model; one row per (model, segment) with timings."""
    validate_models(cfg, model_names)
    tables = [_run_one(name, cfg, data, spark) for name in model_names]
    return pl.concat(tables, how="diagonal_relaxed")


def _run_one(name: str, cfg: Config, data: StageData, spark: SparkSession) -> pl.DataFrame:
    users = data.window.ground_truth["user_id"].unique()
    k = max(cfg.evaluation.ks)
    ctx = BuildContext(spark=spark, seed=cfg.seed, threads=cfg.spark.cores, catalog=cfg.models)
    model = build_model(name, cfg.models[name], ctx)
    started = clock.perf_counter()
    try:
        model.fit(data.train)
        fitted = clock.perf_counter()
        recs = model.recommend(users, k)
        recommended = clock.perf_counter()
    finally:
        # RePlay models cache Spark DataFrames and never release them; without this,
        # caches of earlier models accumulate until Spark runs out of memory
        _release_spark_caches(spark)
    logger.info("%s: fit %.1fs, recommend %.1fs", name, fitted - started, recommended - fitted)
    return (
        evaluate(recs, data.window, cfg.evaluation)
        .with_columns(
            model=pl.lit(name),
            fit_seconds=pl.lit(round(fitted - started, 2)),
            recommend_seconds=pl.lit(round(recommended - fitted, 2)),
            run_at=pl.lit(datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")),
        )
        .select("model", pl.exclude("model"))
    )


def _release_spark_caches(spark: SparkSession) -> None:
    spark.catalog.clearCache()


def write_results(table: pl.DataFrame, results_dir: Path, name: str) -> ResultPaths:
    """Merge into ``metrics/<name>.parquet`` (re-run models replace their rows), then
    regenerate the CSV and markdown views from it. Each file is written atomically."""
    out_dir = results_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = ResultPaths(
        parquet=out_dir / f"{name}.parquet",
        csv=out_dir / f"{name}.csv",
        markdown=out_dir / f"{name}.md",
    )
    if paths.parquet.is_file():
        previous = pl.read_parquet(paths.parquet)
        kept = previous.filter(~pl.col("model").is_in(table["model"].unique().implode()))
        table = pl.concat([kept, table], how="diagonal_relaxed")
    _atomic_write(paths.parquet, table.write_parquet)
    _atomic_write(paths.csv, table.write_csv)
    _atomic_write(paths.markdown, lambda p: p.write_text(_markdown_summary(table, name)))
    logger.info("Wrote %s (+ .csv, .md)", paths.parquet)
    return paths


def _atomic_write(path: Path, write: Callable[[Path], object]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    write(tmp)
    tmp.replace(path)


def _markdown_summary(table: pl.DataFrame, name: str) -> str:
    metrics = [f"{m}@{SUMMARY_K}" for m in SUMMARY_METRICS if f"{m}@{SUMMARY_K}" in table.columns]
    columns = ["model", "segment", "users", *metrics]
    header = f"# {name} metrics @{SUMMARY_K}\n\n| " + " | ".join(columns) + " |\n"
    divider = "|" + "---|" * len(columns) + "\n"
    rows = "".join(
        "| " + " | ".join(_fmt(row[c]) for c in columns) + " |\n"
        for row in table.select(columns).iter_rows(named=True)
    )
    return header + divider + rows


def _fmt(value: object) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)
