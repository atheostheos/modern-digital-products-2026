"""Experiment runner: fit configured models on a stage's train data and evaluate them.

Stages:
- ``val``:  fit on train,       evaluate on val  (model selection / tuning)
- ``test``: fit on train + val, evaluate on test (final numbers)

Timing note: Spark is lazy, so part of a model's fitting work can be executed during
its first ``recommend`` call; ``fit_seconds`` + ``recommend_seconds`` is the reliable total.
"""

import json
import logging
import math
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
from recsys.evaluation.metrics import top_k
from recsys.models.base import TrainData
from recsys.models.hybrid import fuse
from recsys.models.registry import BuildContext, build_model
from recsys.models.specs import HybridSpec
from recsys.spark import clear_caches

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


@dataclass(frozen=True)
class RunOutput:
    recs: pl.DataFrame
    fit_seconds: float
    recommend_seconds: float


def run_experiment(
    cfg: Config,
    data: StageData,
    model_names: list[str],
    spark: SparkSession,
    recs_dir: Path | None = None,
) -> pl.DataFrame:
    """Fit + evaluate each named model; one row per (model, segment) with timings.

    Hybrids run last; when all their components are part of the same experiment, they are
    fused from the components' cached recommendations instead of re-fitting them (their
    timings are then the components' totals plus the fusion). With ``recs_dir``, each
    model's top-k lists are saved to ``<recs_dir>/<model>.parquet`` for later analysis.
    """
    validate_models(cfg, model_names)
    ordered = sorted(model_names, key=lambda n: isinstance(cfg.models[n], HybridSpec))
    k = max(cfg.evaluation.ks)
    depths = _component_depths(cfg, ordered, k)
    cache: dict[str, RunOutput] = {}
    tables = []
    for name in ordered:
        output = _fused_from_cache(name, cfg, cache, k) or _fit_and_recommend(
            name, cfg, data, spark, depths.get(name, k)
        )
        if name in depths:
            cache[name] = output
        if recs_dir is not None:
            _save_recs(top_k(output.recs, k), recs_dir / f"{name}.parquet")
        tables.append(_evaluate_output(name, output, data, cfg))
    return pl.concat(tables, how="diagonal_relaxed")


def _component_depths(cfg: Config, names: list[str], k: int) -> dict[str, int]:
    """How deep each hybrid component of this experiment must recommend."""
    depths: dict[str, int] = {}
    for name in names:
        spec = cfg.models[name]
        if isinstance(spec, HybridSpec):
            for part in spec.components:
                if part in names:
                    depths[part] = max(depths.get(part, k), spec.depth)
    return depths


def _fused_from_cache(
    name: str, cfg: Config, cache: dict[str, RunOutput], k: int
) -> RunOutput | None:
    spec = cfg.models[name]
    if not isinstance(spec, HybridSpec) or not all(p in cache for p in spec.components):
        return None
    started = clock.perf_counter()
    recs = fuse(
        [(cache[p].recs, w) for p, w in spec.components.items()],
        k=k,
        depth=spec.depth,
        rrf_k=spec.rrf_k,
    )
    fusion = clock.perf_counter() - started
    parts = [cache[p] for p in spec.components]
    logger.info("%s: fused from cached components %s in %.1fs", name, list(spec.components), fusion)
    return RunOutput(
        recs=recs,
        fit_seconds=sum(p.fit_seconds for p in parts),
        recommend_seconds=sum(p.recommend_seconds for p in parts) + fusion,
    )


def _fit_and_recommend(
    name: str, cfg: Config, data: StageData, spark: SparkSession, depth: int
) -> RunOutput:
    users = data.window.ground_truth["user_id"].unique()
    ctx = BuildContext(spark=spark, seed=cfg.seed, threads=cfg.spark.cores, catalog=cfg.models)
    model = build_model(name, cfg.models[name], ctx)
    started = clock.perf_counter()
    try:
        model.fit(data.train)
        fitted = clock.perf_counter()
        recs = model.recommend(users, depth)
        recommended = clock.perf_counter()
    finally:
        # RePlay models cache Spark DataFrames and never release them; without this,
        # caches of earlier models accumulate until Spark runs out of memory
        clear_caches(spark)
    logger.info("%s: fit %.1fs, recommend %.1fs", name, fitted - started, recommended - fitted)
    return RunOutput(recs, fitted - started, recommended - fitted)


def _evaluate_output(name: str, output: RunOutput, data: StageData, cfg: Config) -> pl.DataFrame:
    return (
        evaluate(output.recs, data.window, cfg.evaluation)
        .with_columns(
            model=pl.lit(name),
            fit_seconds=pl.lit(round(output.fit_seconds, 2)),
            recommend_seconds=pl.lit(round(output.recommend_seconds, 2)),
            run_at=pl.lit(datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")),
        )
        .select("model", pl.exclude("model"))
    )


MANIFEST = "manifest.json"


def write_recs_manifest(recs_dir: Path, stage: Stage, budget: TrialBudget) -> Path:
    """Record how saved lists were produced so analyses rebuild the same window."""
    path = recs_dir / MANIFEST
    payload = {"stage": stage, "sample_users": budget.sample_users, "train_days": budget.train_days}
    _atomic_write(path, lambda p: p.write_text(json.dumps(payload, indent=2) + "\n"))
    return path


def read_recs_manifest(recs_dir: Path) -> TrialBudget:
    """The budget the lists in ``recs_dir`` were made with (full budget if unrecorded)."""
    path = recs_dir / MANIFEST
    if not path.is_file():
        logger.warning("%s missing: assuming lists were made with the full budget", path)
        return FULL_BUDGET
    payload = json.loads(path.read_text())
    return TrialBudget(sample_users=payload["sample_users"], train_days=payload["train_days"])


def _save_recs(recs: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, recs.write_parquet)


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


def write_table(table: pl.DataFrame, stem: Path, title: str) -> ResultPaths:
    """Write ``<stem>.parquet`` (source of truth), ``.csv`` and a full ``.md`` table."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = ResultPaths(
        parquet=stem.with_suffix(".parquet"),
        csv=stem.with_suffix(".csv"),
        markdown=stem.with_suffix(".md"),
    )
    _atomic_write(paths.parquet, table.write_parquet)
    _atomic_write(paths.csv, table.write_csv)
    _atomic_write(paths.markdown, lambda p: p.write_text(_markdown(table, title)))
    return paths


def _markdown(table: pl.DataFrame, title: str) -> str:
    header = f"# {title}\n\n| " + " | ".join(table.columns) + " |\n"
    divider = "|" + "---|" * len(table.columns) + "\n"
    rows = "".join("| " + " | ".join(_fmt(v) for v in row) + " |\n" for row in table.iter_rows())
    return header + divider + rows


def _atomic_write(path: Path, write: Callable[[Path], object]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    write(tmp)
    tmp.replace(path)


def _markdown_summary(table: pl.DataFrame, name: str) -> str:
    metrics = [f"{m}@{SUMMARY_K}" for m in SUMMARY_METRICS if f"{m}@{SUMMARY_K}" in table.columns]
    columns = ["model", "segment", "users", *metrics]
    return _markdown(table.select(columns), f"{name} metrics @{SUMMARY_K}")


MISSING = "-"
LARGE = 1000.0  # render as a grouped integer from here on
SMALL = 0.001  # below this, scientific notation keeps the digits visible


def _fmt(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return MISSING
    if not isinstance(value, float):
        return str(value)
    if abs(value) >= LARGE:
        return f"{value:,.0f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.4f}" if abs(value) >= SMALL or value == 0 else f"{value:.2e}"
