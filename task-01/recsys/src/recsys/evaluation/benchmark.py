"""Serving-latency benchmark: single-request latency, batch throughput and precomputed lookup.

Batch timings from experiments hide the per-request overhead a live service pays
(e.g. a Spark job per call). Here each model is fitted once, then asked for one user
at a time (``requests`` random users) and for one batch (``batch_size`` users).
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from pyspark.sql import SparkSession

from recsys.config import Config
from recsys.evaluation.runner import StageData
from recsys.models.base import Recommender, TrainData
from recsys.models.registry import BuildContext, build_model
from recsys.spark import clear_caches

MS = 1000.0
MIN_REQUESTS = 3
TIME_BUDGET_SECONDS = 30.0  # per model, for the single-request loop


@dataclass(frozen=True)
class BenchSettings:
    requests: int  # single-user requests per model
    batch_size: int  # users in the one batch request
    seed: int


def run_benchmark(
    cfg: Config, data: StageData, names: list[str], spark: SparkSession, settings: BenchSettings
) -> list[dict[str, float | int | str]]:
    """Benchmark each named model on warm users of the stage (they have history to use)."""
    warm = data.train.interactions["user_id"].unique().implode()
    candidates = data.window.ground_truth["user_id"].unique().sort()
    users = candidates.filter(candidates.is_in(warm))
    ctx = BuildContext(spark=spark, seed=cfg.seed, threads=cfg.spark.cores, catalog=cfg.models)
    rows = []
    for name in names:
        model = build_model(name, cfg.models[name], ctx)
        rows.append(
            benchmark_model(
                model,
                data.train,
                users,
                k=max(cfg.evaluation.ks),
                requests=settings.requests,
                batch_size=settings.batch_size,
                seed=settings.seed,
                time_budget_seconds=TIME_BUDGET_SECONDS,
                after_call=lambda: clear_caches(spark),  # RePlay re-broadcasts per predict
            )
        )
    return rows


def precomputed_row(name: str, path: Path, settings: BenchSettings) -> dict[str, float | int | str]:
    """Latency of serving the precomputed lists stored in ``path``."""
    recs = pl.read_parquet(path)
    return lookup_latency(
        name, recs, recs["user_id"].unique().sort(), requests=settings.requests, seed=settings.seed
    )


def percentile_ms(seconds: Sequence[float], q: float) -> float:
    """``q``-th percentile of durations given in seconds, in milliseconds."""
    return float(np.percentile(np.asarray(seconds), q) * MS)


def benchmark_model(
    model: Recommender,
    train: TrainData,
    users: pl.Series,
    *,
    k: int,
    requests: int,
    batch_size: int,
    seed: int,
    time_budget_seconds: float | None = None,
    min_requests: int = MIN_REQUESTS,
    after_call: Callable[[], None] = lambda: None,
) -> dict[str, float | int | str]:
    """Fit ``model`` and time single-user requests and one batch request.

    Single requests stop once ``time_budget_seconds`` is spent (but at least
    ``min_requests`` run); ``after_call`` runs after every request (e.g. cache cleanup).
    """
    if users.is_empty():
        raise ValueError(f"{model.name}: no users to benchmark")
    users = users.unique().sort()  # deterministic pool -> seeded samples are reproducible
    started = time.perf_counter()
    model.fit(train)
    fit_seconds = time.perf_counter() - started
    model.recommend(users.head(1), k)  # untimed warm-up (lazy init, first Spark job)
    after_call()
    single_times = _single_requests(
        model,
        users.sample(n=min(requests, users.len()), seed=seed),
        k=k,
        budget=time_budget_seconds,
        min_requests=min_requests,
        after_call=after_call,
    )
    batch = users.sample(n=min(batch_size, users.len()), seed=seed + 1)
    batch_seconds = _timed(lambda: model.recommend(batch, k))
    after_call()
    return {
        "model": model.name,
        "fit_seconds": round(fit_seconds, 2),
        "requests": len(single_times),
        **_latency_stats(single_times),
        "batch_users": batch.len(),
        "batch_seconds": round(batch_seconds, 3),
        "batch_ms_per_user": batch_seconds * MS / batch.len(),
    }


def _single_requests(
    model: Recommender,
    users: pl.Series,
    *,
    k: int,
    budget: float | None,
    min_requests: int,
    after_call: Callable[[], None],
) -> list[float]:
    times: list[float] = []
    for user in users:
        if budget is not None and len(times) >= min_requests and sum(times) > budget:
            break
        times.append(_timed(lambda u=user: model.recommend(pl.Series([u]), k)))
        after_call()
    return times


def lookup_latency(
    name: str, recs: pl.DataFrame, users: pl.Series, *, requests: int, seed: int
) -> dict[str, float | int | str]:
    """Serving from precomputed lists: an in-memory key-value lookup per request."""
    table = {
        row["user_id"]: row["item_id"]
        for row in recs.group_by("user_id").agg(pl.col("item_id")).iter_rows(named=True)
    }
    singles = users.sample(n=min(requests, users.len()), seed=seed).to_list()
    single_times = [_timed(lambda u=user: table.get(u, [])) for user in singles]
    return {
        "model": name,
        "fit_seconds": 0.0,
        "requests": len(single_times),
        **_latency_stats(single_times),
    }


def _latency_stats(seconds: list[float]) -> dict[str, float]:
    return {
        "single_p50_ms": percentile_ms(seconds, 50),
        "single_p95_ms": percentile_ms(seconds, 95),
        "single_max_ms": max(seconds) * MS,
    }


def _timed(call: Callable[[], object]) -> float:
    started = time.perf_counter()
    call()
    return time.perf_counter() - started
