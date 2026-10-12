"""Evaluate one model's recommendations overall and per user-history segment."""

from dataclasses import dataclass
from functools import reduce

import polars as pl

from recsys.config import EvaluationConfig
from recsys.evaluation.metrics import compute_metrics

ALL, WARM = "all", "warm"


@dataclass(frozen=True)
class EvalWindow:
    ground_truth: pl.DataFrame  # user_id, item_id inside the evaluation window
    history: pl.DataFrame  # user_id, item_id before the window (what models were fitted on)
    items: pl.DataFrame  # item_id, category


def segment_labels(bins: tuple[int, ...]) -> list[str]:
    """Labels for history-length bins given as lower bounds, e.g. (0, 1, 2, 5)."""
    uppers = [*(b - 1 for b in bins[1:]), None]
    return [
        f"hist={lo}+" if hi is None else (f"hist={lo}" if lo == hi else f"hist={lo}-{hi}")
        for lo, hi in zip(bins, uppers, strict=True)
    ]


def segment_users(
    ground_truth: pl.DataFrame, history: pl.DataFrame, bins: tuple[int, ...]
) -> pl.DataFrame:
    """One row per ground-truth user: history length and segment label."""
    labels = segment_labels(bins)
    length = pl.col("history_len")
    segment = reduce(
        lambda expr, pair: pl.when(length >= pair[0]).then(pl.lit(pair[1])).otherwise(expr),
        zip(bins[1:], labels[1:], strict=True),
        pl.lit(labels[0]),
    )
    return (
        ground_truth.select("user_id")
        .unique(maintain_order=True)
        .join(history.group_by("user_id").len("history_len"), on="user_id", how="left")
        .with_columns(length.fill_null(0).cast(pl.Int64))
        .with_columns(segment.alias("segment"))
    )


def evaluate(recs: pl.DataFrame, window: EvalWindow, cfg: EvaluationConfig) -> pl.DataFrame:
    """Wide table: one row per segment (all, warm, then history bins) with users + metrics."""
    segments = segment_users(window.ground_truth, window.history, cfg.history_bins)
    groups = [
        (ALL, segments),
        (WARM, segments.filter(pl.col("history_len") > 0)),
        *(
            (label, segments.filter(pl.col("segment") == label))
            for label in segment_labels(cfg.history_bins)
        ),
    ]
    rows = [
        _evaluate_users(name, users["user_id"], recs, window, cfg)
        for name, users in groups
        if not users.is_empty()
    ]
    return pl.DataFrame(rows)


def _evaluate_users(
    name: str, users: pl.Series, recs: pl.DataFrame, window: EvalWindow, cfg: EvaluationConfig
) -> dict[str, float | int | str]:
    keep = pl.col("user_id").is_in(users.implode())
    metrics = compute_metrics(
        recs.filter(keep),
        window.ground_truth.filter(keep),
        window.history,
        window.items,
        list(cfg.ks),
    )
    return {"segment": name, "users": users.len(), **metrics}
