"""Offline metric suite: RePlay accuracy/beyond-accuracy metrics + popularity-bias measures.

Inputs (polars):
- recs: user_id, item_id, score (higher = better); top-k is taken per user by score
- ground_truth: user_id, item_id
- history: user_id, item_id interactions available before the evaluation window
- items: item_id, category
"""

import math
import warnings

import numpy as np
import polars as pl
from replay.metrics import (
    MAP,
    MRR,
    NDCG,
    CategoricalDiversity,
    Coverage,
    HitRate,
    Novelty,
    OfflineMetrics,
    Precision,
    Recall,
    Surprisal,
)

DEFAULT_HEAD_SHARE = 0.2  # "head" = most popular 20% of history items
UNKNOWN_CATEGORY = "__unknown__"
COLUMNS = {"query_column": "user_id", "item_column": "item_id", "rating_column": "score"}


ACCURACY_METRICS = (Precision, Recall, NDCG, MAP, MRR, HitRate)
BEYOND_ACCURACY_METRICS = (Coverage, Surprisal, CategoricalDiversity)  # + Novelty, see below
OWN_METRICS = ("Gini", "ARP", "LongTailShare")


def compute_metrics(
    recs: pl.DataFrame,
    ground_truth: pl.DataFrame,
    history: pl.DataFrame,
    items: pl.DataFrame,
    ks: list[int],
) -> dict[str, float]:
    """All metrics @k.

    Ground-truth users without recommendations count as misses for accuracy metrics.
    Exposure metrics (Coverage, Novelty, Gini, ARP, ...) describe only the users that
    received recommendations; with no recommendations at all they are NaN.
    """
    ranked = _rank(recs, max(ks))
    if ranked.is_empty():
        return _no_recommendation_metrics(ks)
    replay_metrics = [metric(ks) for metric in (*ACCURACY_METRICS, *BEYOND_ACCURACY_METRICS)]
    with warnings.catch_warnings():
        # intended semantics: ground-truth users without recommendations count as misses
        warnings.filterwarnings("ignore", message=".*not presented in recommendations")
        scored = OfflineMetrics(replay_metrics, category_column="category", **COLUMNS)(
            _with_category(ranked, items), ground_truth, history
        )
    scored.update(_novelty(ranked, history, ks))
    for k in ks:
        top = top_k(ranked, k)
        scored[f"Gini@{k}"] = gini_index(top, items["item_id"])
        scored[f"ARP@{k}"] = average_recommendation_popularity(top, history)
        scored[f"LongTailShare@{k}"] = long_tail_share(top, history)
    return scored


def top_k(recs: pl.DataFrame, k: int) -> pl.DataFrame:
    """The k best recommendations per user; score ties are broken by item_id."""
    return (
        recs.sort(["user_id", "score", "item_id"], descending=[False, True, False])
        .group_by("user_id", maintain_order=True)
        .head(k)
    )


def _rank(recs: pl.DataFrame, k: int) -> pl.DataFrame:
    """Top-k with strictly decreasing scores, so RePlay and our metrics see the same order."""
    position = pl.int_range(pl.len()).over("user_id")
    return top_k(recs, k).with_columns(score=(k - position).cast(pl.Float64))


def _novelty(ranked: pl.DataFrame, history: pl.DataFrame, ks: list[int]) -> dict[str, float]:
    """RePlay Novelty averages over every user in ``train``; restrict it to served users."""
    served = ranked["user_id"].unique().implode()
    own_history = history.filter(pl.col("user_id").is_in(served))
    if own_history.is_empty():  # nobody served has history: everything is novel
        return {f"Novelty@{k}": 1.0 for k in ks}
    return Novelty(ks, **COLUMNS)(ranked, own_history)


def _no_recommendation_metrics(ks: list[int]) -> dict[str, float]:
    accuracy = {f"{m.__name__}@{k}": 0.0 for m in ACCURACY_METRICS for k in ks}
    exposure_names = [m.__name__ for m in (*BEYOND_ACCURACY_METRICS, Novelty)] + list(OWN_METRICS)
    exposure = {f"{name}@{k}": math.nan for name in exposure_names for k in ks}
    return {**accuracy, **exposure}


def gini_index(recs: pl.DataFrame, catalog: pl.Series) -> float:
    """Gini of recommendation exposure over the catalog (0 = uniform, →1 = one item)."""
    counts = (
        catalog.to_frame("item_id")
        .join(recs.group_by("item_id").len(), on="item_id", how="left")
        .fill_null(0)["len"]
        .sort()
        .to_numpy()
        .astype(float)
    )
    n, total = counts.size, counts.sum()
    if n == 0 or total == 0:
        return 0.0
    ranks = np.arange(1, n + 1)
    return float(((2 * ranks - n - 1) * counts).sum() / (n * total))


def average_recommendation_popularity(recs: pl.DataFrame, history: pl.DataFrame) -> float:
    """ARP: mean over served users of the mean history popularity of their recommended items."""
    popularity = history.group_by("item_id").len().rename({"len": "popularity"})
    per_user = (
        recs.join(popularity, on="item_id", how="left")
        .fill_null(0)
        .group_by("user_id")
        .agg(pl.col("popularity").mean())
    )
    return float(per_user["popularity"].mean() or 0.0)


def long_tail_share(
    recs: pl.DataFrame, history: pl.DataFrame, head_share: float = DEFAULT_HEAD_SHARE
) -> float:
    """Share of recommendation slots filled with items outside the popularity head."""
    if recs.is_empty():
        return 0.0
    by_popularity = history.group_by("item_id").len().sort(["len", "item_id"], descending=True)
    head_size = max(1, round(by_popularity.height * head_share))
    head = by_popularity.head(head_size)["item_id"].implode()
    return float((~recs["item_id"].is_in(head)).mean())


def _with_category(recs: pl.DataFrame, items: pl.DataFrame) -> pl.DataFrame:
    return recs.join(items.select("item_id", "category"), on="item_id", how="left").with_columns(
        pl.col("category").fill_null(UNKNOWN_CATEGORY)
    )
