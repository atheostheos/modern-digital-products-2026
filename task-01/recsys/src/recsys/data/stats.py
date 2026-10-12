"""Descriptive dataset statistics used in the report (sparsity, long tail, cold start)."""

import polars as pl

from recsys.data.split import Splits

HEAVY_USER_MIN = 5
HEAD_ITEMS_SHARE = 0.2
FEATURE_COLUMNS = ("brand", "category", "price", "bought_together")
HISTORY_THRESHOLDS = (1, 2, 5)


def interaction_stats(df: pl.DataFrame) -> dict[str, float | int | str]:
    """Size, density, user activity and item long-tail concentration."""
    users, items = df["user_id"].n_unique(), df["item_id"].n_unique()
    per_user = df.group_by("user_id").len()["len"]
    per_item = df.group_by("item_id").len()["len"].sort(descending=True)
    head = max(1, round(items * HEAD_ITEMS_SHARE))
    return {
        "interactions": df.height,
        "users": users,
        "items": items,
        "density": df.height / (users * items),
        "first_timestamp": str(df["timestamp"].min()),
        "last_timestamp": str(df["timestamp"].max()),
        "interactions_per_user_mean": float(per_user.mean()),
        "interactions_per_item_median": float(per_item.median()),
        "users_with_1_interaction_share": float((per_user == 1).mean()),
        f"users_with_{HEAVY_USER_MIN}plus_interactions_share": float(
            (per_user >= HEAVY_USER_MIN).mean()
        ),
        "top_20pct_items_interaction_share": float(per_item.head(head).sum() / df.height),
    }


def item_feature_coverage(items: pl.DataFrame) -> dict[str, float]:
    """Share of items with a non-null value per content feature."""
    return {c: float(items[c].is_not_null().mean()) for c in FEATURE_COLUMNS}


def history_coverage(splits: Splits) -> dict[str, dict[str, float | int]]:
    """Warm-user and unseen-item shares for val (history=train) and test (history=train+val)."""
    return {
        "val": _window_coverage(splits.val, splits.train),
        "test": _window_coverage(splits.test, pl.concat([splits.train, splits.val])),
    }


def _window_coverage(window: pl.DataFrame, history: pl.DataFrame) -> dict[str, float | int]:
    counts = (
        window.select("user_id")
        .unique()
        .join(history.group_by("user_id").len(), on="user_id", how="left")
        .fill_null(0)["len"]
    )
    seen_items = history["item_id"].unique().implode()
    stats: dict[str, float | int] = {
        "interactions": window.height,
        "users": counts.len(),
        "items": window["item_id"].n_unique(),
    }
    for k in HISTORY_THRESHOLDS:
        stats[f"users_with_history_{k}plus_share"] = float((counts >= k).mean())
    stats["interactions_on_unseen_items_share"] = float(
        (~window["item_id"].is_in(seen_items)).mean()
    )
    return stats
