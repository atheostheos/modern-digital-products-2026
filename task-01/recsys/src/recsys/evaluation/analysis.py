"""Business / fairness / cold-start / quality analyses of saved top-k recommendation lists.

Every function takes a model's recommendations and returns a flat dict of numbers, so the
same functions can describe what users actually did (ground truth passed as "recs").
"""

import polars as pl

from recsys.evaluation.evaluator import EvalWindow
from recsys.evaluation.metrics import top_k

PAIR = ["user_id", "item_id"]


def as_recs(interactions: pl.DataFrame) -> pl.DataFrame:
    """Ground-truth pairs in recommendation shape (all with equal score)."""
    return interactions.select(*PAIR).with_columns(score=pl.lit(1.0))


def business_proxies(
    recs: pl.DataFrame, truth: pl.DataFrame, items: pl.DataFrame, k: int
) -> dict[str, float]:
    """Price exposure and the share of (priced) ground-truth value captured by hits."""
    top = top_k(recs, k).join(items.select("item_id", "price"), on="item_id", how="left")
    hits = top.join(truth.select(PAIR), on=PAIR, how="semi")
    truth_value = truth.join(items.select("item_id", "price"), on="item_id")["price"].sum()
    return {
        "priced_slot_share": _share(top["price"].is_not_null()),
        "mean_rec_price": _mean(top["price"]),
        "captured_revenue_share": float(hits["price"].sum() / truth_value) if truth_value else 0.0,
    }


def brand_concentration(
    recs: pl.DataFrame, items: pl.DataFrame, k: int, top: int = 10
) -> dict[str, float]:
    """How much recommendation exposure the ``top`` most-recommended brands receive."""
    slots = top_k(recs, k).join(items.select("item_id", "brand"), on="item_id", how="left")
    per_brand = (
        slots.drop_nulls("brand")
        .group_by("brand")
        .len()
        .sort(["len", "brand"], descending=[True, False])
    )
    return {
        "distinct_brands": float(per_brand.height),
        "top_brand_slot_share": float(per_brand.head(top)["len"].sum() / slots.height)
        if slots.height
        else 0.0,
    }


def item_cold_start(
    recs: pl.DataFrame, truth: pl.DataFrame, history: pl.DataFrame, k: int
) -> dict[str, float]:
    """Exposure of items without history and how well a model catches demand for them."""
    seen = history["item_id"].unique().implode()
    top = top_k(recs, k)
    new_truth = truth.filter(~pl.col("item_id").is_in(seen))
    caught = new_truth.join(top.select(PAIR), on=PAIR, how="semi")
    return {
        "new_item_slot_share": _share(~top["item_id"].is_in(seen)),
        "truth_new_item_share": _share(~truth["item_id"].is_in(seen)),
        "new_item_hit_rate": float(caught.height / new_truth.height) if new_truth.height else 0.0,
    }


def quality_exposure(
    recs: pl.DataFrame, items: pl.DataFrame, k: int, max_rating: float, min_ratings: int
) -> dict[str, float]:
    """Share of slots on reliably low-rated items (avg < max_rating, >= min_ratings ratings)."""
    top = top_k(recs, k).join(
        items.select("item_id", "average_rating", "rating_number"), on="item_id", how="left"
    )
    low = (pl.col("average_rating") < max_rating) & (pl.col("rating_number") >= min_ratings)
    return {
        "low_rated_slot_share": _share(top.select(low.fill_null(False)).to_series()),
        "mean_rec_rating": _mean(top["average_rating"]),
    }


def _share(mask: pl.Series) -> float:
    return float(mask.mean()) if mask.len() else 0.0


def _mean(values: pl.Series) -> float:
    mean = values.drop_nulls().mean()
    return float(mean) if mean is not None else float("nan")


ACTUAL = "actual (window interactions)"
LOW_RATING = 3.5  # stars: below this an item counts as poorly rated
MIN_RATINGS = 10  # ... but only with enough ratings to be reliable
TOP_BRANDS = 10


def analyze_stage(window: EvalWindow, saved: dict[str, pl.DataFrame], k: int) -> pl.DataFrame:
    """One row per model (plus the ground-truth reference first) with all analysis numbers."""
    rows = [_row(ACTUAL, as_recs(window.ground_truth), window, k=None)]
    rows += [_row(name, recs, window, k=k) for name, recs in saved.items()]
    return pl.DataFrame(rows)


def _row(name: str, recs: pl.DataFrame, window: EvalWindow, k: int | None) -> dict:
    depth = k if k is not None else max(recs.group_by("user_id").len()["len"].max() or 1, 1)
    return {
        "model": name,
        **business_proxies(recs, window.ground_truth, window.items, depth),
        **brand_concentration(recs, window.items, depth, top=TOP_BRANDS),
        **item_cold_start(recs, window.ground_truth, window.history, depth),
        **quality_exposure(recs, window.items, depth, LOW_RATING, MIN_RATINGS),
    }
