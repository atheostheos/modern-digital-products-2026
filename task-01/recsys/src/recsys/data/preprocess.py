"""Clean ingested reviews/meta into model-ready interactions and an item feature table."""

import polars as pl

ROOT_CATEGORY_DEPTH = 1  # categories[0] is the dataset-wide root ("Musical Instruments")


def build_interactions(reviews: pl.DataFrame) -> pl.DataFrame:
    """One row per (user, item): the latest review, with a datetime ``timestamp``."""
    return (
        reviews.rename({"parent_asin": "item_id"})
        .with_columns(pl.from_epoch("timestamp", time_unit="ms"))
        .sort("timestamp", maintain_order=True)
        .unique(subset=["user_id", "item_id"], keep="last", maintain_order=True)
        .select("user_id", "item_id", "timestamp", "rating", "verified_purchase")
    )


def build_items(meta: pl.DataFrame) -> pl.DataFrame:
    """Item table: brand, category levels, concatenated text and numeric attributes."""
    below_root = pl.col("categories").list.slice(ROOT_CATEGORY_DEPTH)
    has_levels = below_root.list.len() > 0
    brand = pl.col("store").str.strip_chars()
    text_parts = pl.concat_list(
        pl.concat_list(pl.col("title").fill_null("")),
        pl.col("features").fill_null([]),
        pl.col("description").fill_null([]),
        below_root.fill_null([]),
    )
    text = text_parts.list.eval(pl.element().filter(pl.element() != "")).list.join(" ")
    bought_together = pl.col("bought_together")
    return meta.unique(subset=["parent_asin"], keep="first", maintain_order=True).select(
        pl.col("parent_asin").alias("item_id"),
        pl.col("title"),
        pl.when(brand != "").then(brand).alias("brand"),
        pl.when(has_levels).then(below_root.list.last()).alias("category"),
        pl.when(has_levels).then(below_root.list.first()).alias("top_category"),
        pl.when(text != "").then(text).alias("text"),
        "price",
        "average_rating",
        "rating_number",
        pl.when(bought_together.list.len() > 0).then(bought_together).alias("bought_together"),
    )
