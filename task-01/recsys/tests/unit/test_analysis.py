import polars as pl
import pytest

from recsys.evaluation.analysis import (
    ACTUAL,
    analyze_stage,
    brand_concentration,
    business_proxies,
    item_cold_start,
    quality_exposure,
)
from recsys.evaluation.evaluator import EvalWindow
from recsys.evaluation.runner import write_table

ITEMS = pl.DataFrame(
    {
        "item_id": ["a", "b", "c", "d"],
        "brand": ["Fender", "Fender", "Yamaha", None],
        "price": [100.0, None, 50.0, 10.0],
        "average_rating": [4.8, 2.9, 3.0, 4.5],
        "rating_number": [500, 40, 5, 20],
    }
)
HISTORY = pl.DataFrame({"user_id": ["h", "h"], "item_id": ["a", "b"]})


def recs(rows: list[tuple[str, str]]) -> pl.DataFrame:
    n = len(rows)
    return pl.DataFrame(
        {
            "user_id": [u for u, _ in rows],
            "item_id": [i for _, i in rows],
            "score": [float(n - i) for i in range(n)],
        }
    )


def pairs(rows: list[tuple[str, str]]) -> pl.DataFrame:
    return pl.DataFrame({"user_id": [u for u, _ in rows], "item_id": [i for _, i in rows]})


def test_business_proxies_price_coverage_mean_price_and_captured_revenue():
    rec = recs([("u1", "a"), ("u1", "b"), ("u2", "c"), ("u2", "d")])
    truth = pairs([("u1", "a"), ("u2", "d"), ("u2", "b")])

    out = business_proxies(rec, truth, ITEMS, k=2)

    assert out["priced_slot_share"] == pytest.approx(3 / 4)
    assert out["mean_rec_price"] == pytest.approx((100 + 50 + 10) / 3)
    # priced ground truth: a (100) + d (10) = 110; both hit -> all of it captured
    assert out["captured_revenue_share"] == pytest.approx(1.0)


def test_brand_concentration_counts_slots_of_the_top_brands():
    rec = recs([("u1", "a"), ("u1", "b"), ("u2", "a"), ("u2", "c"), ("u3", "d")])

    out = brand_concentration(rec, ITEMS, k=2, top=1)

    assert out["distinct_brands"] == 2  # Fender, Yamaha (null brand ignored)
    assert out["top_brand_slot_share"] == pytest.approx(3 / 4)  # Fender a, b, a of 4 branded
    assert out["branded_slot_share"] == pytest.approx(4 / 5)  # d has no brand


def test_item_cold_start_shares_and_hits_on_new_items():
    rec = recs([("u1", "c"), ("u1", "a"), ("u2", "d")])
    truth = pairs([("u1", "c"), ("u2", "a"), ("u2", "c")])

    out = item_cold_start(rec, truth, HISTORY, k=2)

    assert out["new_item_slot_share"] == pytest.approx(2 / 3)  # c, d unseen in history
    assert out["truth_new_item_share"] == pytest.approx(2 / 3)  # c, c
    assert out["new_item_hit_rate"] == pytest.approx(1 / 2)  # u1 hit c; u2 missed c


def test_quality_exposure_counts_reliably_low_rated_items_only():
    rec = recs([("u1", "b"), ("u1", "c"), ("u2", "a"), ("u2", "b")])

    out = quality_exposure(rec, ITEMS, k=2, max_rating=3.5, min_ratings=10)

    # b is low-rated with 40 ratings; c is low-rated but only 5 ratings -> not counted
    assert out["low_rated_slot_share"] == pytest.approx(2 / 4)
    assert out["mean_rec_rating"] == pytest.approx((2.9 + 3.0 + 4.8 + 2.9) / 4)


def test_analyses_respect_k_truncation():
    rec = recs([("u1", "a"), ("u1", "b"), ("u1", "c")])

    out = brand_concentration(rec, ITEMS, k=1, top=1)

    assert out["top_brand_slot_share"] == 1.0  # only a@1 counted


def test_analyze_stage_adds_a_ground_truth_reference_row():
    window = EvalWindow(
        ground_truth=pairs([("u1", "a"), ("u2", "c")]), history=HISTORY, items=ITEMS
    )
    saved = {"m1": recs([("u1", "a"), ("u2", "d")]), "m2": recs([("u1", "c")])}

    table = analyze_stage(window, saved, k=2)

    assert table["model"].to_list() == [ACTUAL, "m1", "m2"]
    assert {
        "captured_revenue_share",
        "top_brand_slot_share",
        "new_item_slot_share",
        "low_rated_slot_share",
    } <= set(table.columns)
    m1 = table.filter(pl.col("model") == "m1").row(0, named=True)
    assert m1["captured_revenue_share"] == pytest.approx(100 / 150)  # hit a (100) of a+c (150)


def test_write_table_writes_parquet_csv_and_markdown(tmp_path):
    table = pl.DataFrame({"model": ["m"], "value": [0.123456]})

    paths = write_table(table, tmp_path / "analysis" / "test", title="Test analysis")

    assert pl.read_parquet(paths.parquet).equals(table)
    assert paths.csv.is_file()
    text = paths.markdown.read_text()
    assert text.startswith("# Test analysis") and "| m | 0.1235 |" in text


def test_brand_share_is_measured_over_branded_slots_and_branded_share_reported():
    rec = recs([("u1", "a"), ("u1", "d"), ("u2", "c"), ("u2", "d")])  # d has no brand

    out = brand_concentration(rec, ITEMS, k=2, top=1)

    assert out["branded_slot_share"] == pytest.approx(2 / 4)
    assert out["top_brand_slot_share"] == pytest.approx(1 / 2)  # Fender: 1 of 2 branded slots


def test_captured_revenue_ignores_duplicate_truth_rows():
    rec = recs([("u1", "a")])
    truth = pairs([("u1", "a"), ("u1", "a"), ("u2", "c")])

    out = business_proxies(rec, truth, ITEMS, k=1)

    assert out["captured_revenue_share"] == pytest.approx(100 / 150)


def test_quality_exposure_reports_rated_slot_share():
    rec = recs([("u1", "a"), ("u1", "z")])  # z unknown -> no rating

    out = quality_exposure(rec, ITEMS, k=2, max_rating=3.5, min_ratings=10)

    assert out["rated_slot_share"] == pytest.approx(1 / 2)


def test_markdown_renders_missing_values_as_dash_and_keeps_small_values(tmp_path):
    table = pl.DataFrame(
        {"model": ["m"], "nan": [float("nan")], "none": [None], "tiny": [0.000912]},
        schema_overrides={"none": pl.Float64},
    )

    text = write_table(table, tmp_path / "t", title="T").markdown.read_text()

    assert "| m | - | - | 9.12e-04 |" in text


def test_markdown_formats_large_and_regular_numbers_readably(tmp_path):
    table = pl.DataFrame({"model": ["m"], "big": [17380.4], "mid": [9.2941], "small": [0.0245]})

    text = write_table(table, tmp_path / "t", title="T").markdown.read_text()

    assert "| m | 17,380 | 9.29 | 0.0245 |" in text
