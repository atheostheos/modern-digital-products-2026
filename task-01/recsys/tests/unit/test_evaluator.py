import polars as pl
import pytest

from recsys.config import EvaluationConfig
from recsys.evaluation.evaluator import EvalWindow, evaluate, segment_labels, segment_users

CFG = EvaluationConfig(ks=(1,), history_bins=(0, 1, 2))


def pairs(rows: list[tuple[str, str]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=["user_id", "item_id"], orient="row")


HISTORY = pairs([("warm1", "a"), ("heavy", "a"), ("heavy", "b"), ("heavy", "c")])
TRUTH = pairs([("cold", "a"), ("warm1", "b"), ("heavy", "d")])
ITEMS = pl.DataFrame({"item_id": ["a", "b", "c", "d"], "category": ["x", "x", "y", "y"]})
WINDOW = EvalWindow(ground_truth=TRUTH, history=HISTORY, items=ITEMS)


def test_segment_labels_follow_bin_lower_bounds():
    assert segment_labels((0, 1, 2, 5)) == ["hist=0", "hist=1", "hist=2-4", "hist=5+"]


def test_segment_users_assigns_each_ground_truth_user_once():
    seg = segment_users(TRUTH, HISTORY, CFG.history_bins).sort("user_id")

    assert seg.to_dicts() == [
        {"user_id": "cold", "history_len": 0, "segment": "hist=0"},
        {"user_id": "heavy", "history_len": 3, "segment": "hist=2+"},
        {"user_id": "warm1", "history_len": 1, "segment": "hist=1"},
    ]


def test_evaluate_reports_overall_warm_and_per_segment_rows():
    recs = pl.DataFrame(
        {
            "user_id": ["cold", "warm1", "heavy"],
            "item_id": ["a", "c", "d"],
            "score": [1.0] * 3,
        }
    )

    table = evaluate(recs, WINDOW, CFG)

    by_segment = {r["segment"]: r for r in table.to_dicts()}
    assert list(by_segment) == ["all", "warm", "hist=0", "hist=1", "hist=2+"]
    assert by_segment["all"]["users"] == 3
    assert by_segment["all"]["HitRate@1"] == pytest.approx(2 / 3)
    assert by_segment["warm"]["users"] == 2
    assert by_segment["warm"]["HitRate@1"] == pytest.approx(1 / 2)
    assert by_segment["hist=0"]["HitRate@1"] == 1.0
    assert by_segment["hist=1"]["HitRate@1"] == 0.0


def test_evaluate_skips_empty_segments():
    window = EvalWindow(ground_truth=pairs([("heavy", "d")]), history=HISTORY, items=ITEMS)
    recs = pl.DataFrame({"user_id": ["heavy"], "item_id": ["d"], "score": [1.0]})

    table = evaluate(recs, window, CFG)

    assert table["segment"].to_list() == ["all", "warm", "hist=2+"]


def test_evaluate_scores_unserved_segment_as_misses_instead_of_crashing():
    recs = pl.DataFrame({"user_id": ["heavy"], "item_id": ["d"], "score": [1.0]})  # cold unserved

    table = evaluate(recs, WINDOW, CFG)

    by_segment = {r["segment"]: r for r in table.to_dicts()}
    assert by_segment["hist=0"]["users"] == 1
    assert by_segment["hist=0"]["HitRate@1"] == 0.0
    assert by_segment["all"]["HitRate@1"] == pytest.approx(1 / 3)


def test_evaluate_computes_exposure_metrics_on_segment_recs_only():
    recs = pl.DataFrame(
        {"user_id": ["cold", "warm1", "heavy"], "item_id": ["a", "c", "d"], "score": [1.0] * 3}
    )

    table = evaluate(recs, WINDOW, CFG)

    by_segment = {r["segment"]: r for r in table.to_dicts()}
    assert by_segment["hist=0"]["ARP@1"] == 2.0  # only cold's rec (a: 2 history interactions)
    assert by_segment["all"]["ARP@1"] == pytest.approx((2 + 1 + 0) / 3)
