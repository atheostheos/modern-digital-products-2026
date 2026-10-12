import time
from typing import Self

import polars as pl
import pytest

from recsys.evaluation.benchmark import benchmark_model, lookup_latency, percentile_ms
from recsys.models.base import TrainData

TRAIN = TrainData(interactions=pl.DataFrame(), items=pl.DataFrame())
USERS = pl.Series([f"u{i}" for i in range(50)])


class SlowStub:
    """Takes ~1 ms per call plus 0.01 ms per requested user."""

    name = "stub"

    def __init__(self) -> None:
        self.calls: list[int] = []

    def fit(self, data: TrainData) -> Self:
        return self

    def recommend(self, users: pl.Series, k: int) -> pl.DataFrame:
        self.calls.append(users.len())
        time.sleep(0.001 + 0.00001 * users.len())
        return pl.DataFrame(
            {"user_id": users, "item_id": ["x"] * users.len(), "score": [1.0] * users.len()}
        )


def test_percentiles_in_milliseconds():
    assert percentile_ms([0.001, 0.002, 0.003, 0.004], 50) == pytest.approx(2.5)
    assert percentile_ms([0.010] * 9 + [0.100], 95) == pytest.approx(59.5)


def test_benchmark_times_single_requests_and_a_batch():
    stub = SlowStub()

    result = benchmark_model(stub, TRAIN, USERS, k=10, requests=5, batch_size=20, seed=0)

    assert stub.calls == [1, 1, 1, 1, 1, 1, 20]  # warm-up + 5 singles + batch
    assert result["model"] == "stub"
    assert result["requests"] == 5 and result["batch_users"] == 20
    assert 1.0 <= result["single_p50_ms"] <= result["single_p95_ms"]
    assert result["batch_ms_per_user"] < result["single_p50_ms"]  # batching amortizes overhead
    assert result["fit_seconds"] >= 0


def test_benchmark_caps_requests_and_batch_at_the_user_pool():
    result = benchmark_model(
        SlowStub(), TRAIN, USERS.head(3), k=10, requests=10, batch_size=100, seed=0
    )

    assert result["requests"] == 3 and result["batch_users"] == 3


def test_lookup_latency_serves_precomputed_lists():
    recs = pl.DataFrame(
        {"user_id": ["u0", "u0", "u1"], "item_id": ["a", "b", "c"], "score": [2.0, 1.0, 1.0]}
    )

    result = lookup_latency(
        "hybrid (precomputed)", recs, pl.Series(["u0", "u1"]), requests=2, seed=0
    )

    assert result["model"] == "hybrid (precomputed)"
    assert result["requests"] == 2
    assert result["single_p95_ms"] < 5  # an in-memory key-value lookup


class SleepyStub(SlowStub):
    """Every call takes ~30 ms."""

    def recommend(self, users: pl.Series, k: int) -> pl.DataFrame:
        self.calls.append(users.len())
        time.sleep(0.03)
        return pl.DataFrame(
            {"user_id": users, "item_id": ["x"] * users.len(), "score": [1.0] * users.len()}
        )


def test_time_budget_stops_single_requests_early_but_keeps_a_minimum():
    stub = SleepyStub()

    result = benchmark_model(
        stub,
        TRAIN,
        USERS,
        k=10,
        requests=20,
        batch_size=5,
        seed=0,
        time_budget_seconds=0.05,
        min_requests=3,
    )

    assert result["requests"] == 3  # budget exceeded after 2 calls, minimum of 3 kept
    assert stub.calls == [1, 1, 1, 1, 5]  # warm-up + 3 singles + batch


def test_after_call_hook_runs_after_every_request():
    calls: list[str] = []

    benchmark_model(
        SlowStub(),
        TRAIN,
        USERS,
        k=10,
        requests=2,
        batch_size=3,
        seed=0,
        after_call=lambda: calls.append("cleared"),
    )

    assert calls == ["cleared"] * 4  # warm-up + 2 single requests + 1 batch


def test_benchmark_picks_the_same_users_for_the_same_seed():
    a, b = SlowStub(), SlowStub()
    asked_a: list[str] = []
    asked_b: list[str] = []
    a.recommend = lambda users, k, s=a: (
        asked_a.extend(users.to_list()) or SlowStub.recommend(s, users, k)
    )
    b.recommend = lambda users, k, s=b: (
        asked_b.extend(users.to_list()) or SlowStub.recommend(s, users, k)
    )
    pool = pl.Series([f"u{i}" for i in range(40)])

    benchmark_model(a, TRAIN, pool, k=5, requests=5, batch_size=3, seed=7)
    benchmark_model(b, TRAIN, pool.reverse(), k=5, requests=5, batch_size=3, seed=7)

    assert asked_a == asked_b  # independent of the pool's order


def test_benchmark_warms_up_with_one_untimed_request():
    stub = SlowStub()

    result = benchmark_model(stub, TRAIN, USERS, k=10, requests=2, batch_size=3, seed=0)

    assert stub.calls == [1, 1, 1, 3]  # warm-up + 2 timed singles + batch
    assert result["requests"] == 2


def test_benchmark_rejects_an_empty_user_pool():
    with pytest.raises(ValueError, match="no users"):
        benchmark_model(
            SlowStub(), TRAIN, pl.Series([], dtype=pl.String), k=1, requests=1, batch_size=1, seed=0
        )
