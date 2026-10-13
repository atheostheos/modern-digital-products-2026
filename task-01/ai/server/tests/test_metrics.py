import pytest

from tdml.runtime.metrics import NodeMetrics, process_snapshot


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def test_empty_snapshot_has_zero_counters_and_no_latency(clock):
    snap = NodeMetrics("classify", clock=clock).snapshot()

    assert snap == {
        "node": "classify",
        "requests": 0,
        "processed": 0,
        "dropped": 0,
        "errors": 0,
        "p50_ms": None,
        "p95_ms": None,
        "fps": 0.0,
    }


def test_counts_requests_drops_and_errors(clock):
    metrics = NodeMetrics("gesture", clock=clock)

    for _ in range(3):
        metrics.record_request()
    metrics.record_drop()
    metrics.record_error()

    snap = metrics.snapshot()
    assert (snap["requests"], snap["dropped"], snap["errors"]) == (3, 1, 1)


def test_latency_percentiles_over_recent_inferences(clock):
    metrics = NodeMetrics("speech", clock=clock)

    for ms in range(1, 101):  # 1..100 ms
        metrics.record_inference(float(ms))

    snap = metrics.snapshot()
    assert snap["processed"] == 100
    assert snap["p50_ms"] == pytest.approx(50.5)
    assert snap["p95_ms"] == pytest.approx(95.05)


def test_latency_window_keeps_only_last_n_samples(clock):
    metrics = NodeMetrics("sentiment", window=10, clock=clock)

    for _ in range(50):
        metrics.record_inference(1000.0)
    for _ in range(10):
        metrics.record_inference(5.0)

    assert metrics.snapshot()["p95_ms"] == pytest.approx(5.0)


def test_fps_counts_inferences_in_last_seconds(clock):
    metrics = NodeMetrics("gesture", fps_window_s=2.0, clock=clock)

    for _ in range(10):
        metrics.record_inference(5.0)
    clock.now += 1.0
    for _ in range(10):
        metrics.record_inference(5.0)

    assert metrics.snapshot()["fps"] == pytest.approx(10.0)  # 20 inferences / 2 s
    clock.now += 1.5  # first batch now older than the window
    assert metrics.snapshot()["fps"] == pytest.approx(5.0)


def test_process_snapshot_reports_cpu_rss_and_threads():
    snap = process_snapshot()

    assert set(snap) == {"cpu_percent", "rss_mb", "threads"}
    assert snap["rss_mb"] > 0
    assert snap["threads"] >= 1


def test_fps_window_is_trimmed_on_write_even_without_snapshots(clock):
    metrics = NodeMetrics("gesture", fps_window_s=1.0, clock=clock)

    for _ in range(1000):
        metrics.record_inference(1.0)
        clock.now += 0.1

    assert len(metrics._done_at) <= 11  # only the last ~1 s is kept


def test_process_cpu_counter_is_primed_so_first_reading_is_meaningful():
    import tdml.runtime.metrics as metrics_module

    assert metrics_module._CPU_PRIMED is True
