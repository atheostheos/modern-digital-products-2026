import threading

import pytest

from tdml.runtime.metrics import NodeMetrics
from tdml.runtime.worker import Job, LatestWinsWorker, Result
from tests.conftest import GatedNode, wait_until


def make_job(seq: int, payload: str, sink: list) -> Job:
    return Job(seq=seq, payload=payload, t_send=None, t_recv=0.0, on_done=sink.append)


@pytest.fixture
def results() -> list[Result]:
    return []


def test_processes_job_and_reports_result(results):
    metrics = NodeMetrics("echo")
    worker = LatestWinsWorker("echo", lambda text: {"upper": text.upper()}, metrics)
    worker.start()
    try:
        worker.submit(make_job(1, "hi", results))
        assert wait_until(lambda: len(results) == 1)
    finally:
        worker.stop()

    result = results[0]
    assert result.job.seq == 1
    assert result.data == {"upper": "HI"}
    assert result.error is None
    assert result.inference_ms >= 0
    assert metrics.snapshot()["processed"] == 1


def test_newer_job_replaces_pending_one_and_counts_drop(results):
    node = GatedNode()
    metrics = NodeMetrics("gated")
    worker = LatestWinsWorker("gated", node.infer, metrics)
    worker.start()
    try:
        worker.submit(make_job(1, "first", results))
        assert node.entered.wait(timeout=2)  # job 1 is now running and blocked

        assert worker.submit(make_job(2, "stale", results)) is False
        assert worker.submit(make_job(3, "latest", results)) is True  # replaced job 2

        node.release.set()
        assert wait_until(lambda: len(results) == 2)
    finally:
        worker.stop()

    assert node.seen == ["first", "latest"]
    assert [r.job.seq for r in results] == [1, 3]
    assert metrics.snapshot()["dropped"] == 1


def test_inference_exception_becomes_error_result(results):
    def boom(_: str) -> dict:
        raise RuntimeError("model exploded")

    metrics = NodeMetrics("boom")
    worker = LatestWinsWorker("boom", boom, metrics)
    worker.start()
    try:
        worker.submit(make_job(5, "x", results))
        assert wait_until(lambda: len(results) == 1)
    finally:
        worker.stop()

    assert results[0].data is None
    assert "model exploded" in results[0].error
    assert metrics.snapshot()["errors"] == 1


def test_callback_exception_does_not_kill_worker(results):
    calls = []

    def bad_callback(result: Result) -> None:
        calls.append(result)
        raise ValueError("client went away")

    worker = LatestWinsWorker("echo", lambda text: {"t": text}, NodeMetrics("echo"))
    worker.start()
    try:
        worker.submit(Job(seq=1, payload="a", t_send=None, t_recv=0.0, on_done=bad_callback))
        assert wait_until(lambda: len(calls) == 1)
        worker.submit(make_job(2, "b", results))
        assert wait_until(lambda: len(results) == 1)
    finally:
        worker.stop()


def test_stop_joins_thread_and_rejects_new_jobs(results):
    worker = LatestWinsWorker("echo", lambda text: {}, NodeMetrics("echo"))
    worker.start()
    worker.stop()

    assert not any(t.name == "worker-echo" for t in threading.enumerate())
    with pytest.raises(RuntimeError, match="not running"):
        worker.submit(make_job(1, "late", results))


def test_prepare_runs_before_infer_and_protocol_errors_become_error_results(results, caplog):
    from tdml.protocol import ProtocolError

    def prepare(raw: str) -> str:
        if raw == "bad":
            raise ProtocolError("not a valid image")
        return raw.strip()

    metrics = NodeMetrics("prep")
    worker = LatestWinsWorker("prep", lambda text: {"t": text}, metrics, prepare=prepare)
    worker.start()
    try:
        worker.submit(make_job(1, "  ok  ", results))
        assert wait_until(lambda: len(results) == 1)
        with caplog.at_level("WARNING"):
            worker.submit(make_job(2, "bad", results))
            assert wait_until(lambda: len(results) == 2)
    finally:
        worker.stop()

    assert results[0].data == {"t": "ok"}
    assert "not a valid image" in results[1].error
    assert metrics.snapshot()["errors"] == 1
    bad_records = [r for r in caplog.records if "seq 2" in r.getMessage()]
    assert bad_records and all(r.levelname == "WARNING" and r.exc_info is None for r in bad_records)
