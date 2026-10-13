"""Latest-wins worker: one thread per node, a single pending slot, stale jobs are dropped.

Real-time clients (TD at 30-60 fps) send faster than slow models can infer. Queueing would grow
latency without bound; instead a newer job replaces the pending one and the drop is counted.
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from tdml.protocol import ProtocolError
from tdml.runtime.metrics import NodeMetrics

logger = logging.getLogger(__name__)

STOP_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class Job:
    seq: int
    payload: Any
    t_send: float | None
    t_recv: float
    on_done: Callable[["Result"], None]


@dataclass(frozen=True)
class Result:
    job: Job
    data: dict | None
    error: str | None
    inference_ms: float
    t_done: float


class LatestWinsWorker:
    def __init__(
        self,
        name: str,
        infer: Callable[[Any], dict],
        metrics: NodeMetrics,
        prepare: Callable[[Any], Any] | None = None,
    ) -> None:
        self.name = name
        self.metrics = metrics
        self._infer_fn = infer
        self._prepare = prepare  # e.g. JPEG decode; runs outside the model lock and timing
        self._infer_lock = threading.Lock()  # models are not assumed thread-safe (MediaPipe isn't)
        self._cond = threading.Condition()
        self._pending: Job | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self._cond:
            if self._running:
                return
            self._running = True
        self._thread = threading.Thread(target=self._loop, name=f"worker-{self.name}", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = STOP_TIMEOUT_S) -> None:
        with self._cond:
            self._running = False
            self._pending = None
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def submit(self, job: Job) -> bool:
        """Queue a job. Returns True when it replaced (dropped) an older pending job."""
        with self._cond:
            if not self._running:
                raise RuntimeError(f"worker {self.name} is not running")
            replaced = self._pending is not None
            self._pending = job
            self._cond.notify()
        self.metrics.record_request()
        if replaced:
            self.metrics.record_drop()
        return replaced

    def run_now(self, payload: Any) -> tuple[dict, float]:
        """Synchronous inference for request/response callers (REST). Shares the model lock."""
        self.metrics.record_request()
        return self._infer(payload)

    def _infer(self, payload: Any) -> tuple[dict, float]:
        if self._prepare is not None:
            try:
                payload = self._prepare(payload)
            except Exception:
                self.metrics.record_error()
                raise
        with self._infer_lock:
            t0 = time.perf_counter()
            try:
                data = self._infer_fn(payload)
            except Exception:
                self.metrics.record_error()
                raise
            inference_ms = (time.perf_counter() - t0) * 1000
        self.metrics.record_inference(inference_ms)
        return data, inference_ms

    def _next_job(self) -> Job | None:
        with self._cond:
            while self._running and self._pending is None:
                self._cond.wait()
            if not self._running:
                return None
            job, self._pending = self._pending, None
            return job

    def _loop(self) -> None:
        while (job := self._next_job()) is not None:
            try:
                data, inference_ms = self._infer(job.payload)
                result = Result(job, data, None, inference_ms, time.perf_counter())
            except ProtocolError as exc:  # bad client input: one line, no traceback per frame
                logger.warning("node %s: rejected seq %s: %s", self.name, job.seq, exc)
                result = Result(job, None, str(exc), 0.0, time.perf_counter())
            except Exception as exc:
                logger.exception("node %s: inference failed for seq %s", self.name, job.seq)
                result = Result(job, None, f"{type(exc).__name__}: {exc}", 0.0, time.perf_counter())
            try:
                job.on_done(result)
            except Exception:
                logger.exception("node %s: result callback failed for seq %s", self.name, job.seq)
