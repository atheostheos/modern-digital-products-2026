"""Per-node inference metrics (latency percentiles, throughput, drops) and process stats."""

import threading
import time
from collections import deque
from collections.abc import Callable

import numpy as np
import psutil

DEFAULT_LATENCY_WINDOW = 200
DEFAULT_FPS_WINDOW_S = 5.0
BYTES_PER_MB = 1024 * 1024

_PROCESS = psutil.Process()
_PROCESS.cpu_percent(interval=None)  # psutil's first reading is always 0.0; prime it at import
_CPU_PRIMED = True


class NodeMetrics:
    """Thread-safe counters; recorded from worker threads, read from the event loop."""

    def __init__(
        self,
        node: str,
        window: int = DEFAULT_LATENCY_WINDOW,
        fps_window_s: float = DEFAULT_FPS_WINDOW_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.node = node
        self._fps_window_s = fps_window_s
        self._clock = clock
        self._lock = threading.Lock()
        self._latencies: deque[float] = deque(maxlen=window)
        self._done_at: deque[float] = deque()
        self._requests = 0
        self._processed = 0
        self._dropped = 0
        self._errors = 0

    def record_request(self) -> None:
        with self._lock:
            self._requests += 1

    def record_drop(self) -> None:
        with self._lock:
            self._dropped += 1

    def record_error(self) -> None:
        with self._lock:
            self._errors += 1

    def record_inference(self, inference_ms: float) -> None:
        with self._lock:
            now = self._clock()
            self._processed += 1
            self._latencies.append(inference_ms)
            self._done_at.append(now)
            self._trim(now)

    def _trim(self, now: float) -> None:
        cutoff = now - self._fps_window_s
        while self._done_at and self._done_at[0] < cutoff:
            self._done_at.popleft()

    def snapshot(self) -> dict:
        with self._lock:
            self._trim(self._clock())
            latencies = np.fromiter(self._latencies, dtype=float)
            return {
                "node": self.node,
                "requests": self._requests,
                "processed": self._processed,
                "dropped": self._dropped,
                "errors": self._errors,
                "p50_ms": _percentile(latencies, 50),
                "p95_ms": _percentile(latencies, 95),
                "fps": round(len(self._done_at) / self._fps_window_s, 3),
            }


def _percentile(values: np.ndarray, q: float) -> float | None:
    return round(float(np.percentile(values, q)), 3) if values.size else None


def process_snapshot() -> dict:
    return {
        "cpu_percent": _PROCESS.cpu_percent(interval=None),
        "rss_mb": round(_PROCESS.memory_info().rss / BYTES_PER_MB, 1),
        "threads": _PROCESS.num_threads(),
    }
