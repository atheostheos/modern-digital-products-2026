"""Fan-out of server-produced events (stream nodes, e.g. speech) to every subscribed connection."""

import threading
import time
from typing import Protocol

from tdml.protocol import Envelope, Timing, failure, to_jsonable
from tdml.runtime.metrics import NodeMetrics

MS_PER_S = 1000


class Subscriber(Protocol):
    def post_threadsafe(self, text: str) -> None: ...


class Broadcaster:
    def __init__(self, node: str, metrics: NodeMetrics) -> None:
        self.node = node
        self.metrics = metrics
        self._lock = threading.Lock()
        self._subscribers: set[Subscriber] = set()
        self._seq = 0

    @property
    def subscribers(self) -> int:
        with self._lock:
            return len(self._subscribers)

    def subscribe(self, subscriber: Subscriber) -> None:
        with self._lock:
            self._subscribers.add(subscriber)

    def unsubscribe(self, subscriber: Subscriber) -> None:
        with self._lock:
            self._subscribers.discard(subscriber)

    def publish(self, data: dict, inference_ms: float | None = None) -> None:
        """Send one event to all subscribers. `inference_ms` is None for non-inference events."""
        now = time.perf_counter()
        with self._lock:
            self._seq += 1
            seq, targets = self._seq, list(self._subscribers)
        timing = None
        if inference_ms is not None:
            self.metrics.record_inference(inference_ms)
            timing = Timing(
                t_recv=now - inference_ms / MS_PER_S, t_done=now, inference_ms=round(inference_ms, 3)
            )
        try:
            text = Envelope(ok=True, node=self.node, seq=seq, data=to_jsonable(data), timing=timing).to_json()
        except (TypeError, ValueError) as exc:
            text = failure(self.node, seq, f"event not serialisable: {exc}").to_json()
        for subscriber in targets:
            subscriber.post_threadsafe(text)
