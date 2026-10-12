import json

from tdml.runtime.broadcast import Broadcaster
from tdml.runtime.metrics import NodeMetrics


class RecordingOutbox:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    def post_threadsafe(self, text: str) -> None:
        self.messages.append(json.loads(text))


def test_publish_reaches_every_subscriber_with_increasing_seq():
    metrics = NodeMetrics("speech")
    broadcaster = Broadcaster("speech", metrics)
    first, second = RecordingOutbox(), RecordingOutbox()
    broadcaster.subscribe(first)
    broadcaster.subscribe(second)

    broadcaster.publish({"type": "final", "text": "hello"}, inference_ms=120.0)
    broadcaster.publish({"type": "start"})  # not an inference: no timing, not counted

    assert first.messages == second.messages
    assert [m["seq"] for m in first.messages] == [1, 2]
    assert first.messages[0]["data"] == {"type": "final", "text": "hello"}
    assert first.messages[0]["timing"]["inference_ms"] == 120.0
    assert first.messages[1]["timing"] is None
    assert metrics.snapshot()["processed"] == 1


def test_unsubscribed_outbox_gets_nothing():
    broadcaster = Broadcaster("speech", NodeMetrics("speech"))
    outbox = RecordingOutbox()
    broadcaster.subscribe(outbox)
    broadcaster.unsubscribe(outbox)

    broadcaster.publish({"type": "start"})

    assert outbox.messages == []
    assert broadcaster.subscribers == 0


def test_unserialisable_event_becomes_error_envelope():
    broadcaster = Broadcaster("speech", NodeMetrics("speech"))
    outbox = RecordingOutbox()
    broadcaster.subscribe(outbox)

    broadcaster.publish({"bad": object()}, inference_ms=1.0)

    assert outbox.messages[0]["ok"] is False
    assert "serialis" in outbox.messages[0]["error"]
