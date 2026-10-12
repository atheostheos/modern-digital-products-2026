import threading

import numpy as np
import pytest

from tdml.nodes.base import InputKind
from tdml.nodes.speech import SpeechNode
from tests.conftest import wait_until

RATE = 16000
CHUNK = RATE // 4


class FakeSource:
    def __init__(self, on_chunk) -> None:
        self.on_chunk = on_chunk
        self.started = self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def push(self, pattern: str) -> None:
        for symbol in pattern:
            self.on_chunk(np.full(CHUNK, 1.0 if symbol == "S" else 0.0, dtype=np.float32))


class Harness:
    def __init__(self, transcribe=None) -> None:
        self.events: list[tuple[dict, float]] = []
        self.lock = threading.Lock()
        self.source: FakeSource | None = None
        self.node = SpeechNode(
            transcriber_factory=lambda: transcribe or (lambda audio: f"{len(audio) // CHUNK} chunks"),
            vad_factory=lambda: lambda chunk: bool(chunk.mean() > 0.5),
            source_factory=self._make_source,
        )

    def _make_source(self, on_chunk) -> FakeSource:
        self.source = FakeSource(on_chunk)
        return self.source

    def publish(self, data: dict, inference_ms: float) -> None:
        with self.lock:
            self.events.append((data, inference_ms))

    def types(self) -> list[str]:
        with self.lock:
            return [event["type"] for event, _ in self.events]


@pytest.fixture
def harness():
    h = Harness()
    h.node.load()
    h.node.start(h.publish)
    yield h
    h.node.stop()


def test_node_contract():
    assert SpeechNode.name == "speech"
    assert SpeechNode.input_kind is InputKind.STREAM


def test_start_opens_audio_source(harness):
    assert harness.source.started


def test_utterance_produces_start_and_transcribed_final(harness):
    harness.source.push("SSSS...")

    assert wait_until(lambda: harness.types() == ["start", "final"])
    (start, start_ms), (final, final_ms) = harness.events
    assert start_ms is None
    assert final == {"type": "final", "text": "7 chunks", "audio_s": 1.75}
    assert final_ms >= 0


def test_blip_final_has_empty_text_without_transcribing():
    calls = []
    h = Harness(transcribe=lambda audio: calls.append(audio) or "should not run")
    h.node.load()
    h.node.start(h.publish)
    try:
        h.source.push("S...")
        assert wait_until(lambda: h.types() == ["start", "final"])
    finally:
        h.node.stop()

    assert h.events[-1][0] == {"type": "final", "text": "", "audio_s": 0.0}
    assert calls == []


def test_transcriber_error_is_survived():
    attempts = []

    def flaky(audio: np.ndarray) -> str:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("decoder crashed")
        return "second try"

    h = Harness(transcribe=flaky)
    h.node.load()
    h.node.start(h.publish)
    try:
        h.source.push("SSSS...")
        h.source.push("SSSS...")
        assert wait_until(lambda: any(e.get("text") == "second try" for e, _ in h.events))
    finally:
        h.node.stop()

    assert {"type": "error", "error": "RuntimeError: decoder crashed"} in [e for e, _ in h.events]


def test_stop_stops_source_and_is_idempotent(harness):
    harness.node.stop()
    harness.node.stop()

    assert harness.source.stopped


def test_start_before_load_raises():
    with pytest.raises(RuntimeError, match="not loaded"):
        SpeechNode().start(lambda data, ms: None)


@pytest.mark.models
def test_real_vad_and_whisper_transcribe_a_spoken_sentence(tmp_path):
    import shutil
    import subprocess

    if shutil.which("say") is None:
        pytest.skip("macOS `say` is needed to synthesise the test sentence")
    from faster_whisper import decode_audio

    clip = tmp_path / "sentence.aiff"
    subprocess.run(["say", "-o", str(clip), "The quick brown fox jumps over the lazy dog."], check=True)
    audio = np.concatenate([decode_audio(str(clip), sampling_rate=RATE), np.zeros(RATE, dtype=np.float32)])

    events: list[dict] = []
    sources: list[FakeSource] = []
    node = SpeechNode(source_factory=lambda on_chunk: sources.append(FakeSource(on_chunk)) or sources[-1])
    node.load()
    node.start(lambda data, ms: events.append(data))
    try:
        for start in range(0, len(audio) - CHUNK + 1, CHUNK):
            sources[0].on_chunk(audio[start : start + CHUNK])
        assert wait_until(lambda: any(e["type"] == "final" for e in events), timeout=30)
    finally:
        node.stop()

    final = next(e for e in events if e["type"] == "final")
    assert "quick brown fox" in final["text"].lower()
