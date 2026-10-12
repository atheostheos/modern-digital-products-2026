import threading

import numpy as np
import pytest

from tdml.nodes.base import InputKind
from tdml.nodes.speech import SpeechNode
from tdml.protocol import AudioChunk
from tests.conftest import wait_until

RATE = 16000
CHUNK = RATE // 4


def audio(pattern: str, rate: int = RATE) -> list[np.ndarray]:
    """'S' = 0.25 s of speech (ones), '.' = 0.25 s of silence, at `rate`."""
    return [np.full(rate // 4, 1.0 if c == "S" else 0.0, dtype=np.float32) for c in pattern]


class Recorder:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.events: list[tuple[int, float | None, dict, float | None]] = []

    def __call__(self, seq: int, t_send: float | None, data: dict, inference_ms: float | None) -> None:
        with self.lock:
            self.events.append((seq, t_send, data, inference_ms))

    def types(self) -> list[str]:
        with self.lock:
            return [data["type"] for _, _, data, _ in self.events]

    def finals(self) -> list[dict]:
        with self.lock:
            return [data for _, _, data, _ in self.events if data["type"] == "final"]


def make_node(transcribe=None) -> SpeechNode:
    node = SpeechNode(
        transcriber_factory=lambda: transcribe or (lambda samples: f"{round(len(samples) / CHUNK)} chunks"),
        vad_factory=lambda: lambda chunk: bool(chunk.mean() > 0.5),
    )
    node.load()
    return node


def feed(session, pieces: list[np.ndarray], rate: int = RATE, first_seq: int = 1) -> None:
    for i, samples in enumerate(pieces):
        assert session.feed(AudioChunk(seq=first_seq + i, t_send=10.0 + i, sample_rate=rate, samples=samples))


@pytest.fixture
def recorder() -> Recorder:
    return Recorder()


def test_node_contract():
    assert SpeechNode.name == "speech"
    assert SpeechNode.input_kind is InputKind.AUDIO


def test_open_session_before_load_raises():
    with pytest.raises(RuntimeError, match="not loaded"):
        SpeechNode().open_session(lambda *args: None)


def test_utterance_emits_start_and_final_tagged_with_triggering_chunk(recorder):
    session = make_node().open_session(recorder)
    try:
        feed(session, audio("SSSS..."))
        assert wait_until(lambda: recorder.types() == ["start", "final"])
    finally:
        session.close()

    (start_seq, _, start, start_ms), (final_seq, final_t, final, final_ms) = recorder.events
    assert (start_seq, start, start_ms) == (1, {"type": "start"}, None)
    assert final_seq == 7 and final_t == 16.0
    assert final == {"type": "final", "text": "7 chunks", "audio_s": 1.75}
    assert final_ms >= 0


def test_chunks_of_any_size_are_rechunked(recorder):
    session = make_node().open_session(recorder)
    pieces = [p for chunk in audio("SSSS...") for p in np.split(chunk, 4)]  # 1000-sample pieces
    try:
        feed(session, pieces)
        assert wait_until(lambda: recorder.finals())
    finally:
        session.close()

    assert recorder.finals() == [{"type": "final", "text": "7 chunks", "audio_s": 1.75}]


def test_48khz_input_is_resampled_to_16khz(recorder):
    session = make_node().open_session(recorder)
    try:
        feed(session, audio("SSSS....", rate=48000), rate=48000)
        assert wait_until(lambda: recorder.finals())
    finally:
        session.close()

    assert recorder.finals()[0]["audio_s"] == pytest.approx(1.75, abs=0.01)


def test_transcription_error_is_reported_and_session_continues(recorder):
    attempts = []

    def flaky(samples: np.ndarray) -> str:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("decoder crashed")
        return "second try"

    session = make_node(transcribe=flaky).open_session(recorder)
    try:
        feed(session, audio("SSSS..."))
        feed(session, audio("SSSS..."), first_seq=8)
        assert wait_until(lambda: any(f["text"] == "second try" for f in recorder.finals()))
    finally:
        session.close()

    assert {"type": "error", "error": "RuntimeError: decoder crashed"} in [
        d for _, _, d, _ in recorder.events
    ]


def test_sessions_are_independent():
    node, first, second = make_node(), Recorder(), Recorder()
    session_a, session_b = node.open_session(first), node.open_session(second)
    try:
        feed(session_a, audio("SS"))  # unfinished utterance on A...
        feed(session_b, audio("SSSS..."))  # ...must not leak into B
        assert wait_until(lambda: second.finals())
    finally:
        session_a.close()
        session_b.close()

    assert second.finals() == [{"type": "final", "text": "7 chunks", "audio_s": 1.75}]
    assert "final" not in first.types()


def test_feed_after_close_is_refused(recorder):
    session = make_node().open_session(recorder)
    session.close()
    session.close()

    assert session.feed(AudioChunk(seq=1, t_send=None, sample_rate=RATE, samples=audio("S")[0])) is False


def test_full_queue_drops_chunks_instead_of_blocking(recorder):
    gate = threading.Event()
    node = SpeechNode(
        transcriber_factory=lambda: lambda samples: "x",
        vad_factory=lambda: lambda chunk: gate.wait(5) and False,
        max_queued_chunks=2,
    )
    node.load()
    session = node.open_session(recorder)
    try:
        results = [
            session.feed(AudioChunk(seq=i, t_send=None, sample_rate=RATE, samples=audio(".")[0]))
            for i in range(6)
        ]
    finally:
        gate.set()
        session.close()

    assert results[:2] == [True, True]
    assert False in results


@pytest.mark.models
def test_real_vad_and_whisper_transcribe_a_spoken_sentence_at_48khz(tmp_path):
    import shutil
    import subprocess

    if shutil.which("say") is None:
        pytest.skip("macOS `say` is needed to synthesise the test sentence")
    from faster_whisper import decode_audio

    clip = tmp_path / "sentence.aiff"
    subprocess.run(["say", "-o", str(clip), "The quick brown fox jumps over the lazy dog."], check=True)
    rate = 48000
    samples = np.concatenate([decode_audio(str(clip), sampling_rate=rate), np.zeros(rate, dtype=np.float32)])
    recorder = Recorder()
    node = SpeechNode()
    node.load()
    session = node.open_session(recorder)
    try:
        step = rate // 4
        feed(session, [samples[i : i + step] for i in range(0, len(samples), step)], rate=rate)
        assert wait_until(lambda: recorder.finals(), timeout=30)
    finally:
        session.close()

    assert "quick brown fox" in recorder.finals()[0]["text"].lower()
