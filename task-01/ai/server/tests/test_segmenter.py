import numpy as np
import pytest

from tdml.audio.segmenter import UtteranceSegmenter

RATE = 16000
CHUNK = RATE // 4  # 0.25 s


def chunk(value: float = 0.0) -> np.ndarray:
    return np.full(CHUNK, value, dtype=np.float32)


def feed(segmenter: UtteranceSegmenter, pattern: str) -> list:
    """pattern: 'S' = speech chunk, '.' = silent chunk; returns all emitted events in order."""
    events = []
    for symbol in pattern:
        events.extend(segmenter.feed(chunk(1.0 if symbol == "S" else 0.0), is_speech=symbol == "S"))
    return events


@pytest.fixture
def segmenter() -> UtteranceSegmenter:
    return UtteranceSegmenter(sample_rate=RATE, silence_s=0.75, partial_every_s=1.5, max_utterance_s=10.0)


def test_silence_emits_nothing(segmenter):
    assert feed(segmenter, "........") == []


def test_short_utterance_emits_start_then_final_with_all_audio(segmenter):
    events = feed(segmenter, "SSSS...")

    assert [e.kind for e in events] == ["start", "final"]
    assert events[0].audio is None
    assert len(events[1].audio) == 7 * CHUNK  # speech plus trailing silence


def test_long_utterance_emits_partials_every_interval(segmenter):
    events = feed(segmenter, "SSSSSSSSSSSS...")  # 3 s of speech

    assert [e.kind for e in events] == ["start", "partial", "partial", "final"]
    assert len(events[1].audio) == 6 * CHUNK


def test_utterance_is_cut_at_max_length():
    segmenter = UtteranceSegmenter(sample_rate=RATE, partial_every_s=10.0, max_utterance_s=1.0)

    events = feed(segmenter, "SSSSSS")

    assert [e.kind for e in events] == ["start", "final", "start"]
    assert len(events[1].audio) == 4 * CHUNK


def test_blip_shorter_than_min_speech_ends_without_audio():
    segmenter = UtteranceSegmenter(sample_rate=RATE, silence_s=0.5, min_speech_s=0.3)

    events = feed(segmenter, "S..")

    assert [e.kind for e in events] == ["start", "final"]
    assert events[1].audio is None


def test_brief_pause_does_not_split_an_utterance(segmenter):
    events = feed(segmenter, "SS..SS...")
    kinds = [e.kind for e in events]

    assert kinds.count("start") == 1 and kinds.count("final") == 1
    assert kinds[-1] == "final"
