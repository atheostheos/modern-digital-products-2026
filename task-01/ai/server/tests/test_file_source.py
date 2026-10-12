import numpy as np
import pytest

from tdml.audio.file_source import FileAudioSource
from tests.conftest import wait_until

RATE = 16000
CHUNK = RATE // 4


def make_source(audio: np.ndarray, sink: list, **kwargs) -> FileAudioSource:
    return FileAudioSource(sink.append, path="clip.wav", loader=lambda path: audio, realtime=False, **kwargs)


def test_emits_whole_file_as_fixed_chunks_with_zero_padded_tail():
    chunks: list[np.ndarray] = []
    audio = np.arange(CHUNK * 2 + 100, dtype=np.float32)
    source = make_source(audio, chunks, loop=False, gap_s=0.0)

    source.start()
    assert wait_until(lambda: source.finished)
    source.stop()

    assert len(chunks) == 3
    assert all(len(c) == CHUNK for c in chunks)
    assert np.array_equal(np.concatenate(chunks)[: len(audio)], audio)
    assert not chunks[-1][100:].any()


def test_loops_with_silent_gap_between_repeats():
    chunks: list[np.ndarray] = []
    audio = np.ones(CHUNK, dtype=np.float32)
    source = make_source(audio, chunks, loop=True, gap_s=0.5)

    source.start()
    assert wait_until(lambda: len(chunks) >= 7)
    source.stop()

    pattern = "".join("S" if c.any() else "." for c in chunks[:7])
    assert pattern == "S..S..S"


def test_stop_is_idempotent_and_ends_the_thread():
    chunks: list[np.ndarray] = []
    source = make_source(np.ones(CHUNK, dtype=np.float32), chunks, loop=True, gap_s=0.0)
    source.start()

    source.stop()
    source.stop()
    count = len(chunks)

    assert wait_until(lambda: len(chunks) == count, timeout=0.2)


def test_missing_file_fails_at_start(tmp_path):
    source = FileAudioSource(lambda chunk: None, path=tmp_path / "missing.wav")

    with pytest.raises(FileNotFoundError):
        source.start()


def test_realtime_pacing_uses_chunk_duration():
    sleeps: list[float] = []
    clock = iter([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    chunks: list[np.ndarray] = []
    source = FileAudioSource(
        chunks.append,
        path="clip.wav",
        loader=lambda path: np.ones(CHUNK * 2, dtype=np.float32),
        loop=False,
        gap_s=0.0,
        sleep=sleeps.append,
        clock=lambda: next(clock, 0.0),
    )

    source.start()
    assert wait_until(lambda: source.finished)

    assert sleeps[:2] == pytest.approx([0.25, 0.5])
