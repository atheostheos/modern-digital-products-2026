"""Recorded-audio input for the speech node: plays a file as if it were the microphone.

Decodes once to 16 kHz mono, cuts it into the same 0.25 s chunks as the mic, and emits them in real
time (optionally looping with a silent gap), so VAD, segmentation and whisper behave as in live use.
"""

import threading
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np

from tdml.audio.source import CHUNK_S, SAMPLE_RATE

DEFAULT_GAP_S = 2.0
STOP_TIMEOUT_S = 2.0

Loader = Callable[[str | Path], np.ndarray]


def decode_file(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"audio file not found: {path}")
    from faster_whisper import decode_audio

    return decode_audio(str(path), sampling_rate=sample_rate)


class FileAudioSource:
    def __init__(
        self,
        on_chunk: Callable[[np.ndarray], None],
        path: str | Path,
        sample_rate: int = SAMPLE_RATE,
        chunk_s: float = CHUNK_S,
        loop: bool = True,
        gap_s: float = DEFAULT_GAP_S,
        realtime: bool = True,
        loader: Loader | None = None,
        sleep: Callable[[float], object] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._on_chunk = on_chunk
        self._path = path
        self._chunk = round(sample_rate * chunk_s)
        self._chunk_s = chunk_s
        self._loop = loop
        self._gap_chunks = round(gap_s / chunk_s)
        self._realtime = realtime
        self._loader = loader or partial(decode_file, sample_rate=sample_rate)
        self._stop = threading.Event()
        self._sleep = sleep or self._stop.wait
        self._clock = clock
        self._done = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def finished(self) -> bool:
        return self._done.is_set()

    def _chunks(self, audio: np.ndarray) -> list[np.ndarray]:
        padded = np.zeros(-(-len(audio) // self._chunk) * self._chunk, dtype=np.float32)
        padded[: len(audio)] = audio
        return list(padded.reshape(-1, self._chunk))

    def start(self) -> None:
        chunks = self._chunks(self._loader(self._path))  # decode errors surface here, at startup
        gap = [np.zeros(self._chunk, dtype=np.float32)] * self._gap_chunks
        self._stop.clear()
        self._done.clear()
        self._thread = threading.Thread(target=self._run, args=(chunks, gap), name="file-audio", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(STOP_TIMEOUT_S)
            self._thread = None

    def _run(self, chunks: list[np.ndarray], gap: list[np.ndarray]) -> None:
        next_t = self._clock()
        while not self._stop.is_set():
            for chunk in chunks + (gap if self._loop else []):
                if self._stop.is_set():
                    return
                self._on_chunk(chunk.copy())
                next_t = self._pace(next_t)
            if not self._loop:
                self._done.set()
                return

    def _pace(self, next_t: float) -> float:
        if not self._realtime:
            return next_t
        next_t += self._chunk_s
        delay = next_t - self._clock()
        if delay > 0:
            self._sleep(delay)
        return next_t
