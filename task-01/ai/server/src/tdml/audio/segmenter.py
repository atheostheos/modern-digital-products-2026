"""Turns a stream of (audio chunk, is_speech) into utterance events: start, partial, final.

Pure logic (no audio I/O, no models) so it is unit-testable. Durations are tracked in samples.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_SILENCE_S = 0.75  # trailing silence that ends an utterance
DEFAULT_PARTIAL_EVERY_S = 1.5  # running transcript cadence while speaking
DEFAULT_MAX_UTTERANCE_S = 10.0  # hard cut so latency and whisper cost stay bounded
DEFAULT_MIN_SPEECH_S = 0.3  # shorter blips are not transcribed (whisper hallucinates on them)

EventKind = Literal["start", "partial", "final"]


@dataclass(frozen=True)
class SegmentEvent:
    kind: EventKind
    audio: np.ndarray | None  # None for "start" and for discarded blips


class UtteranceSegmenter:
    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        silence_s: float = DEFAULT_SILENCE_S,
        partial_every_s: float = DEFAULT_PARTIAL_EVERY_S,
        max_utterance_s: float = DEFAULT_MAX_UTTERANCE_S,
        min_speech_s: float = DEFAULT_MIN_SPEECH_S,
    ) -> None:
        self._silence_limit = round(silence_s * sample_rate)
        self._partial_every = round(partial_every_s * sample_rate)
        self._max_samples = round(max_utterance_s * sample_rate)
        self._min_speech = round(min_speech_s * sample_rate)
        self._reset()

    def _reset(self) -> None:
        self._chunks: list[np.ndarray] = []
        self._samples = 0
        self._speech = 0
        self._silence = 0
        self._since_partial = 0

    @property
    def active(self) -> bool:
        return bool(self._chunks)

    def _audio(self) -> np.ndarray:
        return np.concatenate(self._chunks)

    def _finish(self) -> SegmentEvent:
        audio = self._audio() if self._speech >= self._min_speech else None
        self._reset()
        return SegmentEvent("final", audio)

    def feed(self, chunk: np.ndarray, is_speech: bool) -> list[SegmentEvent]:
        if not self.active and not is_speech:
            return []
        events: list[SegmentEvent] = [] if self.active else [SegmentEvent("start", None)]
        self._chunks.append(chunk)
        self._samples += len(chunk)
        self._since_partial += len(chunk)
        if is_speech:
            self._speech += len(chunk)
            self._silence = 0
            if self._since_partial >= self._partial_every and self._samples < self._max_samples:
                events.append(SegmentEvent("partial", self._audio()))
                self._since_partial = 0
        else:
            self._silence += len(chunk)
        if self._silence >= self._silence_limit or self._samples >= self._max_samples:
            events.append(self._finish())
        return events
