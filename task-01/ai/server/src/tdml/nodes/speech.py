"""Speech-to-text stream node: microphone → Silero VAD → utterance segmenter → faster-whisper tiny.en.

Unlike request nodes, it produces events on its own and publishes them to subscribers:
{"type": "start"}, {"type": "partial" | "final", "text", "audio_s"}, {"type": "error", "error"}.
"""

import logging
import queue
import threading
import time
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np

from tdml.audio.segmenter import SegmentEvent, UtteranceSegmenter
from tdml.audio.source import SAMPLE_RATE, SoundDeviceSource
from tdml.config import models_dir
from tdml.nodes.base import InputKind, Publish

logger = logging.getLogger(__name__)

DEFAULT_MODEL_SIZE = "tiny.en"
DEFAULT_THREADS = 2
MAX_QUEUED_CHUNKS = 240  # 60 s of 0.25 s chunks; beyond that audio is dropped, not buffered forever
QUEUE_POLL_S = 0.1
STOP_TIMEOUT_S = 5.0
AUDIO_DECIMALS = 3
VAD_THRESHOLD = 0.5
VAD_MIN_SPEECH_MS = 100

Transcribe = Callable[[np.ndarray], str]
IsSpeech = Callable[[np.ndarray], bool]


def _whisper_transcriber(model_size: str, threads: int) -> Transcribe:
    from faster_whisper import WhisperModel

    model = WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
        cpu_threads=threads,
        download_root=str(models_dir() / "whisper"),
    )

    def transcribe(audio: np.ndarray) -> str:
        segments, _ = model.transcribe(
            audio, language="en", beam_size=1, vad_filter=False, condition_on_previous_text=False
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    return transcribe


def _silero_vad() -> IsSpeech:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(
        threshold=VAD_THRESHOLD,
        min_speech_duration_ms=VAD_MIN_SPEECH_MS,
        min_silence_duration_ms=100,
        speech_pad_ms=0,
    )
    return lambda chunk: bool(get_speech_timestamps(chunk, options))


class SpeechNode:
    name = "speech"
    input_kind = InputKind.STREAM

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL_SIZE,
        device: int | str | None = None,
        threads: int = DEFAULT_THREADS,
        transcriber_factory: Callable[[], Transcribe] | None = None,
        vad_factory: Callable[[], IsSpeech] | None = None,
        source_factory: Callable[[Callable[[np.ndarray], None]], Any] | None = None,
    ) -> None:
        self._transcriber_factory = transcriber_factory or partial(_whisper_transcriber, model_size, threads)
        self._vad_factory = vad_factory or _silero_vad
        self._source_factory = source_factory or partial(_source, device)
        self._transcribe: Transcribe | None = None
        self._is_speech: IsSpeech | None = None
        self._source: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._chunks: queue.Queue[np.ndarray] = queue.Queue(maxsize=MAX_QUEUED_CHUNKS)
        self.dropped_chunks = 0

    def load(self) -> None:
        self._transcribe = self._transcriber_factory()
        self._is_speech = self._vad_factory()

    def start(self, publish: Publish) -> None:
        if self._transcribe is None or self._is_speech is None:
            raise RuntimeError("speech model not loaded")
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, args=(publish,), name="speech", daemon=True)
        self._thread.start()
        self._source = self._source_factory(self._on_chunk)
        self._source.start()

    def stop(self) -> None:
        if self._source is not None:
            self._source.stop()
            self._source = None
        self._stop.set()
        if self._thread is not None:
            self._thread.join(STOP_TIMEOUT_S)
            self._thread = None

    def _on_chunk(self, chunk: np.ndarray) -> None:  # audio thread: never block
        try:
            self._chunks.put_nowait(chunk)
        except queue.Full:
            self.dropped_chunks += 1

    def _loop(self, publish: Publish) -> None:
        segmenter = UtteranceSegmenter(sample_rate=SAMPLE_RATE)
        while not self._stop.is_set():
            try:
                chunk = self._chunks.get(timeout=QUEUE_POLL_S)
            except queue.Empty:
                continue
            for event in segmenter.feed(chunk, self._is_speech(chunk)):
                self._handle(event, publish)

    def _handle(self, event: SegmentEvent, publish: Publish) -> None:
        if event.kind == "start":
            publish({"type": "start"}, None)
            return
        if event.audio is None:  # blip too short to transcribe
            publish({"type": event.kind, "text": "", "audio_s": 0.0}, None)
            return
        t0 = time.perf_counter()
        try:
            text = self._transcribe(event.audio)
        except Exception as exc:
            logger.exception("speech: transcription failed")
            publish({"type": "error", "error": f"{type(exc).__name__}: {exc}"}, None)
            return
        inference_ms = (time.perf_counter() - t0) * 1000
        audio_s = round(len(event.audio) / SAMPLE_RATE, AUDIO_DECIMALS)
        publish({"type": event.kind, "text": text, "audio_s": audio_s}, inference_ms)


def _source(device: int | str | None, on_chunk: Callable[[np.ndarray], None]) -> SoundDeviceSource:
    return SoundDeviceSource(on_chunk, device=device)
