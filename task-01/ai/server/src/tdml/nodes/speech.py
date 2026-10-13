"""Speech-to-text node: audio chunks streamed by the client -> 16 kHz -> Silero VAD -> utterance
segmenter -> faster-whisper tiny.en. The server never opens a microphone: TouchDesigner (or the harness)
captures live or recorded audio and sends it.

Each connection gets its own SpeechSession (resampler, re-chunker, segmenter, worker thread); the whisper
model is shared behind a lock. Events go back to that connection, tagged with the seq/t_send of the chunk
that completed them: {"type": "start"}, {"type": "partial" | "final", "text", "audio_s"},
{"type": "error", "error"}.
"""

import logging
import queue
import threading
import time
from collections.abc import Callable
from functools import partial

import numpy as np

from tdml.audio.resample import TARGET_RATE, StreamResampler
from tdml.audio.segmenter import SegmentEvent, UtteranceSegmenter
from tdml.config import models_dir
from tdml.nodes.base import Emit, InputKind
from tdml.protocol import AudioChunk

logger = logging.getLogger(__name__)

DEFAULT_MODEL_SIZE = "tiny.en"
DEFAULT_THREADS = 2
CHUNK_S = 0.25
CHUNK_SAMPLES = round(TARGET_RATE * CHUNK_S)
MAX_QUEUED_CHUNKS = 40  # ~10 s of 0.25 s client chunks per session; beyond that new chunks are dropped
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


class SpeechSession:
    def __init__(self, transcribe: Transcribe, is_speech: IsSpeech, emit: Emit, max_queued: int) -> None:
        self._transcribe = transcribe
        self._is_speech = is_speech
        self._emit = emit
        self._chunks: queue.Queue[AudioChunk] = queue.Queue(maxsize=max_queued)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="speech-session", daemon=True)
        self._thread.start()

    def feed(self, chunk: AudioChunk) -> bool:
        if self._stop.is_set() or not self._thread.is_alive():
            return False
        try:
            self._chunks.put_nowait(chunk)
            return True
        except queue.Full:
            return False

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(STOP_TIMEOUT_S)

    def _reset(self) -> None:
        self._resample = StreamResampler(TARGET_RATE)
        self._segmenter = UtteranceSegmenter(sample_rate=TARGET_RATE)
        self._pending = np.zeros(0, dtype=np.float32)

    def _loop(self) -> None:
        self._reset()
        while not self._stop.is_set():
            try:
                chunk = self._chunks.get(timeout=QUEUE_POLL_S)
            except queue.Empty:
                continue
            try:
                self._process(chunk)
            except Exception as exc:  # resampler, VAD, segmenter: report, start clean, keep the session alive
                logger.exception("speech: processing failed for seq %s", chunk.seq)
                self._emit(
                    chunk.seq, chunk.t_send, {"type": "error", "error": f"{type(exc).__name__}: {exc}"}, None
                )
                self._reset()

    def _process(self, chunk: AudioChunk) -> None:
        self._pending = np.concatenate([self._pending, self._resample(chunk.samples, chunk.sample_rate)])
        while len(self._pending) >= CHUNK_SAMPLES and not self._stop.is_set():
            piece, self._pending = self._pending[:CHUNK_SAMPLES], self._pending[CHUNK_SAMPLES:]
            for event in self._segmenter.feed(piece, self._is_speech(piece)):
                if self._stop.is_set():
                    return
                self._handle(event, chunk)

    def _handle(self, event: SegmentEvent, chunk: AudioChunk) -> None:
        if event.kind == "start":
            self._emit(chunk.seq, chunk.t_send, {"type": "start"}, None)
            return
        if event.audio is None:  # blip too short to transcribe
            self._emit(chunk.seq, chunk.t_send, {"type": event.kind, "text": "", "audio_s": 0.0}, None)
            return
        t0 = time.perf_counter()
        try:
            text = self._transcribe(event.audio)
        except Exception as exc:
            logger.exception("speech: transcription failed")
            self._emit(
                chunk.seq, chunk.t_send, {"type": "error", "error": f"{type(exc).__name__}: {exc}"}, None
            )
            return
        inference_ms = (time.perf_counter() - t0) * 1000
        audio_s = round(len(event.audio) / TARGET_RATE, AUDIO_DECIMALS)
        self._emit(
            chunk.seq, chunk.t_send, {"type": event.kind, "text": text, "audio_s": audio_s}, inference_ms
        )


class SpeechNode:
    name = "speech"
    input_kind = InputKind.AUDIO

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL_SIZE,
        threads: int = DEFAULT_THREADS,
        transcriber_factory: Callable[[], Transcribe] | None = None,
        vad_factory: Callable[[], IsSpeech] | None = None,
        max_queued_chunks: int = MAX_QUEUED_CHUNKS,
    ) -> None:
        self._transcriber_factory = transcriber_factory or partial(_whisper_transcriber, model_size, threads)
        self._vad_factory = vad_factory or _silero_vad
        self._max_queued = max_queued_chunks
        self._transcribe: Transcribe | None = None
        self._is_speech: IsSpeech | None = None
        self._model_lock = threading.Lock()  # sessions share one whisper model

    def load(self) -> None:
        self._transcribe = self._transcriber_factory()
        self._is_speech = self._vad_factory()

    def _locked_transcribe(self, audio: np.ndarray) -> str:
        with self._model_lock:
            return self._transcribe(audio)

    def open_session(self, emit: Emit) -> SpeechSession:
        if self._transcribe is None or self._is_speech is None:
            raise RuntimeError("speech model not loaded")
        return SpeechSession(self._locked_transcribe, self._is_speech, emit, self._max_queued)
