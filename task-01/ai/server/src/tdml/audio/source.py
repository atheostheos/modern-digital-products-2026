"""Microphone capture with sounddevice (PortAudio): 16 kHz mono float32 chunks to a callback.

macOS asks for microphone permission for the app that runs the server (Terminal, IDE, ...).
"""

import logging
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_S = 0.25

OnChunk = Callable[[np.ndarray], None]


class SoundDeviceSource:
    def __init__(
        self,
        on_chunk: OnChunk,
        device: int | str | None = None,
        sample_rate: int = SAMPLE_RATE,
        chunk_s: float = CHUNK_S,
    ) -> None:
        self._on_chunk = on_chunk
        self._device = device
        self._sample_rate = sample_rate
        self._blocksize = round(sample_rate * chunk_s)
        self._stream = None

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:  # PortAudio thread
        if status:
            logger.debug("audio input status: %s", status)
        self._on_chunk(indata[:, 0].copy())

    def start(self) -> None:
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self._blocksize,
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()
        logger.info("microphone started: device=%s", self._device if self._device is not None else "default")

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
