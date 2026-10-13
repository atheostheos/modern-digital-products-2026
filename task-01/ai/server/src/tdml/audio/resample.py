"""Streaming resampler (PyAV / libswresample): any client sample rate -> 16 kHz mono float32.

Stateful across chunks, so filtering is continuous at chunk boundaries; a rate change starts afresh.
"""

import numpy as np

TARGET_RATE = 16000


class StreamResampler:
    def __init__(self, target_rate: int = TARGET_RATE) -> None:
        self._target = target_rate
        self._rate: int | None = None
        self._resampler = None

    def __call__(self, samples: np.ndarray, rate: int) -> np.ndarray:
        if rate == self._target:
            return samples.astype(np.float32, copy=False)
        import av

        if rate != self._rate:
            self._resampler = av.AudioResampler(format="flt", layout="mono", rate=self._target)
            self._rate = rate
        frame = av.AudioFrame.from_ndarray(
            np.ascontiguousarray(samples, dtype=np.float32).reshape(1, -1), format="flt", layout="mono"
        )
        frame.sample_rate = rate
        out = [resampled.to_ndarray().reshape(-1) for resampled in self._resampler.resample(frame)]
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
