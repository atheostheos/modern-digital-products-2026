import numpy as np
import pytest

from tdml.audio.resample import StreamResampler

TARGET = 16000


def sine(rate: int, seconds: float, freq: float = 440.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(round(rate * seconds)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_target_rate_passes_through_unchanged():
    samples = sine(TARGET, 0.25)

    out = StreamResampler(TARGET)(samples, TARGET)

    assert np.array_equal(out, samples)


@pytest.mark.parametrize("rate", [44100, 48000])
def test_streamed_chunks_resample_to_target_length_and_keep_level(rate):
    resampler, signal = StreamResampler(TARGET), sine(rate, 2.0)
    step = rate // 4

    out = np.concatenate([resampler(signal[i : i + step], rate) for i in range(0, len(signal), step)])

    assert abs(len(out) - 2.0 * TARGET) <= 64  # filter delay only
    assert out.dtype == np.float32
    rms = float(np.sqrt(np.mean(out[1000:] ** 2)))
    assert rms == pytest.approx(0.5 / np.sqrt(2), rel=0.02)


def test_rate_change_mid_stream_starts_a_fresh_resampler():
    resampler = StreamResampler(TARGET)
    resampler(sine(48000, 0.5), 48000)

    out = resampler(sine(44100, 1.0), 44100)

    assert abs(len(out) - TARGET) <= 64
