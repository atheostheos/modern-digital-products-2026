import json

import pytest

from tdml.protocol import (
    MAX_FRAME_BYTES,
    MAX_TEXT_CHARS,
    ProtocolError,
    Timing,
    decode_image,
    failure,
    parse_frame,
    parse_text,
    success,
)
from tests.conftest import make_frame, make_jpeg


class TestParseFrame:
    def test_returns_seq_t_send_and_rgb_image(self):
        frame = parse_frame(make_frame(seq=7, t_send=3.25, jpeg=make_jpeg(40, 30, (0, 255, 0))))

        assert frame.seq == 7
        assert frame.t_send == 3.25
        assert frame.image.shape == (30, 40, 3)
        assert frame.image.dtype.name == "uint8"
        assert frame.image[..., 1].mean() > 200

    def test_rejects_payload_shorter_than_header(self):
        with pytest.raises(ProtocolError, match="too short"):
            parse_frame(b"\x00\x01")

    def test_rejects_oversized_payload(self):
        with pytest.raises(ProtocolError, match="too large"):
            parse_frame(b"\x00" * (MAX_FRAME_BYTES + 1))

    def test_rejects_non_image_bytes(self):
        with pytest.raises(ProtocolError, match="not a valid image"):
            parse_frame(make_frame(seq=1, jpeg=b"definitely not a jpeg"))


class TestDecodeImage:
    def test_rejects_images_with_too_many_pixels(self):
        with pytest.raises(ProtocolError, match="too large"):
            decode_image(make_jpeg(4096, 4096))

    def test_converts_greyscale_to_rgb(self):
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("L", (8, 8), 128).save(buf, format="PNG")

        image = decode_image(buf.getvalue())

        assert image.shape == (8, 8, 3)


class TestParseText:
    def test_parses_valid_message(self):
        request = parse_text(json.dumps({"seq": 3, "t_send": 1.0, "text": "hello"}))

        assert (request.seq, request.t_send, request.text) == (3, 1.0, "hello")

    def test_t_send_is_optional(self):
        assert parse_text(json.dumps({"seq": 0, "text": "hi"})).t_send is None

    @pytest.mark.parametrize(
        "message",
        [
            "not json",
            json.dumps({"seq": 1}),
            json.dumps({"seq": -1, "text": "x"}),
            json.dumps({"seq": 1, "text": ""}),
            json.dumps({"seq": 1, "text": "x" * (MAX_TEXT_CHARS + 1)}),
        ],
        ids=["bad-json", "missing-text", "negative-seq", "empty-text", "too-long"],
    )
    def test_rejects_invalid_messages(self, message):
        with pytest.raises(ProtocolError):
            parse_text(message)


class TestEnvelope:
    def test_success_envelope_serialises_all_fields(self):
        timing = Timing(t_recv=1.0, t_done=1.02, inference_ms=15.0)

        payload = json.loads(success("sentiment", 4, {"label": "POSITIVE"}, timing, t_send=0.5).to_json())

        assert payload == {
            "ok": True,
            "node": "sentiment",
            "seq": 4,
            "t_send": 0.5,
            "data": {"label": "POSITIVE"},
            "error": None,
            "timing": {"t_recv": 1.0, "t_done": 1.02, "inference_ms": 15.0},
        }

    def test_failure_envelope_has_error_and_no_data(self):
        payload = json.loads(failure("gesture", 9, "bad frame").to_json())

        assert payload["ok"] is False
        assert payload["error"] == "bad frame"
        assert payload["data"] is None
        assert payload["timing"] is None


class TestToJsonable:
    def test_converts_numpy_scalars_arrays_and_non_finite_floats(self):
        import numpy as np

        from tdml.protocol import to_jsonable

        data = {
            "score": np.float32(0.5),
            "index": np.int64(3),
            "points": np.array([[1.0, 2.0], [3.0, np.nan]]),
            "nested": [{"inf": float("inf")}, (np.bool_(True), "x")],
            7: None,
        }

        assert to_jsonable(data) == {
            "score": 0.5,
            "index": 3,
            "points": [[1.0, 2.0], [3.0, None]],
            "nested": [{"inf": None}, [True, "x"]],
            "7": None,
        }

    def test_rejects_unsupported_objects(self):
        from tdml.protocol import to_jsonable

        with pytest.raises(TypeError, match="not JSON-serialisable"):
            to_jsonable({"model": object()})


def test_parse_text_rejects_oversized_raw_message_before_parsing():
    from tdml.protocol import MAX_TEXT_MESSAGE_CHARS

    with pytest.raises(ProtocolError, match="too long"):
        parse_text("x" * (MAX_TEXT_MESSAGE_CHARS + 1))


class TestParseAudio:
    def test_returns_header_fields_and_float32_samples(self):
        import numpy as np

        from tdml.protocol import parse_audio
        from tests.conftest import make_audio_frame

        chunk = parse_audio(
            make_audio_frame(seq=9, samples=np.linspace(-1, 1, 480), sample_rate=48000, t_send=4.0)
        )

        assert (chunk.seq, chunk.t_send, chunk.sample_rate) == (9, 4.0, 48000)
        assert chunk.samples.dtype.name == "float32"
        assert chunk.samples.shape == (480,)
        assert chunk.samples[0] == -1.0 and chunk.samples[-1] == 1.0

    @pytest.mark.parametrize(
        ("payload_kind", "message"),
        [("short", "too short"), ("odd", "multiple of 4"), ("rate", "sample rate"), ("nan", "non-finite")],
    )
    def test_rejects_invalid_audio(self, payload_kind, message):
        import numpy as np

        from tdml.protocol import AUDIO_HEADER, parse_audio
        from tests.conftest import make_audio_frame

        payloads = {
            "short": b"\x00" * 5,
            "odd": AUDIO_HEADER.pack(1, 0.0, 16000) + b"\x00" * 6,
            "rate": make_audio_frame(1, np.zeros(16), sample_rate=1000),
            "nan": make_audio_frame(1, np.array([0.0, np.nan], dtype=np.float32)),
        }
        with pytest.raises(ProtocolError, match=message):
            parse_audio(payloads[payload_kind])

    def test_rejects_oversized_audio(self):
        from tdml.protocol import parse_audio

        with pytest.raises(ProtocolError, match="too large"):
            parse_audio(b"\x00" * (MAX_FRAME_BYTES + 1))


class TestAudioLimits:
    def test_rejects_frames_longer_than_the_max_duration(self):
        import numpy as np

        from tdml.protocol import MAX_AUDIO_SAMPLES, parse_audio
        from tests.conftest import make_audio_frame

        with pytest.raises(ProtocolError, match="too long"):
            parse_audio(make_audio_frame(5, np.zeros(MAX_AUDIO_SAMPLES + 1), sample_rate=48000))

    def test_rejects_out_of_range_amplitudes(self):
        import numpy as np

        from tdml.protocol import parse_audio
        from tests.conftest import make_audio_frame

        with pytest.raises(ProtocolError, match="out of range"):
            parse_audio(make_audio_frame(5, np.array([0.0, 1e30], dtype=np.float32)))

    def test_error_carries_seq_when_header_was_readable(self):
        import numpy as np

        from tdml.protocol import parse_audio
        from tests.conftest import make_audio_frame

        with pytest.raises(ProtocolError) as exc_info:
            parse_audio(make_audio_frame(42, np.zeros(16), sample_rate=1000))

        assert exc_info.value.seq == 42

    def test_error_without_readable_header_has_no_seq(self):
        from tdml.protocol import parse_audio

        with pytest.raises(ProtocolError) as exc_info:
            parse_audio(b"\x00\x01")

        assert exc_info.value.seq is None
