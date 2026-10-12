"""Wire protocol between TouchDesigner (or any client) and the ML server.

Image nodes: binary WebSocket message = header `<Id` (seq: uint32, t_send: float64) + JPEG bytes.
Text nodes:  text WebSocket message = JSON {"seq": int >= 0, "t_send": float | null, "text": str}.
Replies:     JSON envelope {ok, node, seq, t_send, data, error, timing}.
"""

import io
import math
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError

FRAME_HEADER = struct.Struct("<Id")
MAX_FRAME_BYTES = 4 * 1024 * 1024
MAX_IMAGE_PIXELS = 1920 * 1920
MAX_TEXT_CHARS = 2000
MAX_TEXT_MESSAGE_CHARS = MAX_TEXT_CHARS * 2 + 256  # raw JSON envelope around the text, checked before parsing
MAX_REST_BODY_BYTES = MAX_FRAME_BYTES
UNKNOWN_SEQ = -1


class ProtocolError(ValueError):
    """Client sent a message that violates the protocol; reported back, never fatal."""


class Timing(BaseModel):
    model_config = ConfigDict(frozen=True)

    t_recv: float
    t_done: float
    inference_ms: float


class Envelope(BaseModel):
    model_config = ConfigDict(frozen=True)

    ok: bool
    node: str
    seq: int
    t_send: float | None = None
    data: dict | None = None
    error: str | None = None
    timing: Timing | None = None

    def to_json(self) -> str:
        return self.model_dump_json()


def success(node: str, seq: int, data: dict, timing: Timing, t_send: float | None = None) -> Envelope:
    return Envelope(ok=True, node=node, seq=seq, t_send=t_send, data=data, timing=timing)


def failure(node: str, seq: int, error: str, t_send: float | None = None) -> Envelope:
    return Envelope(ok=False, node=node, seq=seq, t_send=t_send, error=error)


class TextRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    seq: int = Field(ge=0)
    t_send: float | None = None
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)


class TextBody(BaseModel):
    """REST body for text nodes."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)


@dataclass(frozen=True)
class FrameHeader:
    seq: int
    t_send: float
    jpeg: bytes


@dataclass(frozen=True)
class FrameRequest:
    seq: int
    t_send: float
    image: np.ndarray


def split_frame(payload: bytes) -> FrameHeader:
    """Validate size and split the header off; the JPEG is decoded later (off the event loop)."""
    if len(payload) > MAX_FRAME_BYTES:
        raise ProtocolError(f"frame too large: {len(payload)} bytes (max {MAX_FRAME_BYTES})")
    if len(payload) <= FRAME_HEADER.size:
        raise ProtocolError(f"frame too short: {len(payload)} bytes")
    seq, t_send = FRAME_HEADER.unpack_from(payload)
    return FrameHeader(seq=seq, t_send=t_send, jpeg=bytes(payload[FRAME_HEADER.size :]))


def decode_image(data: bytes) -> np.ndarray:
    """Decode an encoded image (JPEG/PNG) to an RGB uint8 array of shape (h, w, 3)."""
    try:
        image = Image.open(io.BytesIO(data))
    except Exception as exc:
        raise ProtocolError(f"not a valid image: {exc}") from exc
    with image:
        width, height = image.size
        if width * height > MAX_IMAGE_PIXELS:
            raise ProtocolError(f"image too large: {width}x{height} (max {MAX_IMAGE_PIXELS} pixels)")
        try:
            return np.asarray(image.convert("RGB"))
        except Exception as exc:
            raise ProtocolError(f"not a valid image: {exc}") from exc


def parse_frame(payload: bytes) -> FrameRequest:
    header = split_frame(payload)
    return FrameRequest(seq=header.seq, t_send=header.t_send, image=decode_image(header.jpeg))


def _validation_summary(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, e['loc'])) or 'message'}: {e['msg']}" for e in exc.errors())


def parse_text(message: str | bytes) -> TextRequest:
    if len(message) > MAX_TEXT_MESSAGE_CHARS:
        raise ProtocolError(f"text message too long: {len(message)} chars (max {MAX_TEXT_MESSAGE_CHARS})")
    try:
        return TextRequest.model_validate_json(message)
    except ValidationError as exc:
        raise ProtocolError(f"invalid text message: {_validation_summary(exc)}") from exc


def parse_text_body(body: bytes) -> str:
    try:
        return TextBody.model_validate_json(body).text
    except ValidationError as exc:
        raise ProtocolError(f"invalid body: {_validation_summary(exc)}") from exc


def to_jsonable(value: Any) -> Any:
    """Convert model outputs (numpy scalars/arrays, tuples, NaN/inf) to strict-JSON-safe values."""
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    raise TypeError(f"{type(value).__name__} is not JSON-serialisable")
