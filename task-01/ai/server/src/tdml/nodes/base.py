"""Node contracts: request nodes (image/text) answer each message; audio nodes keep a session per client."""

from collections.abc import Callable
from enum import StrEnum
from typing import Any, Protocol


class InputKind(StrEnum):
    IMAGE = "image"  # binary JPEG frames, decoded to an RGB uint8 array before infer()
    TEXT = "text"  # UTF-8 text
    AUDIO = "audio"  # binary float32 PCM chunks streamed into a per-connection session


# (seq, t_send of the chunk that produced the event, event data, inference_ms or None)
Emit = Callable[[int, float | None, dict, float | None], None]


class Node(Protocol):
    name: str
    input_kind: InputKind

    def load(self) -> None:
        """Load model weights. Called once, off the event loop, before serving."""

    def infer(self, payload: Any) -> dict:
        """Run one inference. Must return a JSON-serialisable dict."""


class AudioSession(Protocol):
    def feed(self, chunk: Any) -> bool:
        """Queue one AudioChunk without blocking; False when it was dropped (queue full or closed)."""

    def close(self) -> None:
        """Stop processing and release resources. Must be idempotent."""


class AudioNode(Protocol):
    name: str
    input_kind: InputKind  # InputKind.AUDIO

    def load(self) -> None:
        """Load models. Called once, off the event loop."""

    def open_session(self, emit: Emit) -> AudioSession:
        """Start a stateful session for one client connection; `emit` may be called from any thread."""
