"""Node contract: every analysis node (gesture, classify, speech, sentiment, llm) implements this."""

from collections.abc import Callable
from enum import StrEnum
from typing import Any, Protocol


class InputKind(StrEnum):
    IMAGE = "image"  # binary JPEG frames, decoded to an RGB uint8 array before infer()
    TEXT = "text"  # UTF-8 text
    STREAM = "stream"  # no client input: the node produces events itself (e.g. microphone speech)


Publish = Callable[[dict, float | None], None]  # (event data, inference_ms or None)


class Node(Protocol):
    name: str
    input_kind: InputKind

    def load(self) -> None:
        """Load model weights. Called once, off the event loop, before serving."""

    def infer(self, payload: Any) -> dict:
        """Run one inference. Must return a JSON-serialisable dict."""


class StreamNode(Protocol):
    name: str
    input_kind: InputKind  # InputKind.STREAM

    def load(self) -> None:
        """Load models. Called once, off the event loop."""

    def start(self, publish: Publish) -> None:
        """Begin producing events; `publish` may be called from any thread."""

    def stop(self) -> None:
        """Stop producing events and release devices. Must be idempotent."""
