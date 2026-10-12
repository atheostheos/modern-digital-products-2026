"""Shared fixtures: fake nodes (no ML models) and synthetic JPEG frames."""

import io
import struct
import threading
import time

import numpy as np
import pytest
from PIL import Image

from tdml.nodes.base import InputKind

FRAME_HEADER = struct.Struct("<Id")


def make_jpeg(width: int = 32, height: int = 24, color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="JPEG")
    return buf.getvalue()


def make_frame(seq: int, t_send: float = 1.5, jpeg: bytes | None = None) -> bytes:
    return FRAME_HEADER.pack(seq, t_send) + (jpeg if jpeg is not None else make_jpeg())


class EchoTextNode:
    """Text node: upper-cases the text."""

    name = "echo_text"
    input_kind = InputKind.TEXT

    def __init__(self) -> None:
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def infer(self, text: str) -> dict:
        return {"upper": text.upper()}


class ShapeImageNode:
    """Image node: reports the decoded image shape and mean colour."""

    name = "shape_image"
    input_kind = InputKind.IMAGE

    def __init__(self) -> None:
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def infer(self, image: np.ndarray) -> dict:
        return {"h": int(image.shape[0]), "w": int(image.shape[1]), "mean_r": float(image[..., 0].mean())}


class FailingNode:
    """Text node whose inference always raises."""

    name = "failing"
    input_kind = InputKind.TEXT

    def load(self) -> None:
        return None

    def infer(self, text: str) -> dict:
        raise RuntimeError("model exploded")


class GatedNode:
    """Text node that blocks inside infer until released, to make dropping deterministic."""

    name = "gated"
    input_kind = InputKind.TEXT

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.seen: list[str] = []

    def load(self) -> None:
        return None

    def infer(self, text: str) -> dict:
        self.entered.set()
        self.release.wait(timeout=5)
        self.seen.append(text)
        return {"text": text}


@pytest.fixture
def fake_nodes() -> list:
    return [EchoTextNode(), ShapeImageNode(), FailingNode()]


def wait_until(predicate, timeout: float = 2.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()
