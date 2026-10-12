"""Hand gesture node: MediaPipe Gesture Recognizer (hand landmarks + 7 gestures + handedness, CPU).

Landmarks are normalised image coordinates: x right, y DOWN (origin top-left), z relative depth
(smaller = closer to the camera). TouchDesigner's origin is bottom-left, so flip y there (1 - y).
"""

import logging
import urllib.request
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from tdml.config import models_dir
from tdml.nodes.base import InputKind

logger = logging.getLogger(__name__)

NUM_LANDMARKS = 21
DEFAULT_NUM_HANDS = 2
SCORE_DECIMALS = 4
COORD_DECIMALS = 5
DOWNLOAD_TIMEOUT_S = 60
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/latest/gesture_recognizer.task"
)

RecognizerFactory = Callable[[], Any]


def default_model_path() -> Path:
    return models_dir() / "mediapipe" / "gesture_recognizer.task"


def _ensure_model(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("downloading gesture model to %s", path)
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(MODEL_URL, timeout=DOWNLOAD_TIMEOUT_S) as response:
        tmp.write_bytes(response.read())
    tmp.rename(path)


def _mediapipe_factory(model_path: Path, num_hands: int) -> Any:
    from mediapipe.tasks.python import BaseOptions, vision

    _ensure_model(model_path)
    options = vision.GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=num_hands,
    )
    return vision.GestureRecognizer.create_from_options(options)


def _top_category(categories: list) -> tuple[str, float]:
    if not categories:
        return "None", 0.0
    return categories[0].category_name, round(float(categories[0].score), SCORE_DECIMALS)


def _hand(gestures: list, handedness: list, landmarks: list) -> dict:
    gesture, gesture_score = _top_category(gestures)
    side, side_score = _top_category(handedness)
    return {
        "handedness": side,
        "handedness_score": side_score,
        "gesture": gesture,
        "gesture_score": gesture_score,
        "landmarks": [
            [round(p.x, COORD_DECIMALS), round(p.y, COORD_DECIMALS), round(p.z, COORD_DECIMALS)]
            for p in landmarks
        ],
    }


class GestureNode:
    name = "gesture"
    input_kind = InputKind.IMAGE

    def __init__(
        self,
        model_path: Path | None = None,
        num_hands: int = DEFAULT_NUM_HANDS,
        recognizer_factory: RecognizerFactory | None = None,
    ) -> None:
        self._factory = recognizer_factory or partial(
            _mediapipe_factory, model_path or default_model_path(), num_hands
        )
        self._recognizer: Any = None

    def load(self) -> None:
        self._recognizer = self._factory()

    def infer(self, image: np.ndarray) -> dict:
        if self._recognizer is None:
            raise RuntimeError("gesture model not loaded")
        import mediapipe as mp

        frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image))
        result = self._recognizer.recognize(frame)
        hands = [
            _hand(gestures, handedness, landmarks)
            for gestures, handedness, landmarks in zip(
                result.gestures, result.handedness, result.hand_landmarks, strict=True
            )
        ]
        return {"count": len(hands), "hands": hands}
