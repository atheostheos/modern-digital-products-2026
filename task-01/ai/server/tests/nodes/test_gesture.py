from types import SimpleNamespace as NS

import numpy as np
import pytest

from tdml.nodes.base import InputKind
from tdml.nodes.gesture import NUM_LANDMARKS, GestureNode


def landmarks(offset: float) -> list:
    return [NS(x=offset + i / 100, y=0.5, z=-0.01 * i) for i in range(NUM_LANDMARKS)]


def fake_result(hands: int) -> NS:
    return NS(
        gestures=[[NS(category_name="Open_Palm", score=0.91)], [NS(category_name="None", score=0.6)]][:hands],
        handedness=[[NS(category_name="Right", score=0.97)], [NS(category_name="Left", score=0.88)]][:hands],
        hand_landmarks=[landmarks(0.1), landmarks(0.6)][:hands],
    )


class FakeRecognizer:
    def __init__(self, hands: int) -> None:
        self.hands = hands
        self.images = []

    def recognize(self, image) -> NS:
        self.images.append(image)
        return fake_result(self.hands)


def make_node(hands: int) -> tuple[GestureNode, FakeRecognizer]:
    recognizer = FakeRecognizer(hands)
    node = GestureNode(recognizer_factory=lambda: recognizer)
    node.load()
    return node, recognizer


def test_node_contract():
    assert GestureNode.name == "gesture"
    assert GestureNode.input_kind is InputKind.IMAGE


def test_two_hands_are_reported_with_gesture_handedness_and_21_landmarks():
    node, recognizer = make_node(hands=2)

    result = node.infer(np.zeros((120, 160, 3), dtype=np.uint8))

    assert result["count"] == 2
    first = result["hands"][0]
    assert (first["gesture"], first["handedness"]) == ("Open_Palm", "Right")
    assert first["gesture_score"] == pytest.approx(0.91)
    assert len(first["landmarks"]) == NUM_LANDMARKS
    assert first["landmarks"][1] == pytest.approx([0.11, 0.5, -0.01])
    assert result["hands"][1]["handedness"] == "Left"
    assert recognizer.images[0].numpy_view().shape == (120, 160, 3)


def test_no_hands_gives_empty_list():
    node, _ = make_node(hands=0)

    assert node.infer(np.zeros((8, 8, 3), dtype=np.uint8)) == {"count": 0, "hands": []}


def test_hand_without_gesture_category_is_reported_as_none():
    node, recognizer = make_node(hands=1)
    recognizer.recognize = lambda image: NS(
        gestures=[[]], handedness=[[NS(category_name="Left", score=0.9)]], hand_landmarks=[landmarks(0.0)]
    )

    hand = node.infer(np.zeros((8, 8, 3), dtype=np.uint8))["hands"][0]

    assert (hand["gesture"], hand["gesture_score"]) == ("None", 0.0)


def test_infer_before_load_raises():
    with pytest.raises(RuntimeError, match="not loaded"):
        GestureNode().infer(np.zeros((8, 8, 3), dtype=np.uint8))


@pytest.mark.models
def test_real_recognizer_handles_frame_without_hands():
    node = GestureNode()
    node.load()

    assert node.infer(np.zeros((240, 320, 3), dtype=np.uint8)) == {"count": 0, "hands": []}
