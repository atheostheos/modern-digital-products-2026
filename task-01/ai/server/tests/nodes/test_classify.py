import numpy as np
import pytest

from tdml.nodes.base import InputKind
from tdml.nodes.classify import ClassifyNode

CATEGORIES = ["cat", "dog", "mug", "lamp", "guitar", "umbrella"]


def fake_factory():
    seen: list[np.ndarray] = []

    def predict(image: np.ndarray) -> np.ndarray:
        seen.append(image)
        return np.array([0.05, 0.1, 0.5, 0.2, 0.1, 0.05], dtype=np.float32)

    return (lambda: (predict, CATEGORIES)), seen


def test_node_contract():
    assert ClassifyNode.name == "classify"
    assert ClassifyNode.input_kind is InputKind.IMAGE


def test_returns_top_k_labels_sorted_by_score():
    factory, seen = fake_factory()
    node = ClassifyNode(top_k=3, model_factory=factory)
    node.load()
    image = np.zeros((48, 64, 3), dtype=np.uint8)

    result = node.infer(image)

    assert result["label"] == "mug"
    assert result["score"] == pytest.approx(0.5)
    assert [t["label"] for t in result["top"]] == ["mug", "lamp", "dog"]
    assert all(isinstance(t["score"], float) for t in result["top"])
    assert seen[0] is image


def test_top_k_is_clamped_to_number_of_classes():
    factory, _ = fake_factory()
    node = ClassifyNode(top_k=50, model_factory=factory)
    node.load()

    assert len(node.infer(np.zeros((8, 8, 3), dtype=np.uint8))["top"]) == len(CATEGORIES)


def test_infer_before_load_raises():
    with pytest.raises(RuntimeError, match="not loaded"):
        ClassifyNode().infer(np.zeros((8, 8, 3), dtype=np.uint8))


def test_invalid_top_k_is_rejected():
    with pytest.raises(ValueError, match="top_k"):
        ClassifyNode(top_k=0)


@pytest.mark.models
def test_real_model_returns_imagenet_labels_for_a_frame():
    node = ClassifyNode()
    node.load()

    result = node.infer(np.full((240, 320, 3), 128, dtype=np.uint8))

    assert len(result["top"]) == 5
    assert 0.0 < result["score"] <= 1.0
    assert sum(t["score"] for t in result["top"]) <= 1.0 + 1e-6
