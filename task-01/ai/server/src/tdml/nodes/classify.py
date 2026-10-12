"""Image classification node: MobileNetV3-Small, ImageNet-1k weights (PyTorch Hub / torchvision, CPU)."""

from collections.abc import Callable

import numpy as np

from tdml.nodes.base import InputKind

DEFAULT_TOP_K = 5
SCORE_DECIMALS = 4

Predict = Callable[[np.ndarray], np.ndarray]  # RGB uint8 (h, w, 3) -> class probabilities
ModelFactory = Callable[[], tuple[Predict, list[str]]]


def _torchvision_factory() -> tuple[Predict, list[str]]:
    import torch
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = mobilenet_v3_small(weights=weights).eval()
    transform = weights.transforms()  # resize 256, centre crop 224, normalise

    def predict(image: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
        with torch.inference_mode():
            logits = model(transform(tensor).unsqueeze(0))
        return torch.softmax(logits, dim=1)[0].numpy()

    return predict, list(weights.meta["categories"])


class ClassifyNode:
    name = "classify"
    input_kind = InputKind.IMAGE

    def __init__(self, top_k: int = DEFAULT_TOP_K, model_factory: ModelFactory | None = None) -> None:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        self._top_k = top_k
        self._factory = model_factory or _torchvision_factory
        self._predict: Predict | None = None
        self._categories: list[str] = []

    def load(self) -> None:
        self._predict, self._categories = self._factory()

    def infer(self, image: np.ndarray) -> dict:
        if self._predict is None:
            raise RuntimeError("classification model not loaded")
        probs = self._predict(image)
        best = np.argsort(-probs, kind="stable")[: self._top_k]  # ties keep class order
        top = [{"label": self._categories[i], "score": round(float(probs[i]), SCORE_DECIMALS)} for i in best]
        return {"label": top[0]["label"], "score": top[0]["score"], "top": top}
