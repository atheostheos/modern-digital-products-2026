"""Text sentiment node: DistilBERT fine-tuned on SST-2 (Hugging Face transformers, CPU).

Output is shaped for TouchDesigner: positive/negative probabilities plus `valence` in [-1, 1]
(positive - negative), which can drive a colour ramp or a CHOP directly.
"""

from collections.abc import Callable
from typing import Any

from tdml.nodes.base import InputKind

MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
SCORE_DECIMALS = 4

PipelineFactory = Callable[[str], Callable[..., Any]]


def _hf_pipeline(model_id: str) -> Callable[..., Any]:
    from transformers import pipeline  # heavy import, only when the node is actually loaded

    return pipeline("sentiment-analysis", model=model_id, device="cpu", top_k=None)


class SentimentNode:
    name = "sentiment"
    input_kind = InputKind.TEXT

    def __init__(self, model_id: str = MODEL_ID, pipeline_factory: PipelineFactory | None = None) -> None:
        self._model_id = model_id
        self._factory = pipeline_factory or _hf_pipeline
        self._pipe: Callable[..., Any] | None = None

    def load(self) -> None:
        self._pipe = self._factory(self._model_id)
        self._pipe("warm up", truncation=True)  # first real request should not pay the ~0.5 s lazy init

    def infer(self, text: str) -> dict:
        if self._pipe is None:
            raise RuntimeError("sentiment model not loaded")
        scores = {
            item["label"].lower(): float(item["score"]) for item in self._pipe(text, truncation=True)[0]
        }
        positive, negative = scores.get("positive", 0.0), scores.get("negative", 0.0)
        return {
            "label": "positive" if positive >= negative else "negative",
            "positive": round(positive, SCORE_DECIMALS),
            "negative": round(negative, SCORE_DECIMALS),
            "valence": round(positive - negative, SCORE_DECIMALS),
        }
