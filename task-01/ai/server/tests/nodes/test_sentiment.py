import pytest

from tdml.nodes.base import InputKind
from tdml.nodes.sentiment import MODEL_ID, SentimentNode


class FakePipeline:
    def __init__(self, positive: float) -> None:
        self.positive = positive
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, text: str, **kwargs) -> list[list[dict]]:
        self.calls.append((text, kwargs))
        return [
            [{"label": "POSITIVE", "score": self.positive}, {"label": "NEGATIVE", "score": 1 - self.positive}]
        ]


def make_node(positive: float = 0.9) -> tuple[SentimentNode, FakePipeline, list[str]]:
    pipe, loaded_ids = FakePipeline(positive), []

    def factory(model_id: str) -> FakePipeline:
        loaded_ids.append(model_id)
        return pipe

    node = SentimentNode(pipeline_factory=factory)
    node.load()
    return node, pipe, loaded_ids


def test_node_contract():
    node = SentimentNode()

    assert node.name == "sentiment"
    assert node.input_kind is InputKind.TEXT


def test_load_uses_default_model_id():
    _, _, loaded_ids = make_node()

    assert loaded_ids == [MODEL_ID]


def test_positive_text_gives_positive_label_and_valence():
    node, pipe, _ = make_node(positive=0.9)

    result = node.infer("what a lovely day")

    assert result == {"label": "positive", "positive": 0.9, "negative": 0.1, "valence": 0.8}
    assert pipe.calls[-1] == ("what a lovely day", {"truncation": True})


def test_load_runs_a_warm_up_inference():
    _, pipe, _ = make_node()

    assert len(pipe.calls) == 1  # first real request should not pay the ~0.5 s lazy init


def test_negative_text_gives_negative_label_and_valence():
    node, _, _ = make_node(positive=0.25)

    result = node.infer("this is awful")

    assert result["label"] == "negative"
    assert result["valence"] == pytest.approx(-0.5)


def test_infer_before_load_raises():
    with pytest.raises(RuntimeError, match="not loaded"):
        SentimentNode().infer("hello")


@pytest.mark.models
def test_real_model_separates_clear_positive_and_negative():
    node = SentimentNode()
    node.load()

    positive = node.infer("I absolutely love how this looks!")
    negative = node.infer("This is the worst thing I have ever seen.")

    assert positive["label"] == "positive" and positive["valence"] > 0.9
    assert negative["label"] == "negative" and negative["valence"] < -0.9
