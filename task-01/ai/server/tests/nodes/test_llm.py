from types import SimpleNamespace as NS

import pytest

from tdml.nodes.base import InputKind
from tdml.nodes.llm import DEFAULT_MODEL, MAX_CAPTION_WORDS, LlmNode

NS_PER_S = 1_000_000_000


class FakeClient:
    def __init__(self, reply: str = "Rain falls, a muted, gray surrender.") -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def chat(self, **kwargs) -> NS:
        self.calls.append(kwargs)
        return NS(
            message=NS(content=self.reply),
            eval_count=10,
            eval_duration=NS_PER_S // 5,  # 0.2 s -> 50 tok/s
            prompt_eval_duration=NS_PER_S // 20,  # 50 ms
        )


def make_node(reply: str = "Rain falls, a muted, gray surrender.") -> tuple[LlmNode, FakeClient]:
    client = FakeClient(reply)
    node = LlmNode(client_factory=lambda host: client)
    node.load()
    return node, client


def test_node_contract():
    assert LlmNode.name == "llm"
    assert LlmNode.input_kind is InputKind.TEXT


def test_load_warms_up_model_on_cpu():
    _, client = make_node()

    warmup = client.calls[0]
    assert warmup["model"] == DEFAULT_MODEL
    assert warmup["options"]["num_gpu"] == 0


def test_infer_returns_caption_and_speed_metrics():
    node, client = make_node()

    result = node.infer('{"gesture": "Thumb_Down", "speech": "this rain is awful"}')

    assert result == {
        "caption": "Rain falls, a muted, gray surrender.",
        "tokens": 10,
        "tokens_per_s": 50.0,
        "prompt_ms": 50.0,
    }
    request = client.calls[-1]
    assert request["messages"][0]["role"] == "system"
    assert "Thumb_Down" in request["messages"][1]["content"]
    assert request["options"]["num_gpu"] == 0


@pytest.mark.parametrize(
    ("reply", "caption"),
    [
        ('  "Quiet light on a sleepy desk."  ', "Quiet light on a sleepy desk."),
        ("\n\nFirst line here.\nSecond line ignored.", "First line here."),
        (
            "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen"
            " seventeen eighteen",
            " ".join(
                [
                    "one",
                    "two",
                    "three",
                    "four",
                    "five",
                    "six",
                    "seven",
                    "eight",
                    "nine",
                    "ten",
                    "eleven",
                    "twelve",
                    "thirteen",
                    "fourteen",
                    "fifteen",
                    "sixteen",
                ]
            ),
        ),
    ],
    ids=["quotes", "first-line", "too-long"],
)
def test_caption_is_cleaned(reply, caption):
    node, _ = make_node(reply)

    assert node.infer("scene")["caption"] == caption
    assert len(caption.split()) <= MAX_CAPTION_WORDS


def test_empty_model_output_is_an_error():
    node, _ = make_node("   \n  ")

    with pytest.raises(ValueError, match="empty caption"):
        node.infer("scene")


def test_infer_before_load_raises():
    with pytest.raises(RuntimeError, match="not loaded"):
        LlmNode(client_factory=lambda host: FakeClient()).infer("scene")


@pytest.mark.models
def test_real_ollama_writes_a_short_caption():
    node = LlmNode()
    node.load()

    result = node.infer(
        '{"gesture": "Victory", "image": "electric guitar", "speech": "let us make some noise"}'
    )

    assert 2 <= len(result["caption"].split()) <= MAX_CAPTION_WORDS
    assert result["tokens_per_s"] > 5
