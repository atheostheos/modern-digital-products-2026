import base64
import json
from types import SimpleNamespace as NS

import pytest

from tdml.nodes.base import InputKind
from tdml.nodes.llm import DEFAULT_MODEL, SCENES, LlmNode

NS_PER_S = 1_000_000_000


class FakeClient:
    """Answers the scene prompt and the emoji prompt with canned replies."""

    def __init__(self, scene_reply: str = "Storm.", emoji_reply: str = "🌧️") -> None:
        self.scene_reply, self.emoji_reply = scene_reply, emoji_reply
        self.calls: list[dict] = []

    def chat(self, **kwargs) -> NS:
        self.calls.append(kwargs)
        system = kwargs["messages"][0]["content"]
        reply = self.emoji_reply if "emoji" in system.lower() else self.scene_reply
        return NS(
            message=NS(content=reply), eval_count=4, eval_duration=NS_PER_S // 50, prompt_eval_duration=0
        )


def fake_renderer(emoji: str) -> bytes:
    return ("PNG:" + emoji).encode()


def make_node(scene_reply="Storm.", emoji_reply="🌧️", renderer=fake_renderer) -> tuple[LlmNode, FakeClient]:
    client = FakeClient(scene_reply, emoji_reply)
    node = LlmNode(client_factory=lambda host: client, emoji_renderer=renderer)
    node.load()
    return node, client


def scene_json(speech: str, valence: float = 0.0, gesture: str = "None") -> str:
    return json.dumps({"speech": speech, "gesture": gesture, "valence": valence})


def test_node_contract():
    assert LlmNode.name == "llm"
    assert LlmNode.input_kind is InputKind.TEXT
    assert SCENES == ("storm", "sunrise", "fireworks", "ocean", "neon", "starfield")


def test_load_warms_up_model_on_cpu():
    _, client = make_node()

    assert client.calls[0]["model"] == DEFAULT_MODEL
    assert client.calls[0]["options"]["num_gpu"] == 0


def test_infer_picks_scene_and_emoji_from_the_sentence():
    node, client = make_node()

    result = node.infer(scene_json("The rain outside is awful and my train was late again.", valence=-0.9))

    assert result["scene"] == "storm"
    assert result["scene_index"] == 0
    assert result["emoji"] == "🌧️"
    assert base64.b64decode(result["emoji_png"]) == "PNG:🌧️".encode()
    assert result["fallback"] == []
    assert result["tokens_per_s"] == 200.0
    asked = [call["messages"][1]["content"] for call in client.calls[1:]]
    assert asked == ["The rain outside is awful and my train was late again."] * 2
    assert all(call["options"]["num_gpu"] == 0 for call in client.calls)


def test_plain_sentence_input_is_accepted():
    node, _ = make_node(scene_reply="ocean")

    assert node.infer("I wish I was on a beach somewhere warm.")["scene"] == "ocean"


@pytest.mark.parametrize(
    ("valence", "scene", "emoji"),
    [(0.8, "sunrise", "😊"), (-0.8, "storm", "😟"), (0.1, "starfield", "😐")],
    ids=["positive", "negative", "neutral"],
)
def test_unusable_answers_fall_back_by_sentiment(valence, scene, emoji):
    node, _ = make_node(scene_reply="I think a nice garden.", emoji_reply="no idea")

    result = node.infer(scene_json("something", valence=valence))

    assert (result["scene"], result["emoji"]) == (scene, emoji)
    assert result["fallback"] == ["scene", "emoji"]


def test_scene_word_is_found_inside_a_longer_answer():
    node, _ = make_node(scene_reply="I would choose Fireworks for this!")

    assert node.infer(scene_json("let's make some noise"))["scene"] == "fireworks"


def test_empty_speech_makes_no_llm_calls():
    node, client = make_node()
    calls_after_load = len(client.calls)

    result = node.infer(scene_json("", valence=0.9))

    assert len(client.calls) == calls_after_load
    assert (result["scene"], result["emoji"], result["fallback"]) == ("sunrise", "😊", ["scene", "emoji"])


def test_missing_renderer_output_gives_no_png():
    node, _ = make_node(renderer=lambda emoji: None)

    assert node.infer(scene_json("hello there", 0.5))["emoji_png"] is None


def test_infer_before_load_raises():
    with pytest.raises(RuntimeError, match="not loaded"):
        LlmNode(client_factory=lambda host: FakeClient()).infer("scene")


@pytest.mark.models
def test_real_ollama_directs_a_rainy_sentence():
    node = LlmNode()
    node.load()

    result = node.infer(scene_json("The rain outside is awful and my train was late again.", valence=-1.0))

    assert result["scene"] in SCENES
    assert result["emoji"]
    assert result["emoji_png"]


class CountingClient(FakeClient):
    """Like FakeClient, but each prompt reports its own token count and duration (or None)."""

    def __init__(self, scene_eval: tuple, emoji_eval: tuple) -> None:
        super().__init__()
        self.evals = {"scene": scene_eval, "emoji": emoji_eval}

    def chat(self, **kwargs) -> NS:
        reply = super().chat(**kwargs)
        kind = "emoji" if "emoji" in kwargs["messages"][0]["content"].lower() else "scene"
        reply.eval_count, reply.eval_duration = self.evals[kind]
        return reply


def make_counting_node(scene_eval: tuple, emoji_eval: tuple) -> LlmNode:
    client = CountingClient(scene_eval, emoji_eval)
    node = LlmNode(client_factory=lambda host: client, emoji_renderer=fake_renderer)
    node.load()
    return node


def test_tokens_per_s_covers_both_prompts():
    node = make_counting_node(scene_eval=(9, NS_PER_S // 10), emoji_eval=(3, NS_PER_S // 10))

    result = node.infer(scene_json("Waves crash on the shore."))

    assert result["tokens_per_s"] == 60.0  # 12 tokens in 0.2 s


def test_missing_token_counts_give_no_speed_but_a_valid_reply():
    node = make_counting_node(scene_eval=(None, None), emoji_eval=(None, NS_PER_S // 10))

    result = node.infer(scene_json("Waves crash on the shore."))

    assert result["tokens_per_s"] is None
    assert result["scene"] == "storm"


def test_scene_mentioned_first_in_the_answer_wins():
    node, _ = make_node(scene_reply="Ocean, definitely not storm.")

    assert node.infer(scene_json("Waves crash on the shore."))["scene"] == "ocean"
