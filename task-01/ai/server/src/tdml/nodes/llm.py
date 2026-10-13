"""Scene director: a local LLM (gemma3:1b via Ollama / llama.cpp) turns what a visitor said into visuals.

For each finished sentence it makes two small decisions that need language understanding:
  * which of six generative worlds fits the sentence (storm, sunrise, fireworks, ocean, neon, starfield),
  * one emoji that captures it (rendered to a PNG here, because TD's Text TOP cannot draw colour emoji).
Answers are validated; anything unusable falls back by sentiment, so the visuals never break.

Runs on CPU (`num_gpu: 0`): on this machine (Apple M5, macOS 26.0, Ollama 0.40.3) the Metal backend
produced corrupted output for gemma3:1b, while CPU output was correct. Choosing from a fixed list is
reliable for the 1B model (8/8 in the spike) where free-form JSON was not.
"""

import base64
import json
import re
from collections.abc import Callable
from typing import Any

from tdml.nodes.base import InputKind
from tdml.nodes.emoji_render import render_emoji

DEFAULT_MODEL = "gemma3:1b"
DEFAULT_HOST = "http://127.0.0.1:11434"
SCENES = ("storm", "sunrise", "fireworks", "ocean", "neon", "starfield")
MAX_NEW_TOKENS = 24
TEMPERATURE = 0.4
KEEP_ALIVE = "30m"
NS_PER_S = 1_000_000_000
VALENCE_THRESHOLD = 0.3
CPU_OPTIONS = {"num_gpu": 0, "temperature": TEMPERATURE, "num_predict": MAX_NEW_TOKENS}

SCENE_PROMPT = (
    "You direct the visuals of an art installation. "
    "Choose the ONE scene that best matches what the visitor said. Scenes: "
    + ", ".join(SCENES)
    + ". Answer with just the scene word."
)
EMOJI_PROMPT = (
    "Answer with exactly ONE emoji that best captures the feeling and subject of the sentence. "
    "Only the emoji."
)
# One pictographic code point, optionally followed by a variation selector (e.g. 🌧️ = U+1F327 U+FE0F).
EMOJI_RE = re.compile("[\U0001f300-\U0001faff\u2600-\u27bf]\ufe0f?")

ClientFactory = Callable[[str], Any]
EmojiRenderer = Callable[[str], bytes | None]


def _ollama_client(host: str) -> Any:
    import ollama

    return ollama.Client(host=host)


def parse_scene(text: str) -> tuple[str, float, str]:
    """(speech, valence, gesture) from the scene JSON; a plain string is treated as the speech."""
    try:
        data = json.loads(text)
    except ValueError:
        return text.strip(), 0.0, "None"
    if not isinstance(data, dict):
        return text.strip(), 0.0, "None"
    valence = data.get("valence", 0.0)
    return (
        str(data.get("speech", "")).strip(),
        float(valence) if isinstance(valence, int | float) else 0.0,
        str(data.get("gesture", "None")),
    )


def fallback_scene(valence: float) -> str:
    if valence > VALENCE_THRESHOLD:
        return "sunrise"
    return "storm" if valence < -VALENCE_THRESHOLD else "starfield"


def fallback_emoji(valence: float) -> str:
    if valence > VALENCE_THRESHOLD:
        return "😊"
    return "😟" if valence < -VALENCE_THRESHOLD else "😐"


def pick_scene(answer: str) -> str | None:
    """The scene word mentioned first in the answer ("ocean, not storm" -> ocean)."""
    lowered = answer.lower()
    found = [(lowered.find(scene), scene) for scene in SCENES if scene in lowered]
    return min(found)[1] if found else None


def tokens_per_second(responses: list[Any]) -> float | None:
    """Generation speed over all calls; None when Ollama reported no counts or durations."""
    tokens = sum(response.eval_count or 0 for response in responses)
    seconds = sum(response.eval_duration or 0 for response in responses) / NS_PER_S
    return round(tokens / seconds, 1) if tokens and seconds else None


def pick_emoji(answer: str) -> str | None:
    match = EMOJI_RE.search(answer)
    return match.group(0) if match else None


class LlmNode:
    name = "llm"
    input_kind = InputKind.TEXT

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        client_factory: ClientFactory | None = None,
        emoji_renderer: EmojiRenderer | None = None,
    ) -> None:
        self._model = model
        self._host = host
        self._factory = client_factory or _ollama_client
        self._render = emoji_renderer or render_emoji
        self._client: Any = None

    def load(self) -> None:
        client = self._factory(self._host)
        # Warm-up: loads the weights now so the first real request does not pay the ~2 s load.
        client.chat(
            model=self._model,
            messages=[{"role": "user", "content": "hi"}],
            options={**CPU_OPTIONS, "num_predict": 1},
            keep_alive=KEEP_ALIVE,
        )
        self._client = client

    def _ask(self, system: str, sentence: str) -> tuple[str, Any]:  # (answer text, raw response)
        response = self._client.chat(
            model=self._model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": sentence}],
            options=CPU_OPTIONS,
            keep_alive=KEEP_ALIVE,
        )
        return response.message.content, response

    def infer(self, scene_text: str) -> dict:
        if self._client is None:
            raise RuntimeError("llm not loaded")
        speech, valence, _ = parse_scene(scene_text)
        scene = emoji = None
        tokens_per_s = None
        if speech:
            scene_answer, scene_response = self._ask(SCENE_PROMPT, speech)
            emoji_answer, emoji_response = self._ask(EMOJI_PROMPT, speech)
            scene, emoji = pick_scene(scene_answer), pick_emoji(emoji_answer)
            tokens_per_s = tokens_per_second([scene_response, emoji_response])
        fallback = [name for name, value in (("scene", scene), ("emoji", emoji)) if value is None]
        scene = scene or fallback_scene(valence)
        emoji = emoji or fallback_emoji(valence)
        png = self._render(emoji)
        return {
            "scene": scene,
            "scene_index": SCENES.index(scene),
            "emoji": emoji,
            "emoji_png": base64.b64encode(png).decode("ascii") if png else None,
            "fallback": fallback,
            "tokens_per_s": tokens_per_s,
        }
