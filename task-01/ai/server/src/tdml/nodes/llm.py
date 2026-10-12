"""Narrator node: a local LLM (gemma3:1b via Ollama / llama.cpp) writes one poetic caption per scene.

Runs on CPU (`num_gpu: 0`): on this machine (Apple M5, macOS 26.0, Ollama 0.40.3) the Metal backend
produced corrupted output for gemma3:1b, while CPU output was correct at ~66 tok/s. The 1B model only
writes plain text; asking it for JSON made it copy the prompt's example (see the Phase 0 log).
"""

from collections.abc import Callable
from typing import Any

from tdml.nodes.base import InputKind

DEFAULT_MODEL = "gemma3:1b"
DEFAULT_HOST = "http://127.0.0.1:11434"
MAX_CAPTION_WORDS = 16
MAX_NEW_TOKENS = 40
TEMPERATURE = 0.7
KEEP_ALIVE = "30m"
NS_PER_S = 1_000_000_000
MS_PER_S = 1000
SYSTEM_PROMPT = (
    "You narrate a live audiovisual installation. Read the sensor data and answer with ONE poetic line "
    "of 5-12 words about the scene. No quotes, no explanations."
)
CPU_OPTIONS = {"num_gpu": 0, "temperature": TEMPERATURE, "num_predict": MAX_NEW_TOKENS}

ClientFactory = Callable[[str], Any]


def _ollama_client(host: str) -> Any:
    import ollama

    return ollama.Client(host=host)


def clean_caption(raw: str) -> str:
    """First non-empty line, outer quotes stripped, at most MAX_CAPTION_WORDS words."""
    line = next((part.strip() for part in raw.splitlines() if part.strip()), "")
    words = line.strip("\"'“”").split()
    return " ".join(words[:MAX_CAPTION_WORDS])


class LlmNode:
    name = "llm"
    input_kind = InputKind.TEXT

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self._model = model
        self._host = host
        self._factory = client_factory or _ollama_client
        self._client: Any = None

    def load(self) -> None:
        client = self._factory(self._host)
        # Warm-up: loads the weights now so the first real caption does not pay the ~2 s load.
        client.chat(
            model=self._model,
            messages=[{"role": "user", "content": "hi"}],
            options={**CPU_OPTIONS, "num_predict": 1},
            keep_alive=KEEP_ALIVE,
        )
        self._client = client

    def infer(self, scene: str) -> dict:
        if self._client is None:
            raise RuntimeError("llm not loaded")
        response = self._client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Scene: {scene}"},
            ],
            options=CPU_OPTIONS,
            keep_alive=KEEP_ALIVE,
        )
        caption = clean_caption(response.message.content)
        if not caption:
            raise ValueError("model returned an empty caption")
        eval_s = (response.eval_duration or 0) / NS_PER_S
        return {
            "caption": caption,
            "tokens": response.eval_count,
            "tokens_per_s": round(response.eval_count / eval_s, 1) if eval_s else None,
            "prompt_ms": round((response.prompt_eval_duration or 0) / NS_PER_S * MS_PER_S, 1),
        }
