# tdml — local ML server for TouchDesigner nodes

Runs five small pretrained models behind one localhost server so TouchDesigner (or any client)
can stream frames, text and audio events without installing ML packages inside TD.

| Node | Content | Model | Framework |
|---|---|---|---|
| `gesture` | video | MediaPipe Gesture Recognizer | MediaPipe |
| `classify` | image | MobileNetV3-Small (ImageNet-1k) | PyTorch Hub / torchvision |
| `speech` | audio | faster-whisper `tiny.en` + Silero VAD | CTranslate2 |
| `sentiment` | text | DistilBERT SST-2 | Hugging Face transformers |
| `llm` | LLM | `gemma3:1b` | Ollama / llama.cpp (CPU) |

## Run

```bash
ollama serve                 # separate terminal; once: ollama pull gemma3:1b
uv run tdml-server           # http://127.0.0.1:8000, all nodes
uv run tdml-server --nodes sentiment,llm --threads 2
```

Model weights are cached in `../models/` (git-ignored).

## Interfaces

- `GET /health`, `GET /metrics`, WebSocket `/ws/metrics` (pushed every 0.5 s)
- WebSocket `/ws/{node}`: image nodes take binary `<Id` header (seq, t_send) + JPEG; text nodes take
  `{"seq", "t_send", "text"}`; `speech` pushes `start` / `partial` / `final` events
- REST `POST /api/{node}`: JSON `{"text": ...}` or a raw `image/jpeg` body

Replies are envelopes: `{ok, node, seq, t_send, data, error, timing}`.

## Test

```bash
uv run pytest --cov            # fast suite, no models
uv run pytest -m models        # real models (needs cached weights and a running Ollama)
uv run ruff check src tests
```
