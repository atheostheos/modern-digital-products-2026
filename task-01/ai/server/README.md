# tdml — local ML server for TouchDesigner nodes

Runs five small pretrained models behind one localhost server. TouchDesigner (or any client) captures
every input — camera or recorded video, microphone or recorded audio, text — and streams it here; the
server never opens a device or media file and needs no ML packages inside TD.

| Node | Content | Model | Framework |
|---|---|---|---|
| `gesture` | video | MediaPipe Gesture Recognizer | MediaPipe |
| `classify` | image | MobileNetV3-Small (ImageNet-1k) | PyTorch Hub / torchvision |
| `speech` | audio | faster-whisper `tiny.en` + Silero VAD | CTranslate2 |
| `sentiment` | text | DistilBERT SST-2 | Hugging Face transformers |
| `llm` | LLM (scene director) | `gemma3:1b` picks a visual world + an emoji per sentence | Ollama / llama.cpp (CPU) |

TouchDesigner setup and per-node wiring: [../docs/quickstart.md](../docs/quickstart.md),
[../docs/nodes/](../docs/nodes/README.md).

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
  `{"seq", "t_send", "text"}`; `speech` takes binary `<IdI` header (seq, t_send, sample_rate) + float32
  mono PCM (any rate, resampled to 16 kHz) and answers on the same socket with `start` / `partial` /
  `final` events
- REST `POST /api/{node}`: JSON `{"text": ...}` or a raw `image/jpeg` body

Replies are envelopes: `{ok, node, seq, t_send, data, error, timing}`.

## Try it without TouchDesigner

```bash
uv run python ../scripts/make_media.py          # recorded test media into ../media (git-ignored)
uv run python ../harness/stream_media.py        # streams recorded video + audio like TD would
uv run python ../harness/stream_media.py --live # webcam + microphone instead
```

## Test

```bash
uv run pytest --cov            # fast suite, no models
uv run pytest -m models        # real models (needs cached weights and a running Ollama)
uv run ruff check src tests
```
