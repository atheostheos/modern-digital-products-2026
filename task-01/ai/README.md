# ai — real-time ML nodes for TouchDesigner

Five pretrained models packaged as TouchDesigner (TD) nodes. TD captures every input (video,
audio, text) and streams it over WebSockets to one local Python server. The server runs the
models and sends results back, which TD draws as live visuals.

| Node | Content | Model | Framework |
|---|---|---|---|
| `ml_gesture` | video | MediaPipe Gesture Recognizer (21 landmarks per hand, 7 gestures) | MediaPipe |
| `ml_classify` | image | MobileNetV3-Small (ImageNet-1k) | PyTorch / torchvision |
| `ml_speech` | audio | faster-whisper `tiny.en` + Silero VAD (English speech to text) | CTranslate2 |
| `ml_sentiment` | text | DistilBERT fine-tuned on SST-2 | Hugging Face transformers |
| `ml_llm` | local LLM | `gemma3:1b` picks a visual world and an emoji for each sentence | Ollama / llama.cpp |

The server also exposes a REST API (`POST /api/{node}`) and a live metrics stream.

## Requirements

- **Hardware and OS:** macOS on Apple Silicon (tested on an M5, macOS 26.0).
  - The emoji renderer uses the system *Apple Color Emoji* font.
  - `scripts/make_media.py` uses `say`.
- **[uv](https://docs.astral.sh/uv/):** installs Python 3.12 and all packages.
- **[Ollama](https://ollama.com):** for the local LLM.
- **[TouchDesigner](https://derivative.ca) 2025.x:** the free Non-Commercial licence is enough.

  The licence caps textures at 1280×1280, and every node stays within that limit.
- **About 2.5 GB of disk:**
  - ~1.2 GB of Python packages in `server/.venv`;
  - ~350 MB of model weights in `models/`;
  - ~0.8 GB for gemma3:1b.

## Setup (once)

```bash
cd task-01/ai
make setup     # Python 3.12 venv with locked dependencies (server/uv.lock)
make llm       # ollama pull gemma3:1b
make media     # recorded test media -> media/ (gesture video + spoken sentences)
```

The first server start downloads the model weights into `models/`. After that, `make server-offline`
starts without any network access.

## Run

```bash
make ollama    # terminal 1 (skip if the Ollama app is already running)
make server    # terminal 2: all five nodes on http://127.0.0.1:8000
make health    # every node should report "loaded": true
make td        # open ai.toe in TouchDesigner
```

In TD:
1. Open the dashboard: select `window_dashboard` and pulse *Open*. It shows the six viewer tiles.
2. To switch between recorded media and the live camera and microphone, click `button_live` in the
   network editor.
3. To hear the input audio, use `button_monitor`.

The first switch to live makes macOS ask for camera and microphone access for TouchDesigner. The
server itself never opens a device or a file.

**Without TouchDesigner:** `make harness` streams the same recorded media the way TD does and
prints scored results; `make harness-live` uses the webcam and microphone instead.

`make help` lists every target. Server options:
```bash
make server NODES=sentiment,llm THREADS=2
```

## Develop and test

```bash
make check         # ruff + fast test suite with coverage (no models needed)
make test-models   # real-model tests (cached weights + running Ollama)
make format        # auto-fix lint and formatting
```

**Editing TD nodes.** The nodes load their Python extensions, callbacks and GLSL shaders from `td/`,
relative to `ai.toe`. Edit those files on disk and TD reloads them live. The node builders in
`td/builder/` recreate any node from code, and `td/toxes/` holds every node as a `.tox` for use in
other projects.

## Layout

```
server/        tdml: FastAPI server, protocol, latest-wins workers, model nodes, tests (uv project)
td/ext/        TD Python extensions, one per client node (+ shared MlClientExt)
td/callbacks/  WebSocket / Script / Execute DAT callbacks shared by all nodes
td/glsl/       viewer and dashboard shaders + shared includes (common, worlds, hands)
td/builder/    build_clients.py, build_viewers.py: (re)build nodes from code inside TD
td/viewers/    helpers used by the viewer nodes (metrics table text)
td/toxes/      every node as a .tox
harness/       stream_media.py: TD stand-in client for testing and benchmarks
scripts/       make_media.py: generates the recorded test media
bench/         TD -> server transport benchmark
docs/          quickstart and per-node attach instructions
ai.toe         the TouchDesigner project
```

`media/`, `models/` and TD backups are git-ignored and are recreated by the steps above.

## Docs

- [docs/quickstart.md](docs/quickstart.md): full walkthrough and troubleshooting.
- [docs/nodes/](docs/nodes/README.md): how to attach each node, with its parameters, outputs and
  wiring.
- [server/README.md](server/README.md): server interfaces (WebSocket and REST protocol).
