# Quickstart

Five pretrained models run in one local Python server. TouchDesigner (TD) captures every input
(video, audio, text), streams it to the server over WebSockets and draws the results.

```
TouchDesigner (ai.toe)                                    ML server 127.0.0.1:8000
──────────────────────                                    ────────────────────────
ml_input ─ video ─▶ ml_gesture    ── JPEG ─ /ws/gesture   MediaPipe Gesture Recognizer
                 ─▶ ml_classify   ── JPEG ─ /ws/classify  MobileNetV3-Small (torchvision)
         ─ audio ─▶ ml_speech     ── PCM ── /ws/speech    faster-whisper tiny.en + Silero VAD
                      ├ final ─▶ ml_sentiment ─ /ws/sentiment  DistilBERT SST-2 (transformers)
                      └ final ─▶ ml_llm  ─ JSON ─ /ws/llm      gemma3:1b via Ollama (scene director)
                     ml_server ◀─ metrics every 0.5 s ─ /ws/metrics
view_* ◀─ results       dashboard ◀─ six view_* tiles ─▶ window_dashboard
```

Per-node wiring, parameters and outputs: [nodes/README.md](nodes/README.md).

## 1. One-time setup

You need:
- macOS on Apple Silicon (tested: M5, macOS 26.0);
- [uv](https://docs.astral.sh/uv/);
- [Ollama](https://ollama.com);
- TouchDesigner 2025.x (the free Non-Commercial licence works).

```bash
cd task-01/ai
ollama pull gemma3:1b                         # ~0.8 GB
cd server && uv sync                          # Python 3.12 venv with all model packages
uv run python ../scripts/make_media.py        # recorded test media → ../media (git-ignored)
```

On its first start the server downloads the model weights (about 350 MB) into `models/`, which is
git-ignored.

## 2. Start the services (two terminals)

```bash
ollama serve                                  # terminal 1 (skip if the Ollama app already runs)
cd task-01/ai/server && uv run tdml-server    # terminal 2: all five nodes on 127.0.0.1:8000
```

Wait for `Uvicorn running on http://127.0.0.1:8000`. Then check that
`curl http://127.0.0.1:8000/health` lists all five nodes with `"loaded": true`.

Useful options:

| Option | Effect |
|---|---|
| `--nodes sentiment,llm` | start only some nodes |
| `--threads 2` | CPU threads for torch and CTranslate2 |
| `--whisper-model base.en` | more accurate speech recognition, but slower |
| `--llm-model` | use a different Ollama model |
| `--port` | serve on another port |
| `--log-level debug` | more detailed logs |

`uv run tdml-server --help` lists them all.

## 3. Open the project

Open `task-01/ai/ai.toe` in TouchDesigner. Every client node connects on its own.

**Dashboard window.** `window_dashboard` shows the 1280×640 dashboard. Select it and pulse *Open*
on its Window page.

**Recorded vs live input.** Click `button_live` in the network editor. Its label switches between
*RECORDED media files* and *LIVE camera + mic*. The button is bound to `ml_input`'s **Live**
parameter, so toggling either one changes both.

**Listening to the audio.** `button_monitor`, bound to **Monitor**, plays the input audio on your
speakers. Keep it off while using the microphone, or the speakers will feed back into it.

**Recorded media.** The recorded media loops: a 6-slide gesture video and 6 spoken sentences.

**Permissions.** The first time you switch to live, macOS asks for camera and microphone access for
TouchDesigner. Allow both under *System Settings → Privacy & Security → Camera / Microphone*, then
restart TD. Only TD needs these permissions; the server never opens a device or a file.

## 4. What you should see

| Tile | Shows |
|---|---|
| gesture | hand skeleton over the video, gesture name and score |
| classify | top-5 ImageNet labels as animated bars (2 per second) |
| speech | grey partial text while someone talks; white final sentences |
| sentiment | valence gauge (−1…1), history trail, colour wash |
| llm | one of six generative worlds (storm, sunrise, fireworks, ocean, neon, starfield) that gemma3:1b picks for each sentence, plus an emoji burst |
| metrics | per node: inference p50/p95, fps, dropped frames; for the server: CPU % and RAM |

## Troubleshooting

| Symptom | Fix |
|---|---|
| A node's `connected` channel stays 0 | Server not running, or wrong Host/Port. Start it, then pulse **Reconnect** on the node. |
| Nodes stopped after a server restart | They reconnect by themselves within a few seconds; pulse **Reconnect** to force it. |
| llm always shows sunrise/storm/starfield with 😊/😟/😐 | Those are the sentiment fallbacks: Ollama is down or the model is missing. Run `ollama serve` and `ollama pull gemma3:1b`, then restart the server. `out_llm`'s `tokens_per_s` is 0 when the LLM did not answer. |
| LLM text is garbage | The Metal (GPU) backend corrupts gemma3:1b output on this setup. The server forces CPU (`num_gpu: 0`); leave that setting alone. |
| Speech never shows text | Check that audio arrives: turn on `button_monitor`. In live mode, check the mic permission and the device in `ml_input/audiodevicein_live`. |
| Live video is black | Camera permission missing, or another app is using the camera. Look for errors on `ml_input/videodevicein_live`. |
| Image is scrambled or cut off | The Non-Commercial licence caps textures at 1280×1280. All nodes stay under it; keep to it if you enlarge a view. |
| TD stutters | Lower **Send Rate** on `ml_gesture` (15 fps by default) or `ml_classify` (2 fps), or switch **Active** off on nodes you don't need. |
| A `.tox` loads with Python errors | The nodes read their scripts from `td/`, relative to the `.toe`. Save your `.toe` in `task-01/ai/` (see [nodes/README.md](nodes/README.md)). |

## Without TouchDesigner

`harness/stream_media.py` stands in for TD. It streams the same recorded media (or the webcam and
microphone with `--live`) and prints scored results:

```bash
cd task-01/ai
uv run --project server python harness/stream_media.py --duration 45
```

The REST API answers single requests (the API the task asks for):

```bash
curl -s -X POST http://127.0.0.1:8000/api/sentiment -H 'content-type: application/json' \
     -d '{"text": "I love this"}'
curl -s -X POST http://127.0.0.1:8000/api/classify -H 'content-type: image/jpeg' \
     --data-binary @media/samples/victory.jpg
```
