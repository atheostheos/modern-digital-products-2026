# TouchDesigner nodes — how to attach them

Every node is a `.tox` in `td/toxes/`. There are three kinds:

| Kind | Nodes | Role |
|---|---|---|
| source | [ml_input](ml_input.md) | recorded or live video and audio |
| client | [ml_gesture](ml_gesture.md), [ml_classify](ml_classify.md), [ml_speech](ml_speech.md), [ml_sentiment](ml_sentiment.md), [ml_llm](ml_llm.md), [ml_server](ml_server.md) | stream input to one server node, turn replies into CHOPs, DATs and TOPs |
| viewer | [view_* and dashboard](viewers.md) | turn client outputs into 640×480 tiles and a 1280×640 dashboard |

The full wiring is already in `ai.toe`. The pages below explain how to rebuild it or reuse a node in
another network.

## Adding a node to a network

1. Save your `.toe` in `task-01/ai/`. Each node loads its Python extensions, callbacks and GLSL from
   `td/` relative to the project folder (for example `td/ext/GestureExt.py`). Edit those files on
   disk and TD reloads them.
2. Drag the `.tox` from Finder into the network editor (or use *File → Import*).
3. Wire its inputs. Connector names match the `in_*` operators inside the node (`in_video`,
   `in_text`, …). Connector order on the COMP is not the order the operators
   were built in, so match connectors by name, not by position.
4. Start the server ([quickstart](../quickstart.md)). The node connects by itself.

To rebuild a node from code instead (for example after changing the builder), run this in the
Textport:

```python
exec(open(project.folder + '/td/builder/build_clients.py', encoding='utf-8').read())
build('ml_gesture')          # or any ml_* name
exec(open(project.folder + '/td/builder/build_viewers.py', encoding='utf-8').read())
build_view('view_gesture'); wire_view('view_gesture')   # build_dashboard() for the dashboard
```

## Shared by every client node

**Server page parameters**

| Parameter | Default | Meaning |
|---|---|---|
| Host | `127.0.0.1` | server address |
| Port | `8000` | server port |
| Active | on | off = disconnect and stop sending (saves CPU) |
| Reconnect | pulse | drop and reopen the WebSocket |

**`out_status` CHOP** (1 sample, updated each reply)

| Channel | Meaning |
|---|---|
| `connected` | 1 while replies arrive |
| `sent`, `received` | message counters |
| `errors` | error replies (bad input, model error) |
| `rtt_ms` | TD send → reply received, measured by TD |
| `inference_ms` | model time reported by the server |
| `replies_per_s` | reply rate |

**Rate limiting.** Each server node keeps only the newest request ("latest wins"). A slow model
skips stale frames instead of queueing them, so latency stays bounded. Skipped frames appear as
`dropped` in [ml_server](ml_server.md).

## Example: the minimal gesture chain

```
ml_input ──out_video──▶ ml_gesture ──out_landmarks──▶ view_gesture.in_landmarks
     └──────out_video──────────────────────────────▶ view_gesture.in_video
                         ├─out_gestures──────────▶ view_gesture.in_gestures
                         └─out_hands─────────────▶ view_gesture.in_hands
```

Any TOP can stand in for `ml_input`: a Movie File In, a Video Device In, an NDI In, or a render.
