# ml_llm — scene director (gemma3:1b via Ollama)

After each new sentence the local LLM makes two choices that need language understanding:

- which of six generative **worlds** fits the sentence: `storm sunrise fireworks ocean neon starfield`;
- one **emoji** that captures it.

The server renders the emoji to a PNG with Pillow (TD's Text TOP cannot draw colour emoji). It
checks both answers; if either is unusable it falls back on sentiment (valence > 0.3 → sunrise/😊,
< −0.3 → storm/😟, otherwise starfield/😐), so the visuals never break. Inference runs on CPU and
takes about 0.6 s per sentence.

| | |
|---|---|
| Input | `in_scene` (optional) — a DAT with a plain sentence or JSON `{"speech", "gesture", "valence"}`. If unwired, the node reads the nodes named on its Narrator page. |
| Server | `/ws/llm`; also `POST /api/llm` |

## Parameters (Narrator page)

| Parameter | Default | Meaning |
|---|---|---|
| React to Speech (Auto) | on | ask whenever a new sentence appears (after the Sentiment node has scored it, at most 1 s wait, so the fallback uses this sentence's valence) |
| Min Interval (s) | 2 | at most one request per interval |
| Send Now | pulse | ask now with the current context |
| Speech Node | `ml_speech` | its `text_final` is the sentence |
| Gesture Node | `ml_gesture` | its first hand's gesture is sent as context |
| Sentiment Node | `ml_sentiment` | its `valence` is sent (used for the fallback) |

## Outputs

| Output | Type | Content |
|---|---|---|
| `out_scene` | text DAT | chosen world name |
| `out_symbol` | text DAT | the emoji character |
| `out_emoji` | TOP | the emoji as premultiplied RGBA (128×128) |
| `out_llm` | CHOP, 1 sample | `scene` (index 0–5), `scene_storm` … `scene_starfield` (one-hot, ready for a Lag CHOP crossfade), `directed_frame` (frame the last answer arrived, useful to trigger bursts), `tokens_per_s`, `latency_ms`, `directions` (count) |
| `out_context` | text DAT | the exact request that was sent |
| `out_status` | CHOP | see [README](README.md) |

## Wiring

```
(Narrator page: Speech = ml_speech, Gesture = ml_gesture, Sentiment = ml_sentiment)
out_emoji ─▶ view_llm.in_emoji     out_llm ─▶ view_llm.in_llm     out_scene ─▶ view_llm.in_scene
ml_speech.out_final ─▶ view_llm.in_quote     ml_sentiment.out_sentiment ─▶ view_llm.in_sentiment
```

Test without speech: wire a Text DAT containing `Fireworks over the sea tonight` into `in_scene`,
then pulse **Send Now**.
