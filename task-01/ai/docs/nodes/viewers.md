# Viewers — view_* and dashboard

Each viewer is one GLSL Multi TOP whose shader lives on disk in `td/glsl/` (shared helpers in
`common.glsl`, `worlds.glsl`, `hands.glsl`), plus a few Text TOPs for labels. Each viewer outputs
`out_view`, a 640×480 TOP. Viewers have no parameters: edit the `.frag` file and TD recompiles it.

| Viewer | Inputs ← source | Shows | Shader |
|---|---|---|---|
| `view_gesture` | `in_video` ← ml_input.out_video, `in_landmarks` ← ml_gesture.out_landmarks, `in_gestures` ← ml_gesture.out_gestures, `in_hands` ← ml_gesture.out_hands | hand skeleton (bones + joints) over the video, gesture name + score | `view_gesture.frag` |
| `view_classify` | `in_classes` ← ml_classify.out_classes, `in_top` ← ml_classify.out_top | top-5 bars, eased between updates (0.25 s lag) | `view_classify.frag` |
| `view_speech` | `in_transcript`, `in_partial` ← ml_speech, `in_speech` ← ml_speech.out_speech | transcript captions, partial line, pulsing "speaking" dot | `view_speech.frag` |
| `view_sentiment` | `in_sentiment`, `in_label` ← ml_sentiment, `in_text` ← ml_speech.out_final | valence gauge, 600-frame (10 s at 60 fps) history trail, colour wash | `view_sentiment.frag` |
| `view_llm` | `in_llm`, `in_scene`, `in_emoji` ← ml_llm, `in_quote` ← ml_speech.out_final, `in_sentiment` ← ml_sentiment.out_sentiment | six worlds crossfading over 0.8 s, sentiment tint, emoji burst, quote and tok/s | `view_llm.frag` |
| `view_metrics` | `in_metrics` ← ml_server.out_metrics, `in_nodes` ← ml_server.out_nodes | p50 history per node (log scale 1–1000 ms) + table: p50/p95, fps, drops, server CPU/RAM | `view_metrics.frag` |
| `dashboard` | `in_gesture`, `in_classify`, `in_speech`, `in_sentiment`, `in_llm`, `in_metrics` ← each `view_*.out_view` | 3×2 grid, 1280×640 → `out_dashboard` | `dashboard.frag` |

`window_dashboard` (Window COMP) shows `dashboard/null_dashboard`.

## Notes

- **Use the viewers anywhere.** Each `out_view` is an ordinary TOP. Composite it, project it, or
  send it out through NDI or Syphon.
- **Wire by name.** Connector order on a COMP is not the build order. Connect by connector name, or run
  `wire_view('<name>')` from `td/builder/build_viewers.py`.
- **Resolution limit.** The Non-Commercial licence caps textures at 1280×1280. That is why the
  dashboard is 1280×640 and the trails hold at most 1200 samples (CHOP-to-TOP makes one pixel per
  sample).
- **Trail cooking.** Lag and Trail CHOPs inside the viewers end in Null CHOPs set to *Cook Type:
  Always*, so they keep animating when nothing downstream asks for a new frame.
