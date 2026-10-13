# ml_input — recorded or live media source

Plays the recorded test media or captures the camera and microphone. It sends nothing to the server
itself; it only feeds the client nodes.

| | |
|---|---|
| Inputs | none |
| Outputs | `out_video` (TOP, 640×480), `out_audio` (CHOP, mono, time-sliced) |
| Inside | `moviefilein_recorded` / `videodevicein_live` → `switch_video` → `null_video`; `audiofilein_recorded` / `audiodevicein_live` → `switch_audio` → `null_audio`; `audiodeviceout_monitor` |

## Parameters (Input page)

| Parameter | Default | Meaning |
|---|---|---|
| Live | off | off = media files, on = camera + microphone. While live, file playback pauses; while recorded, the devices are inactive (camera light off). |
| Video File | `media/gestures.mp4` | recorded video (relative to the `.toe`) |
| Audio File | `media/speech.wav` | recorded audio |
| Monitor | off | play the current audio on the speakers |

`button_live` and `button_monitor` in `/project1` are Button COMPs. Their value is bound
(*Bind* mode) to **Live** and **Monitor**, so clicking a button in the network editor flips the
parameter and the other way round. To add another switch, create a toggle Button COMP and bind its
`value0` to `op('ml_input').par.Live`.

## Wiring

```
ml_input.out_video ─▶ ml_gesture.in_video
                   ─▶ ml_classify.in_video
                   ─▶ view_gesture.in_video   (background for the skeleton)
ml_input.out_audio ─▶ ml_speech.in_audio
```

Make `media/` with `uv run --project server python scripts/make_media.py`. To change the camera or
microphone, set *Device* on `videodevicein_live` or `audiodevicein_live`.
