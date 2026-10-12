# ml_gesture — hand landmarks + gestures (video, MediaPipe)

Sends the input video as small JPEGs to `/ws/gesture`. The server runs the MediaPipe Gesture
Recognizer, which finds up to 2 hands, 21 landmarks per hand, 7 gestures and handedness.

| | |
|---|---|
| Input | `in_video` — any TOP (resized to **Send Size** before sending) |
| Server | `/ws/gesture`, about 11 ms p50 per frame on CPU |

## Parameters

| Page | Parameter | Default | Meaning |
|---|---|---|---|
| Gesture | Send Rate (fps) | 15 | frames sent per second |
| Gesture | Send Size (px) | 256 | longest side of the sent JPEG |
| Server | Host, Port, Active, Reconnect | | see [README](README.md) |

## Outputs

| Output | Type | Content |
|---|---|---|
| `out_landmarks` | CHOP, 21 samples | `h0tx h0ty h0tz h1tx h1ty h1tz`: landmark positions per hand, normalised 0–1, y flipped to TD's bottom-left origin. Sample index = MediaPipe landmark id (0 wrist, 4 thumb tip, 8 index tip, …). |
| `out_gestures` | CHOP, 1 sample | `hands` (0–2), `h0_score`, `h1_score`, and one channel per gesture with the best score across hands: `None Closed_Fist Open_Palm Pointing_Up Thumb_Down Thumb_Up Victory ILoveYou` |
| `out_hands` | table DAT | `hand handedness handedness_score gesture gesture_score`, one row per hand |
| `out_status` | CHOP | see [README](README.md) |

## Wiring

```
ml_input.out_video ─▶ in_video
out_landmarks ─▶ view_gesture.in_landmarks     out_gestures ─▶ view_gesture.in_gestures
out_hands     ─▶ view_gesture.in_hands         (ml_llm reads table_hands through its Gesture parameter)
```

Ideas: drive instancing from `out_landmarks` (Instance CHOP = 21 points), trigger events from a
Logic CHOP on `Thumb_Up > 0.6`, or put `h0tx`/`h0ty` (sample 8) through a Lag CHOP and use it as
a pointer.
