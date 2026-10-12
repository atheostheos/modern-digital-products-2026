"""Build local test media (git-ignored) so nodes can be tested without a live camera or microphone.

Creates in task-01/ai/media/:
  gestures.mp4 / gestures.json  slideshow of MediaPipe's official gesture sample photos (+ a TD sample
                                image with no hands) and the expected gesture per time segment
  speech.wav / speech.json      scripted sentences synthesised with macOS `say`, with expected
                                transcript and sentiment per sentence

Run: uv run --project server python scripts/make_media.py
"""

import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
MEDIA_DIR = PROJECT_DIR / "media"
SAMPLES_DIR = MEDIA_DIR / "samples"
MEDIAPIPE_BASE = "https://storage.googleapis.com/mediapipe-tasks/"
TD_MAP_DIR = Path("/Applications/TouchDesigner.app/Contents/Resources/tfs/Samples/Map")
MAX_DOWNLOAD_BYTES = 1_000_000
DOWNLOAD_TIMEOUT_S = 30

VIDEO_SIZE = (640, 480)
VIDEO_FPS = 30
HOLD_S = 2.0
GAP_S = 0.5

# (file name, source, expected gesture; None = no hand gesture expected)
GESTURE_SLIDES = [
    ("thumbs_up.jpg", "gesture_recognizer/thumbs_up.jpg", "Thumb_Up"),
    ("victory.jpg", "gesture_recognizer/victory.jpg", "Victory"),
    ("pointing_up.jpg", "gesture_recognizer/pointing_up.jpg", "Pointing_Up"),
    ("thumbs_down.jpg", "gesture_recognizer/thumbs_down.jpg", "Thumb_Down"),
    ("woman_hands.jpg", "hand_landmarker/woman_hands.jpg", None),
    ("Banana.tif", "td:Banana.tif", None),
]

SPEECH_LINES = [
    ("Good morning everyone, welcome to the light installation.", "positive"),
    ("I absolutely love how the colours follow my hands.", "positive"),
    ("The rain outside is awful and my train was late again.", "negative"),
    ("Let's turn the music up and make some noise.", "positive"),
    ("Honestly, this part is boring and far too loud.", "negative"),
    ("Thank you all for coming, this was wonderful.", "positive"),
]
SPEECH_PAUSE_MS = 1500


def fetch(name: str, source: str) -> Path:
    target = SAMPLES_DIR / name
    if target.exists():
        return target
    if source.startswith("td:"):
        shutil.copyfile(TD_MAP_DIR / source.removeprefix("td:"), target)
        return target
    with urllib.request.urlopen(MEDIAPIPE_BASE + source, timeout=DOWNLOAD_TIMEOUT_S) as response:
        if response.headers.get_content_type() != "image/jpeg":
            raise ValueError(f"{source}: unexpected content type {response.headers.get_content_type()}")
        data = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise ValueError(f"{source}: larger than {MAX_DOWNLOAD_BYTES} bytes")
    target.write_bytes(data)
    return target


def letterbox(image: np.ndarray) -> np.ndarray:
    width, height = VIDEO_SIZE
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    top, left = (height - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def open_writer(path: Path) -> cv2.VideoWriter:
    for codec in ("avc1", "mp4v"):  # H.264 via AVFoundation first; MPEG-4 part 2 as fallback
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), VIDEO_FPS, VIDEO_SIZE)
        if writer.isOpened():
            print(f"video codec: {codec}")
            return writer
    raise RuntimeError("no usable video codec")


def make_gesture_video() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    writer, segments, t = open_writer(MEDIA_DIR / "gestures.mp4"), [], 0.0
    blank = np.zeros((VIDEO_SIZE[1], VIDEO_SIZE[0], 3), dtype=np.uint8)
    try:
        for name, source, expected in GESTURE_SLIDES:
            frame = letterbox(cv2.imread(str(fetch(name, source)), cv2.IMREAD_COLOR))
            for _ in range(round(HOLD_S * VIDEO_FPS)):
                writer.write(frame)
            segments.append({"start_s": t, "end_s": t + HOLD_S, "image": name, "expected_gesture": expected})
            t += HOLD_S
            for _ in range(round(GAP_S * VIDEO_FPS)):
                writer.write(blank)
            t += GAP_S
    finally:
        writer.release()
    manifest = {"fps": VIDEO_FPS, "size": VIDEO_SIZE, "duration_s": t, "segments": segments}
    (MEDIA_DIR / "gestures.json").write_text(json.dumps(manifest, indent=2))
    print(f"gestures.mp4: {len(segments)} slides, {t:.1f} s")


def make_speech() -> None:
    text = f" [[slnc {SPEECH_PAUSE_MS}]] ".join(line for line, _ in SPEECH_LINES)
    out = MEDIA_DIR / "speech.wav"
    subprocess.run(["say", "--file-format=WAVE", "--data-format=LEI16@16000", "-o", str(out), text], check=True)
    manifest = {"sample_rate": 16000, "lines": [{"text": t, "sentiment": s} for t, s in SPEECH_LINES]}
    (MEDIA_DIR / "speech.json").write_text(json.dumps(manifest, indent=2))
    print(f"speech.wav: {len(SPEECH_LINES)} lines, {out.stat().st_size // 1024} KiB")


if __name__ == "__main__":
    MEDIA_DIR.mkdir(exist_ok=True)
    make_gesture_video()
    make_speech()
