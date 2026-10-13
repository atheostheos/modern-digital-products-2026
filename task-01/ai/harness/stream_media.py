"""TouchDesigner stand-in: capture every input on the client side and stream it to the server.

Recorded (default): media/gestures.mp4 + media/speech.wav.  Live (--live): webcam + microphone.
Video frames -> /ws/gesture (every frame) and /ws/classify (every Nth frame); audio chunks -> /ws/speech
(48 kHz float32, as TD's audio CHOPs deliver it); final transcripts -> POST /api/sentiment; every
--llm-every seconds the current scene -> POST /api/llm. With recorded media, the summary scores the
results against media/gestures.json and media/speech.json.

Start the server first:  uv run --project server tdml-server
Then:                    uv run --project server python harness/stream_media.py --duration 45
"""

import argparse
import json
import queue
import statistics
import struct
import threading
import time
from collections import Counter, defaultdict
from contextlib import ExitStack, suppress
from pathlib import Path

import cv2
import httpx
import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

PROJECT_DIR = Path(__file__).resolve().parents[1]
MEDIA_DIR = PROJECT_DIR / "media"
FRAME_HEADER = struct.Struct("<Id")
AUDIO_HEADER = struct.Struct("<IdI")
AUDIO_RATE = 48000
AUDIO_CHUNK_S = 0.25
AUDIO_LOOP_GAP_S = 2.0
RECV_TIMEOUT_S = 0.5
JPEG_QUALITY = 85


class State:
    """Latest results and statistics shared by the senders, receivers and the LLM loop."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.sent_at: dict[tuple[str, int], float] = {}
        self.video_t: dict[tuple[str, int], float] = {}
        self.rtt_ms: dict[str, list[float]] = defaultdict(list)
        self.inference_ms: dict[str, list[float]] = defaultdict(list)
        self.gesture_by_t: list[tuple[float, str]] = []
        self.latest: dict[str, object] = {"gesture": "None", "image": "", "speech": "", "valence": 0.0}
        self.finals: list[str] = []
        self.directions: list[str] = []


# ---------------------------------------------------------------- video
def open_capture(video: str) -> tuple[cv2.VideoCapture, bool]:
    """'camera' / 'camera:N' opens a webcam (live, no loop); anything else is a video file."""
    if video == "camera" or video.startswith("camera:"):
        return cv2.VideoCapture(int(video.partition(":")[2] or 0)), True
    if not Path(video).is_file():
        raise SystemExit(f"video file not found: {video} (run scripts/make_media.py)")
    return cv2.VideoCapture(video), False


def frames(video: str, fps: float, size: int):
    """Yield (video_time_s, jpeg) at `fps`, long side resized to `size`; files loop forever."""
    while True:
        capture, live = open_capture(video)
        if not capture.isOpened():
            raise SystemExit(f"cannot open video source: {video}")
        source_fps = capture.get(cv2.CAP_PROP_FPS) or fps
        step, index, next_pick = (1.0 if live else source_fps / fps), 0, 0.0
        while (grabbed := capture.read())[0]:
            if index >= next_pick:
                frame = grabbed[1]
                scale = size / max(frame.shape[:2])
                small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ok:
                    yield index / source_fps, buf.tobytes()
                next_pick += step
            index += 1
        capture.release()
        if live:
            return


def stream_video(args, state: State, stop: threading.Event, sockets: dict) -> None:
    for seq, (video_t, jpeg) in enumerate(frames(args.video, args.fps, args.size), start=1):
        if stop.is_set():
            return
        for node, ws in sockets.items():
            if node == "classify" and (seq - 1) % args.classify_every:
                continue
            with state.lock:
                state.sent_at[(node, seq)] = time.perf_counter()
                state.video_t[(node, seq)] = video_t
            ws.send(FRAME_HEADER.pack(seq, time.perf_counter()) + jpeg)
        time.sleep(1 / args.fps)


def receive_frames(ws, node: str, state: State, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            reply = json.loads(ws.recv(timeout=RECV_TIMEOUT_S))
        except TimeoutError:
            continue
        except ConnectionClosed:
            return
        now = time.perf_counter()
        with state.lock:
            sent = state.sent_at.pop((node, reply["seq"]), None)
            video_t = state.video_t.pop((node, reply["seq"]), None)
            if sent is not None:
                state.rtt_ms[node].append((now - sent) * 1000)
            if not reply["ok"]:
                print(f"  {node} error: {reply['error']}")
                continue
            state.inference_ms[node].append(reply["timing"]["inference_ms"])
            data = reply["data"]
            if node == "gesture":
                label = data["hands"][0]["gesture"] if data["hands"] else "None"
                state.latest["gesture"] = label
                if video_t is not None:
                    state.gesture_by_t.append((video_t, label))
            else:
                state.latest["image"] = data["label"]


# ---------------------------------------------------------------- audio
def audio_chunks(source: str, stop: threading.Event):
    """Yield 0.25 s float32 chunks at 48 kHz in real time from a file (looped) or 'mic'."""
    step = round(AUDIO_RATE * AUDIO_CHUNK_S)
    if source == "mic":
        import sounddevice as sd

        captured: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        with sd.InputStream(
            samplerate=AUDIO_RATE,
            channels=1,
            dtype="float32",
            blocksize=step,
            callback=lambda data, frames, t, status: captured.put_nowait(data[:, 0].copy()),
        ):
            while not stop.is_set():
                try:
                    yield captured.get(timeout=RECV_TIMEOUT_S)
                except queue.Empty:
                    continue
        return
    from faster_whisper import decode_audio

    if not Path(source).is_file():
        raise SystemExit(f"audio file not found: {source} (run scripts/make_media.py)")
    samples = decode_audio(source, sampling_rate=AUDIO_RATE)
    samples = np.concatenate([samples, np.zeros(round(AUDIO_RATE * AUDIO_LOOP_GAP_S), dtype=np.float32)])
    next_t = time.perf_counter()
    while not stop.is_set():
        for start in range(0, len(samples) - step + 1, step):
            if stop.is_set():
                return
            yield samples[start : start + step]
            next_t += AUDIO_CHUNK_S
            time.sleep(max(0.0, next_t - time.perf_counter()))


def stream_audio(args, stop: threading.Event, ws) -> None:
    for seq, chunk in enumerate(audio_chunks(args.audio, stop), start=1):
        ws.send(AUDIO_HEADER.pack(seq, time.perf_counter(), AUDIO_RATE) + chunk.astype("<f4").tobytes())


def receive_speech(ws, state: State, stop: threading.Event, http: httpx.Client) -> None:
    while not stop.is_set():
        try:
            reply = json.loads(ws.recv(timeout=RECV_TIMEOUT_S))
        except TimeoutError:
            continue
        except ConnectionClosed:
            return
        if not reply["ok"]:
            print(f"  speech error: {reply['error']}")
            continue
        event = reply["data"]
        if event["type"] != "final" or not event["text"]:
            continue
        latency_ms = (time.perf_counter() - reply["t_send"]) * 1000
        sentiment = http.post("/api/sentiment", json={"text": event["text"]}).json()["data"]
        with state.lock:
            state.finals.append(event["text"])
            state.rtt_ms["speech"].append(latency_ms)
            state.inference_ms["speech"].append(reply["timing"]["inference_ms"])
            state.latest.update(speech=event["text"], valence=sentiment["valence"])
        print(
            f"  speech: {event['text']!r} -> {sentiment['label']} ({sentiment['valence']:+.2f}), "
            f"{latency_ms:.0f} ms after the end-of-speech chunk"
        )


# ---------------------------------------------------------------- llm + summary
def narrate(args, state: State, stop: threading.Event, http: httpx.Client) -> None:
    while not stop.wait(args.llm_every):
        with state.lock:
            scene = json.dumps(state.latest)
        reply = http.post("/api/llm", json={"text": scene}).json()
        if not reply["ok"]:
            print(f"  llm error: {reply['error']}")
        else:
            data, ms = reply["data"], reply["timing"]["inference_ms"]
            with state.lock:
                state.directions.append(data["scene"])
            print(f"  llm: {data['scene']} {data['emoji']}  ({ms:.0f} ms) for {scene}")


def gesture_accuracy(state: State, manifest: dict) -> list[str]:
    lines, correct = [], 0
    for seg in manifest["segments"]:
        labels = [g for t, g in state.gesture_by_t if seg["start_s"] + 0.3 <= t < seg["end_s"]]
        majority = Counter(labels).most_common(1)[0][0] if labels else "-"
        expected = seg["expected_gesture"] or "None"
        correct += majority == expected
        lines.append(f"    {seg['image']:16} expected {expected:12} got {majority:12} ({len(labels)} frames)")
    return [*lines, f"    segment accuracy: {correct}/{len(manifest['segments'])}"]


def summary(args, state: State) -> None:
    print("\n=== summary ===  (speech round trip = end-of-speech chunk sent -> final transcript received)")
    for node in sorted(state.rtt_ms):
        rtt, inf = sorted(state.rtt_ms[node]), state.inference_ms[node]
        print(
            f"  {node:9} replies={len(rtt):4}  inference p50={statistics.median(inf):7.1f} ms  "
            f"round trip p50={statistics.median(rtt):7.1f} p95={rtt[int(0.95 * (len(rtt) - 1))]:7.1f} ms"
        )
    if args.video == str(MEDIA_DIR / "gestures.mp4") and state.gesture_by_t:
        manifest = json.loads((MEDIA_DIR / "gestures.json").read_text())
        print("  gestures vs manifest:", *gesture_accuracy(state, manifest), sep="\n")
    if args.audio == str(MEDIA_DIR / "speech.wav"):
        expected = [line["text"] for line in json.loads((MEDIA_DIR / "speech.json").read_text())["lines"]]
        print(f"  transcripts: {len(state.finals)} finals for {len(expected)} scripted lines")
    print(f"  llm directions: {len(state.directions)} {dict(Counter(state.directions))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--live", action="store_true", help="webcam + microphone instead of recorded media")
    parser.add_argument(
        "--video", help="video file, or camera / camera:N (default: recorded clip, or camera if --live)"
    )
    parser.add_argument("--audio", help="audio file, or mic (default: recorded speech, or mic if --live)")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--classify-every", type=int, default=15, help="send every Nth frame to classify")
    parser.add_argument("--llm-every", type=float, default=8.0, help="LLM request interval, s (0 = off)")
    parser.add_argument("--duration", type=float, default=40.0)
    args = parser.parse_args()
    args.video = args.video or ("camera" if args.live else str(MEDIA_DIR / "gestures.mp4"))
    args.audio = args.audio or ("mic" if args.live else str(MEDIA_DIR / "speech.wav"))
    return args


def main() -> None:
    args = parse_args()
    state, stop = State(), threading.Event()
    ws_base = args.server.replace("http", "ws", 1)
    with ExitStack() as stack:
        frame_sockets = {
            n: stack.enter_context(connect(f"{ws_base}/ws/{n}", max_size=None))
            for n in ("gesture", "classify")
        }
        speech_ws = stack.enter_context(connect(f"{ws_base}/ws/speech", max_size=None))
        http = stack.enter_context(httpx.Client(base_url=args.server, timeout=30))
        jobs = [(receive_frames, (ws, n, state, stop)) for n, ws in frame_sockets.items()]
        jobs += [
            (stream_video, (args, state, stop, frame_sockets)),
            (stream_audio, (args, stop, speech_ws)),
            (receive_speech, (speech_ws, state, stop, http)),
        ]
        if args.llm_every > 0:
            jobs.append((narrate, (args, state, stop, http)))
        threads = [threading.Thread(target=fn, args=a, daemon=True) for fn, a in jobs]
        for thread in threads:
            thread.start()
        with suppress(KeyboardInterrupt):
            stop.wait(args.duration)
        stop.set()
        for thread in threads:
            thread.join(timeout=5)
    summary(args, state)


if __name__ == "__main__":
    main()
