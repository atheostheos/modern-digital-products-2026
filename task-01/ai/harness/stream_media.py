"""Run the full node chain against recorded media, without TouchDesigner, camera or microphone.

Video frames -> /ws/gesture (every frame) and /ws/classify (every Nth frame); /ws/speech events ->
final transcripts -> POST /api/sentiment; every --llm-every seconds the current scene -> POST /api/llm.
At the end it scores gestures against media/gestures.json and transcripts against media/speech.json.

Start the server first:  uv run --project server tdml-server --audio-file media/speech.wav
Then:                    uv run --project server python harness/stream_media.py --duration 45
"""

import argparse
import json
import statistics
import struct
import threading
import time
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

import cv2
import httpx
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

PROJECT_DIR = Path(__file__).resolve().parents[1]
FRAME_HEADER = struct.Struct("<Id")
RECV_TIMEOUT_S = 0.5
JPEG_QUALITY = 85


class State:
    """Latest results and statistics, shared by the sender, receivers and the LLM loop."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.sent_at: dict[tuple[str, int], float] = {}
        self.video_t: dict[tuple[str, int], float] = {}
        self.rtt_ms: dict[str, list[float]] = defaultdict(list)
        self.inference_ms: dict[str, list[float]] = defaultdict(list)
        self.gesture_by_t: list[tuple[float, str]] = []
        self.latest: dict[str, object] = {"gesture": "None", "image": "", "speech": "", "valence": 0.0}
        self.finals: list[str] = []
        self.captions: list[str] = []


def frames(video: Path, fps: float, size: int, loop: bool):
    """Yield (video_time_s, jpeg) at `fps`, resized so the long side is `size` pixels."""
    while True:
        capture = cv2.VideoCapture(str(video))
        step = capture.get(cv2.CAP_PROP_FPS) / fps
        index, next_pick = 0, 0.0
        while (ok_frame := capture.read())[0]:
            frame = ok_frame[1]
            if index >= next_pick:
                scale = size / max(frame.shape[:2])
                small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ok:
                    yield index / capture.get(cv2.CAP_PROP_FPS), buf.tobytes()
                next_pick += step
            index += 1
        capture.release()
        if not loop:
            return


def receive(ws, node: str, state: State, stop: threading.Event) -> None:
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
            elif node == "classify":
                state.latest["image"] = data["label"]


def stream_video(args, state: State, stop: threading.Event, sockets: dict) -> None:
    seq = 0
    for frame_no, (video_t, jpeg) in enumerate(frames(args.video, args.fps, args.size, loop=True)):
        if stop.is_set():
            return
        seq += 1
        for node, ws in sockets.items():
            if node == "classify" and frame_no % args.classify_every:
                continue
            with state.lock:
                state.sent_at[(node, seq)] = time.perf_counter()
                state.video_t[(node, seq)] = video_t
            ws.send(FRAME_HEADER.pack(seq, time.perf_counter()) + jpeg)
        time.sleep(1 / args.fps)


def listen_speech(args, state: State, stop: threading.Event, http: httpx.Client) -> None:
    with connect(args.server.replace("http", "ws", 1) + "/ws/speech") as ws:
        while not stop.is_set():
            try:
                event = json.loads(ws.recv(timeout=RECV_TIMEOUT_S))["data"]
            except TimeoutError:
                continue
            if event["type"] != "final" or not event["text"]:
                continue
            sentiment = http.post("/api/sentiment", json={"text": event["text"]}).json()["data"]
            with state.lock:
                state.finals.append(event["text"])
                state.latest.update(speech=event["text"], valence=sentiment["valence"])
            print(f"  speech: {event['text']!r} -> {sentiment['label']} ({sentiment['valence']:+.2f})")


def narrate(args, state: State, stop: threading.Event, http: httpx.Client) -> None:
    while not stop.wait(args.llm_every):
        with state.lock:
            scene = json.dumps(state.latest)
        reply = http.post("/api/llm", json={"text": scene}).json()
        if reply["ok"]:
            with state.lock:
                state.captions.append(reply["data"]["caption"])
            print(f"  llm: {reply['data']['caption']!r}  ({reply['timing']['inference_ms']:.0f} ms) for {scene}")


def gesture_accuracy(state: State, manifest: dict) -> list[str]:
    lines, correct = [], 0
    for seg in manifest["segments"]:
        labels = [g for t, g in state.gesture_by_t if seg["start_s"] + 0.3 <= t < seg["end_s"]]
        majority = Counter(labels).most_common(1)[0][0] if labels else "-"
        expected = seg["expected_gesture"] or "None"
        correct += majority == expected
        lines.append(f"    {seg['image']:16} expected {expected:12} got {majority:12} ({len(labels)} frames)")
    return [*lines, f"    segment accuracy: {correct}/{len(manifest['segments'])}"]


def summary(state: State) -> None:
    print("\n=== summary ===")
    for node in sorted(state.rtt_ms):
        rtt, inf = sorted(state.rtt_ms[node]), state.inference_ms[node]
        print(
            f"  {node:9} replies={len(rtt):4}  inference p50={statistics.median(inf):6.1f} ms  "
            f"round trip p50={statistics.median(rtt):6.1f} p95={rtt[int(0.95 * (len(rtt) - 1))]:6.1f} ms"
        )
    gestures_json = PROJECT_DIR / "media" / "gestures.json"
    if gestures_json.exists() and state.gesture_by_t:
        print("  gestures vs manifest:", *gesture_accuracy(state, json.loads(gestures_json.read_text())), sep="\n")
    speech_json = PROJECT_DIR / "media" / "speech.json"
    if speech_json.exists():
        expected = [line["text"] for line in json.loads(speech_json.read_text())["lines"]]
        print(f"  transcripts: {len(state.finals)} finals for {len(expected)} scripted lines")
    print(f"  captions: {len(state.captions)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--video", type=Path, default=PROJECT_DIR / "media" / "gestures.mp4")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--classify-every", type=int, default=15, help="send every Nth frame to classify")
    parser.add_argument("--llm-every", type=float, default=8.0, help="seconds between captions (0 = off)")
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument("--no-speech", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state, stop = State(), threading.Event()
    ws_base = args.server.replace("http", "ws", 1)
    with ExitStack() as stack:
        sockets = {
            node: stack.enter_context(connect(f"{ws_base}/ws/{node}", max_size=None))
            for node in ("gesture", "classify")
        }
        http = stack.enter_context(httpx.Client(base_url=args.server, timeout=30))
        threads = [
            threading.Thread(target=receive, args=(ws, n, state, stop), daemon=True) for n, ws in sockets.items()
        ]
        threads.append(threading.Thread(target=stream_video, args=(args, state, stop, sockets), daemon=True))
        if not args.no_speech:
            threads.append(threading.Thread(target=listen_speech, args=(args, state, stop, http), daemon=True))
        if args.llm_every > 0:
            threads.append(threading.Thread(target=narrate, args=(args, state, stop, http), daemon=True))
        for thread in threads:
            thread.start()
        try:
            stop.wait(args.duration)
        except KeyboardInterrupt:
            pass
        stop.set()
        for thread in threads:
            thread.join(timeout=5)
    summary(state)


if __name__ == "__main__":
    main()
