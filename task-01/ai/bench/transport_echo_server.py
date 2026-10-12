"""Transport benchmark server: TD sends JPEG frames over WebSocket, server decodes and replies.

Frame layout (binary WS message): header `<Id` = (seq: uint32, t_send: float64), then JPEG bytes.
Reply (text WS message): JSON with seq, t_send echoed back, frame size and server-side timings.

Run: uv run --project server python bench/transport_echo_server.py [--port 8765]
"""

import argparse
import io
import json
import struct
import time

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from PIL import Image

HEADER = struct.Struct("<Id")
DEFAULT_PORT = 8765

app = FastAPI()


def decode_frame(payload: bytes) -> dict:
    """Parse header + JPEG, decode to RGB uint8 array, return reply fields."""
    if len(payload) <= HEADER.size:
        raise ValueError(f"frame too short: {len(payload)} bytes")
    seq, t_send = HEADER.unpack_from(payload)
    t0 = time.perf_counter()
    image = np.asarray(Image.open(io.BytesIO(payload[HEADER.size :])).convert("RGB"))
    decode_ms = (time.perf_counter() - t0) * 1000
    return {
        "seq": seq,
        "t_send": t_send,
        "bytes": len(payload) - HEADER.size,
        "w": int(image.shape[1]),
        "h": int(image.shape[0]),
        "decode_ms": round(decode_ms, 3),
    }


@app.websocket("/ws/echo")
async def echo(ws: WebSocket) -> None:
    await ws.accept()
    try:
        while True:
            payload = await ws.receive_bytes()
            t_recv = time.perf_counter()
            try:
                reply = {"ok": True, **decode_frame(payload)}
            except Exception as exc:  # report bad frames to the client instead of dropping the socket
                reply = {"ok": False, "error": str(exc)}
            reply["server_ms"] = round((time.perf_counter() - t_recv) * 1000, 3)
            await ws.send_text(json.dumps(reply))
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
