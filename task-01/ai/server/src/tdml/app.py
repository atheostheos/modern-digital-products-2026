"""FastAPI app: WebSocket streaming for TouchDesigner, REST API, health and metrics."""

import asyncio
import logging
import time
from collections.abc import Sequence
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from tdml.nodes.base import InputKind
from tdml.protocol import (
    MAX_REST_BODY_BYTES,
    UNKNOWN_SEQ,
    Envelope,
    ProtocolError,
    Timing,
    failure,
    parse_text,
    parse_text_body,
    split_frame,
    success,
    to_jsonable,
)
from tdml.runtime.broadcast import Broadcaster
from tdml.runtime.hub import NodeHub
from tdml.runtime.metrics import process_snapshot
from tdml.runtime.outbox import Outbox
from tdml.runtime.worker import Job, LatestWinsWorker, Result

logger = logging.getLogger(__name__)

METRICS_INTERVAL_S = 0.5
WS_POLICY_VIOLATION = 1008
WS_INTERNAL_ERROR = 1011


def _result_envelope(node: str, result: Result) -> Envelope:
    job = result.job
    if result.error is not None:
        return failure(node, job.seq, result.error, t_send=job.t_send)
    timing = Timing(t_recv=job.t_recv, t_done=result.t_done, inference_ms=round(result.inference_ms, 3))
    return success(node, job.seq, to_jsonable(result.data or {}), timing, t_send=job.t_send)


def _reply_text(node: str, result: Result) -> str:
    """Serialise a result; a non-serialisable result still produces an error reply, never silence."""
    try:
        return _result_envelope(node, result).to_json()
    except (TypeError, ValueError) as exc:
        logger.error("node %s: result for seq %s is not serialisable: %s", node, result.job.seq, exc)
        return failure(
            node, result.job.seq, f"result not serialisable: {exc}", t_send=result.job.t_send
        ).to_json()


def _ws_request(kind: InputKind, message: dict) -> tuple[int, float | None, Any]:
    """Validate one WebSocket message for a node; returns (seq, t_send, payload for the worker)."""
    if kind is InputKind.IMAGE:
        if message.get("bytes") is None:
            raise ProtocolError("this node expects binary JPEG frames")
        frame = split_frame(message["bytes"])
        return frame.seq, frame.t_send, frame.jpeg
    if message.get("text") is None:
        raise ProtocolError("this node expects text JSON messages")
    request = parse_text(message["text"])
    return request.seq, request.t_send, request.text


def _error_response(status: int, node: str, error: str) -> JSONResponse:
    return JSONResponse(failure(node, UNKNOWN_SEQ, error).model_dump(), status_code=status)


async def _read_body(request: Request) -> bytes:
    """Read a REST body, refusing oversized ones before buffering them."""
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > MAX_REST_BODY_BYTES:
        raise ProtocolError(f"body too large: {declared} bytes (max {MAX_REST_BODY_BYTES})")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_REST_BODY_BYTES:
            raise ProtocolError(f"body too large (max {MAX_REST_BODY_BYTES} bytes)")
    return bytes(body)


async def _serve_requests(
    ws: WebSocket, outbox: Outbox, node: str, kind: InputKind, worker: LatestWinsWorker
) -> None:
    """Request node: each client message is a job; frames dropped by latest-wins get no reply."""

    def deliver(result: Result) -> None:  # runs on the worker thread
        outbox.post_threadsafe(_reply_text(node, result))

    while (message := await ws.receive())["type"] != "websocket.disconnect":
        t_recv = time.perf_counter()
        try:
            seq, t_send, payload = _ws_request(kind, message)
        except ProtocolError as exc:
            outbox.post(failure(node, UNKNOWN_SEQ, str(exc)).to_json())
            continue
        worker.submit(Job(seq=seq, payload=payload, t_send=t_send, t_recv=t_recv, on_done=deliver))


async def _serve_stream(ws: WebSocket, outbox: Outbox, node: str, broadcaster: Broadcaster) -> None:
    """Stream node: the client only listens; anything it sends is answered with an error."""
    broadcaster.subscribe(outbox)
    try:
        while (await ws.receive())["type"] != "websocket.disconnect":
            outbox.post(failure(node, UNKNOWN_SEQ, "stream node accepts no input").to_json())
    finally:
        broadcaster.unsubscribe(outbox)


def create_app(nodes: Sequence[Any], metrics_interval_s: float = METRICS_INTERVAL_S) -> FastAPI:
    hub = NodeHub(nodes)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await hub.startup()
        try:
            yield
        finally:
            await hub.shutdown()

    app = FastAPI(title="tdml", description="Local ML inference server for TouchDesigner", lifespan=lifespan)

    def metrics_payload() -> dict:
        return {"t": time.time(), "nodes": hub.metrics(), "process": process_snapshot()}

    @app.get("/health")
    async def health() -> dict:
        return hub.health()

    @app.get("/metrics")
    async def metrics() -> dict:
        return metrics_payload()

    @app.websocket("/ws/metrics")
    async def metrics_stream(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                await ws.send_json(metrics_payload())
                await asyncio.sleep(metrics_interval_s)
        except (WebSocketDisconnect, RuntimeError):
            return

    @app.post("/api/{node}")
    async def infer_rest(node: str, request: Request) -> JSONResponse:
        kind = hub.kind(node)
        if kind is None:
            return _error_response(404, node, f"unknown node '{node}'")
        if kind is InputKind.STREAM:
            return _error_response(405, node, f"'{node}' is a stream node; subscribe at /ws/{node}")
        if not hub.loaded[node]:
            return _error_response(503, node, f"node '{node}' failed to load")
        t_recv = time.perf_counter()
        try:
            body = await _read_body(request)
        except ProtocolError as exc:
            return _error_response(413, node, str(exc))
        try:
            payload = body if kind is InputKind.IMAGE else parse_text_body(body)
            data, inference_ms = await run_in_threadpool(hub.workers[node].run_now, payload)
            timing = Timing(t_recv=t_recv, t_done=time.perf_counter(), inference_ms=round(inference_ms, 3))
            return JSONResponse(success(node, 0, to_jsonable(data), timing).model_dump())
        except ProtocolError as exc:
            return _error_response(422, node, str(exc))
        except Exception as exc:
            logger.exception("node %s: REST inference failed", node)
            return _error_response(500, node, f"{type(exc).__name__}: {exc}")

    @app.websocket("/ws/{node}")
    async def node_stream(ws: WebSocket, node: str) -> None:
        kind = hub.kind(node)
        if kind is None:
            await ws.close(code=WS_POLICY_VIOLATION)
            return
        if not hub.loaded[node]:
            await ws.close(code=WS_INTERNAL_ERROR, reason=f"node '{node}' failed to load")
            return
        await ws.accept()
        outbox = Outbox(ws.send_text, asyncio.get_running_loop())
        sender = asyncio.create_task(outbox.run())
        try:
            if kind is InputKind.STREAM:
                await _serve_stream(ws, outbox, node, hub.broadcasters[node])
            else:
                await _serve_requests(ws, outbox, node, kind, hub.workers[node])
        finally:
            outbox.close()
            await sender

    return app
