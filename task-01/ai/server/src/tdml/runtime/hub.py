"""Owns the nodes: validation, loading, start/stop, health and metrics for request and stream nodes."""

import logging
from collections.abc import Sequence
from typing import Any

from fastapi.concurrency import run_in_threadpool

from tdml.nodes.base import InputKind
from tdml.protocol import decode_image
from tdml.runtime.broadcast import Broadcaster
from tdml.runtime.metrics import NodeMetrics
from tdml.runtime.worker import LatestWinsWorker

logger = logging.getLogger(__name__)

RESERVED_NODE_NAMES = frozenset({"metrics"})  # would collide with /ws/metrics


def _make_worker(node: Any) -> LatestWinsWorker:
    """Image nodes receive JPEG bytes; the worker decodes them, so dropped frames are never decoded."""
    prepare = decode_image if node.input_kind is InputKind.IMAGE else None
    return LatestWinsWorker(node.name, node.infer, NodeMetrics(node.name), prepare=prepare)


class NodeHub:
    def __init__(self, nodes: Sequence[Any]) -> None:
        self.registry = {node.name: node for node in nodes}
        if len(self.registry) != len(nodes):
            raise ValueError("node names must be unique")
        if reserved := RESERVED_NODE_NAMES & self.registry.keys():
            raise ValueError(f"node names are reserved: {sorted(reserved)}")
        self.workers = {
            name: _make_worker(node)
            for name, node in self.registry.items()
            if node.input_kind is not InputKind.STREAM
        }
        self.broadcasters = {
            name: Broadcaster(name, NodeMetrics(name))
            for name, node in self.registry.items()
            if node.input_kind is InputKind.STREAM
        }
        self.loaded: dict[str, bool] = dict.fromkeys(self.registry, False)

    def kind(self, name: str) -> InputKind | None:
        node = self.registry.get(name)
        return node.input_kind if node is not None else None

    async def startup(self) -> None:
        for node in self.registry.values():
            await self._load(node)
        for worker in self.workers.values():
            worker.start()
        for name, broadcaster in self.broadcasters.items():
            if self.loaded[name]:
                await self._start_stream(self.registry[name], broadcaster)

    async def shutdown(self) -> None:
        for name in self.broadcasters:
            await run_in_threadpool(self.registry[name].stop)
        for worker in self.workers.values():
            await run_in_threadpool(worker.stop)  # joins the thread; keeps the event loop free

    async def _load(self, node: Any) -> None:
        try:
            await run_in_threadpool(node.load)
            self.loaded[node.name] = True
            logger.info("node %s loaded", node.name)
        except Exception:
            logger.exception("node %s failed to load; it will answer with errors", node.name)

    async def _start_stream(self, node: Any, broadcaster: Broadcaster) -> None:
        try:
            await run_in_threadpool(node.start, broadcaster.publish)
            logger.info("stream node %s started", node.name)
        except Exception:
            self.loaded[node.name] = False
            logger.exception("stream node %s failed to start (device or permission?)", node.name)

    def metrics(self) -> list[dict]:
        sources = {
            **{n: w.metrics for n, w in self.workers.items()},
            **{n: b.metrics for n, b in self.broadcasters.items()},
        }
        return [sources[name].snapshot() for name in self.registry]

    def health(self) -> dict:
        nodes = {
            name: {"input": str(node.input_kind), "loaded": self.loaded[name]}
            for name, node in self.registry.items()
        }
        return {"ok": all(self.loaded.values()), "nodes": nodes}
