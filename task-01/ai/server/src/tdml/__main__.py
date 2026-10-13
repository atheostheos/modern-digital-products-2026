"""`tdml-server` entry point: parse options, cap resources, build nodes, serve on localhost.

The server opens no camera, microphone or media file: clients (TouchDesigner) send all input.
"""

import argparse
import logging
import os
from pathlib import Path

import uvicorn

from tdml.app import create_app
from tdml.config import models_dir
from tdml.nodes.classify import ClassifyNode
from tdml.nodes.gesture import GestureNode
from tdml.nodes.llm import DEFAULT_HOST as DEFAULT_OLLAMA_HOST
from tdml.nodes.llm import DEFAULT_MODEL as DEFAULT_LLM_MODEL
from tdml.nodes.llm import LlmNode
from tdml.nodes.sentiment import SentimentNode
from tdml.nodes.speech import DEFAULT_MODEL_SIZE, SpeechNode
from tdml.protocol import MAX_FRAME_BYTES

logger = logging.getLogger("tdml")

ALL_NODES = ("gesture", "classify", "speech", "sentiment", "llm")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_THREADS = 2
MAX_THREADS = 8
MAX_PORT = 65535
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _node_list(value: str) -> list[str]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = sorted(set(names) - set(ALL_NODES))
    if unknown or not names:
        raise argparse.ArgumentTypeError(f"unknown nodes {unknown}; choose from {', '.join(ALL_NODES)}")
    return names


def _bounded_int(low: int, high: int):
    def parse(value: str) -> int:
        number = int(value)
        if not low <= number <= high:
            raise argparse.ArgumentTypeError(f"must be between {low} and {high}, got {number}")
        return number

    return parse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="tdml-server", description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST, help="bind address (keep loopback unless needed)")
    parser.add_argument("--port", type=_bounded_int(1, MAX_PORT), default=DEFAULT_PORT)
    parser.add_argument("--nodes", type=_node_list, default=list(ALL_NODES), help="comma-separated subset")
    parser.add_argument("--threads", type=_bounded_int(1, MAX_THREADS), default=DEFAULT_THREADS)
    parser.add_argument("--whisper-model", default=DEFAULT_MODEL_SIZE, help="e.g. tiny.en, base.en")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    return parser.parse_args(argv)


def build_nodes(args: argparse.Namespace) -> list:
    factories = {
        "gesture": lambda: GestureNode(),
        "classify": lambda: ClassifyNode(),
        "speech": lambda: SpeechNode(model_size=args.whisper_model, threads=args.threads),
        "sentiment": lambda: SentimentNode(),
        "llm": lambda: LlmNode(model=args.llm_model, host=args.ollama_host),
    }
    return [factories[name]() for name in args.nodes]


def configure_runtime(threads: int, models: Path) -> None:
    """Point model caches at the project models dir and cap math-library threads (before torch loads)."""
    os.environ.setdefault("HF_HOME", str(models / "hf"))
    os.environ.setdefault("TORCH_HOME", str(models / "torch"))
    os.environ["OMP_NUM_THREADS"] = str(threads)
    import torch

    torch.set_num_threads(threads)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if args.host not in LOOPBACK_HOSTS:
        logger.warning(
            "binding to %s exposes the server beyond this machine; it has no authentication", args.host
        )
    configure_runtime(args.threads, models_dir())
    app = create_app(build_nodes(args))
    uvicorn.run(app, host=args.host, port=args.port, ws_max_size=MAX_FRAME_BYTES, log_level=args.log_level)


if __name__ == "__main__":
    main()
