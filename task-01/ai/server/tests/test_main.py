import os

import pytest

from tdml import __main__ as cli
from tdml.nodes.classify import ClassifyNode
from tdml.nodes.gesture import GestureNode
from tdml.nodes.llm import LlmNode
from tdml.nodes.sentiment import SentimentNode
from tdml.nodes.speech import SpeechNode
from tdml.protocol import MAX_FRAME_BYTES


def test_defaults_are_loopback_all_nodes_two_threads():
    args = cli.parse_args([])

    assert (args.host, args.port, args.threads) == ("127.0.0.1", 8000, 2)
    assert args.nodes == list(cli.ALL_NODES)


def test_nodes_option_selects_subset_in_given_order():
    assert cli.parse_args(["--nodes", "sentiment,llm"]).nodes == ["sentiment", "llm"]


@pytest.mark.parametrize(
    "argv",
    [["--nodes", "sentiment,teleport"], ["--threads", "0"], ["--threads", "64"], ["--port", "70000"]],
    ids=["unknown-node", "zero-threads", "too-many-threads", "bad-port"],
)
def test_invalid_options_exit_with_usage_error(argv):
    with pytest.raises(SystemExit):
        cli.parse_args(argv)


def test_build_nodes_creates_requested_node_types_without_loading():
    nodes = cli.build_nodes(cli.parse_args([]))

    assert [type(n) for n in nodes] == [GestureNode, ClassifyNode, SpeechNode, SentimentNode, LlmNode]


def test_configure_runtime_points_caches_at_models_dir_and_caps_threads(monkeypatch, tmp_path):
    for var in ("HF_HOME", "TORCH_HOME", "OMP_NUM_THREADS", "TDML_MODELS_DIR"):
        monkeypatch.delenv(var, raising=False)

    cli.configure_runtime(threads=3, models=tmp_path)

    assert os.environ["HF_HOME"] == str(tmp_path / "hf")
    assert os.environ["TORCH_HOME"] == str(tmp_path / "torch")
    assert os.environ["OMP_NUM_THREADS"] == "3"


def test_main_runs_uvicorn_with_frame_sized_websocket_limit(monkeypatch):
    calls = {}
    monkeypatch.setattr(cli, "configure_runtime", lambda threads, models: None)
    monkeypatch.setattr(cli.uvicorn, "run", lambda app, **kwargs: calls.update(app=app, **kwargs))

    cli.main(["--nodes", "sentiment", "--port", "8123"])

    assert calls["host"] == "127.0.0.1"
    assert calls["port"] == 8123
    assert calls["ws_max_size"] == MAX_FRAME_BYTES
    assert [route.path for route in calls["app"].routes if route.path == "/health"] == ["/health"]


def test_audio_file_option_feeds_speech_from_file(tmp_path):
    from tdml.audio.file_source import FileAudioSource

    clip = tmp_path / "speech.wav"
    clip.write_bytes(b"")
    args = cli.parse_args(["--nodes", "speech", "--audio-file", str(clip)])

    (node,) = cli.build_nodes(args)
    source = node._source_factory(lambda chunk: None)

    assert isinstance(source, FileAudioSource)
    assert args.audio_loop is True


def test_audio_file_must_exist():
    with pytest.raises(SystemExit):
        cli.parse_args(["--audio-file", "/definitely/not/here.wav"])
