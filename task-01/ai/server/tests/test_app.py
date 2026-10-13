import json

import pytest
from fastapi.testclient import TestClient

from tdml.app import create_app
from tdml.nodes.base import InputKind
from tests.conftest import make_frame, make_jpeg, wait_until


@pytest.fixture
def client(fake_nodes):
    with TestClient(create_app(fake_nodes, metrics_interval_s=0.05)) as test_client:
        yield test_client


class TestHealthAndMetrics:
    def test_health_lists_loaded_nodes_with_input_kind(self, client, fake_nodes):
        body = client.get("/health").json()

        assert body["ok"] is True
        assert body["nodes"]["echo_text"] == {"input": "text", "loaded": True}
        assert body["nodes"]["shape_image"] == {"input": "image", "loaded": True}

    def test_metrics_has_node_and_process_sections(self, client):
        body = client.get("/metrics").json()

        assert {n["node"] for n in body["nodes"]} == {"echo_text", "shape_image", "failing"}
        assert "rss_mb" in body["process"]

    def test_metrics_websocket_streams_snapshots(self, client):
        with client.websocket_connect("/ws/metrics") as ws:
            first = ws.receive_json()
            second = ws.receive_json()

        assert "nodes" in first and "process" in first
        assert second["t"] >= first["t"]


class TestRest:
    def test_text_node_returns_success_envelope(self, client):
        response = client.post("/api/echo_text", json={"text": "hello"})

        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        assert body["data"] == {"upper": "HELLO"}
        assert body["timing"]["inference_ms"] >= 0

    def test_image_node_accepts_raw_jpeg_body(self, client):
        response = client.post(
            "/api/shape_image", content=make_jpeg(20, 10), headers={"content-type": "image/jpeg"}
        )

        assert response.status_code == 200
        assert response.json()["data"]["w"] == 20

    def test_unknown_node_is_404(self, client):
        assert client.post("/api/nope", json={"text": "x"}).status_code == 404

    def test_invalid_text_body_is_422(self, client):
        response = client.post("/api/echo_text", json={"text": ""})

        assert response.status_code == 422
        assert response.json()["ok"] is False

    def test_invalid_image_body_is_422(self, client):
        response = client.post("/api/shape_image", content=b"garbage", headers={"content-type": "image/jpeg"})

        assert response.status_code == 422
        assert "not a valid image" in response.json()["error"]

    def test_inference_failure_is_500_with_error_envelope(self, client):
        response = client.post("/api/failing", json={"text": "x"})

        assert response.status_code == 500
        assert response.json()["ok"] is False
        assert "model exploded" in response.json()["error"]


class TestNodeWebSocket:
    def test_text_round_trip_echoes_seq_and_t_send(self, client):
        with client.websocket_connect("/ws/echo_text") as ws:
            ws.send_text(json.dumps({"seq": 11, "t_send": 2.5, "text": "abc"}))
            reply = ws.receive_json()

        assert reply["ok"] is True
        assert (reply["seq"], reply["t_send"], reply["data"]) == (11, 2.5, {"upper": "ABC"})

    def test_image_round_trip_from_binary_frame(self, client):
        with client.websocket_connect("/ws/shape_image") as ws:
            ws.send_bytes(make_frame(seq=4, t_send=9.0, jpeg=make_jpeg(16, 12, (200, 0, 0))))
            reply = ws.receive_json()

        assert reply["ok"] is True
        assert reply["seq"] == 4
        assert (reply["data"]["w"], reply["data"]["h"]) == (16, 12)

    def test_bad_frame_gets_error_reply_and_socket_stays_open(self, client):
        with client.websocket_connect("/ws/shape_image") as ws:
            ws.send_bytes(b"\x01")
            error = ws.receive_json()
            ws.send_bytes(make_frame(seq=2))
            ok = ws.receive_json()

        assert error["ok"] is False and "too short" in error["error"]
        assert ok["ok"] is True and ok["seq"] == 2

    def test_wrong_message_type_for_node_is_reported(self, client):
        with client.websocket_connect("/ws/shape_image") as ws:
            ws.send_text("hello")
            reply = ws.receive_json()

        assert reply["ok"] is False
        assert "expects binary" in reply["error"]

    def test_unknown_node_websocket_is_closed(self, client):
        from starlette.websockets import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws/nope") as ws:
            ws.receive_json()


class BrokenLoadNode:
    name = "broken"
    input_kind = InputKind.TEXT

    def load(self) -> None:
        raise OSError("weights missing")

    def infer(self, text: str) -> dict:
        return {}


class TestFailurePaths:
    def test_node_that_fails_to_load_is_reported_and_returns_503(self):
        with TestClient(create_app([BrokenLoadNode()])) as client:
            health = client.get("/health").json()
            response = client.post("/api/broken", json={"text": "x"})

        assert health["ok"] is False
        assert health["nodes"]["broken"]["loaded"] is False
        assert response.status_code == 503

    def test_duplicate_node_names_are_rejected(self, fake_nodes):
        with pytest.raises(ValueError, match="unique"):
            create_app([fake_nodes[0], fake_nodes[0]])

    def test_inference_error_over_websocket_returns_error_envelope(self, client):
        with client.websocket_connect("/ws/failing") as ws:
            ws.send_text(json.dumps({"seq": 3, "text": "x"}))
            reply = ws.receive_json()

        assert reply["ok"] is False
        assert reply["seq"] == 3
        assert "model exploded" in reply["error"]

    def test_text_node_rejects_binary_messages(self, client):
        with client.websocket_connect("/ws/echo_text") as ws:
            ws.send_bytes(b"\x00\x01")
            reply = ws.receive_json()

        assert reply["ok"] is False
        assert "expects text" in reply["error"]


class NumpyNode:
    """Returns numpy types and NaN, like real model outputs."""

    name = "numpy_node"
    input_kind = InputKind.TEXT

    def load(self) -> None:
        return None

    def infer(self, text: str) -> dict:
        import numpy as np

        return {"score": np.float32(0.25), "vec": np.array([1, 2]), "nan": float("nan")}


class UnserialisableNode:
    name = "unserialisable"
    input_kind = InputKind.TEXT

    def load(self) -> None:
        return None

    def infer(self, text: str) -> dict:
        return {"model": object()}


class TestRobustness:
    @pytest.fixture
    def robust_client(self):
        with TestClient(create_app([NumpyNode(), UnserialisableNode(), BrokenLoadNode()])) as test_client:
            yield test_client

    def test_numpy_results_are_serialised_over_websocket(self, robust_client):
        with robust_client.websocket_connect("/ws/numpy_node") as ws:
            ws.send_text(json.dumps({"seq": 1, "text": "x"}))
            reply = ws.receive_json()

        assert reply["ok"] is True
        assert reply["data"] == {"score": 0.25, "vec": [1, 2], "nan": None}

    def test_numpy_results_are_serialised_over_rest(self, robust_client):
        response = robust_client.post("/api/numpy_node", json={"text": "x"})

        assert response.status_code == 200
        assert response.json()["data"]["vec"] == [1, 2]

    def test_unserialisable_result_gives_error_reply_not_silence(self, robust_client):
        with robust_client.websocket_connect("/ws/unserialisable") as ws:
            ws.send_text(json.dumps({"seq": 5, "text": "x"}))
            reply = ws.receive_json()

        assert reply["ok"] is False
        assert reply["seq"] == 5
        assert "serialis" in reply["error"]

    def test_unserialisable_result_over_rest_is_500_envelope(self, robust_client):
        response = robust_client.post("/api/unserialisable", json={"text": "x"})

        assert response.status_code == 500
        assert response.json()["ok"] is False

    def test_websocket_to_unloaded_node_is_closed(self, robust_client):
        from starlette.websockets import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect), robust_client.websocket_connect("/ws/broken") as ws:
            ws.receive_json()

    def test_oversized_rest_body_is_413(self, robust_client):
        from tdml.protocol import MAX_FRAME_BYTES

        response = robust_client.post(
            "/api/numpy_node",
            content=b"x" * (MAX_FRAME_BYTES + 1),
            headers={"content-type": "application/json"},
        )

        assert response.status_code == 413

    def test_reserved_node_name_is_rejected(self):
        class MetricsNamed(NumpyNode):
            name = "metrics"

        with pytest.raises(ValueError, match="reserved"):
            create_app([MetricsNamed()])


class FakeAudioSession:
    def __init__(self, emit) -> None:
        self.emit = emit
        self.closed = False

    def feed(self, chunk) -> bool:
        if chunk.seq == 99:
            self.emit(chunk.seq, chunk.t_send, {"type": "error", "error": "boom"}, None)
        else:
            self.emit(
                chunk.seq,
                chunk.t_send,
                {"type": "echo", "n": len(chunk.samples), "rate": chunk.sample_rate},
                2.0,
            )
        return chunk.seq != 7  # seq 7 simulates a full queue

    def close(self) -> None:
        self.closed = True


class FakeAudioNode:
    name = "listener"
    input_kind = InputKind.AUDIO

    def __init__(self) -> None:
        self.sessions: list[FakeAudioSession] = []

    def load(self) -> None:
        return None

    def open_session(self, emit) -> FakeAudioSession:
        session = FakeAudioSession(emit)
        self.sessions.append(session)
        return session


class TestAudioNodes:
    @pytest.fixture
    def audio_node(self) -> FakeAudioNode:
        return FakeAudioNode()

    @pytest.fixture
    def audio_client(self, audio_node):
        with TestClient(create_app([audio_node, NumpyNode()])) as test_client:
            yield test_client

    def test_audio_frame_events_come_back_on_the_same_connection(self, audio_client):
        import numpy as np

        from tests.conftest import make_audio_frame

        with audio_client.websocket_connect("/ws/listener") as ws:
            ws.send_bytes(make_audio_frame(seq=3, samples=np.zeros(480), sample_rate=48000, t_send=1.25))
            reply = ws.receive_json()

        assert reply["ok"] is True
        assert (reply["seq"], reply["t_send"]) == (3, 1.25)
        assert reply["data"] == {"type": "echo", "n": 480, "rate": 48000}
        assert reply["timing"]["inference_ms"] == 2.0

    def test_error_event_becomes_error_envelope(self, audio_client):
        import numpy as np

        from tests.conftest import make_audio_frame

        with audio_client.websocket_connect("/ws/listener") as ws:
            ws.send_bytes(make_audio_frame(seq=99, samples=np.zeros(16)))
            reply = ws.receive_json()

        assert reply["ok"] is False
        assert (reply["seq"], reply["error"]) == (99, "boom")

    def test_invalid_audio_and_text_messages_get_error_replies(self, audio_client):
        with audio_client.websocket_connect("/ws/listener") as ws:
            ws.send_bytes(b"\x00\x01")
            bad_frame = ws.receive_json()
            ws.send_text("hello")
            text = ws.receive_json()

        assert bad_frame["ok"] is False and "too short" in bad_frame["error"]
        assert text["ok"] is False and "expects binary audio" in text["error"]

    def test_session_is_closed_when_client_disconnects(self, audio_client, audio_node):
        with audio_client.websocket_connect("/ws/listener"):
            pass

        assert wait_until(lambda: audio_node.sessions and audio_node.sessions[0].closed)

    def test_metrics_count_requests_drops_and_inferences(self, audio_client):
        import numpy as np

        from tests.conftest import make_audio_frame

        with audio_client.websocket_connect("/ws/listener") as ws:
            for seq in (1, 7):
                ws.send_bytes(make_audio_frame(seq=seq, samples=np.zeros(16)))
            replies = [ws.receive_json() for _ in range(3)]  # 2 events + 1 drop notice, order not guaranteed
        node = {n["node"]: n for n in audio_client.get("/metrics").json()["nodes"]}["listener"]

        assert (node["requests"], node["dropped"], node["processed"]) == (2, 1, 2)
        (dropped,) = [r for r in replies if not r["ok"]]
        assert dropped["seq"] == 7 and "dropped" in dropped["error"]

    def test_protocol_error_reply_uses_seq_from_readable_header(self, audio_client):
        import numpy as np

        from tests.conftest import make_audio_frame

        with audio_client.websocket_connect("/ws/listener") as ws:
            ws.send_bytes(make_audio_frame(seq=31, samples=np.zeros(16), sample_rate=1000))
            reply = ws.receive_json()

        assert (reply["ok"], reply["seq"]) == (False, 31)

    def test_health_reports_audio_input_kind(self, audio_client):
        assert audio_client.get("/health").json()["nodes"]["listener"] == {"input": "audio", "loaded": True}

    def test_rest_on_audio_node_is_405(self, audio_client):
        assert audio_client.post("/api/listener", content=b"x").status_code == 405
