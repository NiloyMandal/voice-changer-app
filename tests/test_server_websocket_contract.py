import numpy as np
from fastapi.testclient import TestClient

from app.api.server import app


def _pcm16_bytes(size: int) -> bytes:
    x = np.zeros(size, dtype=np.float32)
    return (x * 32767).astype(np.int16).tobytes()


def test_websocket_rejects_invalid_control_message() -> None:
    with TestClient(app) as client:
        with client.websocket_connect("/ws/audio") as ws:
            ws.send_text("{not-json")
            payload = ws.receive_json()
            assert payload["type"] == "error"


def test_websocket_audio_roundtrip_chunk_sizes() -> None:
    with TestClient(app) as client:
        with client.websocket_connect("/ws/audio") as ws:
            chunk = _pcm16_bytes(1024)
            ws.send_bytes(chunk)
            first = ws.receive_bytes()
            assert len(first) == (1024 - 512) * 2

            ws.send_bytes(chunk)
            second = ws.receive_bytes()
            assert len(second) == 1024 * 2
