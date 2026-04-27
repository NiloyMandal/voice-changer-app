from __future__ import annotations

from typing import Literal

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware


ProfileName = Literal["natural", "robot", "deep", "chipmunk"]

app = FastAPI(title="Voice Changer Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _apply_profile(audio: np.ndarray, profile: ProfileName) -> np.ndarray:
    """Placeholder profile transforms for demo/testing."""
    if profile == "natural":
        return audio
    if profile == "deep":
        return np.clip(audio * 0.9, -1.0, 1.0)
    if profile == "chipmunk":
        return np.clip(audio * 1.1, -1.0, 1.0)

    # robot: simple hard clipping distortion effect.
    return np.clip(np.round(audio * 8.0) / 8.0, -1.0, 1.0)


def _process_pcm16_chunk(chunk: bytes, profile: ProfileName) -> bytes:
    if not chunk:
        return b""

    pcm = np.frombuffer(chunk, dtype=np.int16)
    if pcm.size == 0:
        return b""

    audio = pcm.astype(np.float32) / 32768.0
    processed = _apply_profile(audio, profile)
    out_pcm = np.clip(processed * 32767.0, -32768, 32767).astype(np.int16)
    return out_pcm.tobytes()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/ws/audio")
async def stream_audio(websocket: WebSocket) -> None:
    """Receive browser PCM16 chunks, process, and return processed PCM16."""
    await websocket.accept()
    current_profile: ProfileName = "natural"

    try:
        while True:
            message = await websocket.receive()

            text = message.get("text")
            data = message.get("bytes")

            if text is not None:
                # Control message example:
                # {"type":"set_profile","profile":"deep"}
                try:
                    import json

                    payload = json.loads(text)
                    if payload.get("type") == "set_profile":
                        profile = payload.get("profile", "natural")
                        if profile in {"natural", "robot", "deep", "chipmunk"}:
                            current_profile = profile
                except Exception:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "Invalid control message",
                        }
                    )
                continue

            if data is not None:
                processed = _process_pcm16_chunk(data, current_profile)
                if processed:
                    await websocket.send_bytes(processed)
                continue

            if message.get("type") == "websocket.disconnect":
                break

    except WebSocketDisconnect:
        return
