from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.audio.io import pitch_shift_numpy


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
    """Apply lightweight realtime profile transforms.

    Uses pitch_shift_numpy for profile-driven pitch changes while AI models are not yet wired.
    """
    audio = audio.astype(np.float32, copy=False)

    if profile == "natural":
        return audio
    if profile == "deep":
        shifted = pitch_shift_numpy(audio, semitones=-4.0)
        return np.clip(shifted, -1.0, 1.0)
    if profile == "chipmunk":
        shifted = pitch_shift_numpy(audio, semitones=4.0)
        return np.clip(shifted, -1.0, 1.0)

    # robot: simple hard clipping distortion effect.
    return np.clip(np.round(audio * 8.0) / 8.0, -1.0, 1.0)


@dataclass
class StreamState:
    overlap_size: int = 256
    prev_tail: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))


def _crossfade_with_hanning(prev_tail: np.ndarray, current: np.ndarray) -> np.ndarray:
    overlap = min(prev_tail.size, current.size)
    if overlap <= 0:
        return current

    blended = current.copy()
    if overlap == 1:
        blended[0] = 0.5 * prev_tail[-1] + 0.5 * blended[0]
        return blended

    window = np.hanning(overlap * 2).astype(np.float32)
    fade_out = window[:overlap]
    fade_in = window[overlap:]
    blended[:overlap] = (prev_tail[-overlap:] * fade_out) + (blended[:overlap] * fade_in)
    return blended


def _process_pcm16_chunk(chunk: bytes, profile: ProfileName, state: StreamState) -> bytes:
    if not chunk:
        return b""

    pcm = np.frombuffer(chunk, dtype=np.int16)
    if pcm.size == 0:
        return b""

    audio = pcm.astype(np.float32) / 32768.0

    if state.prev_tail.size > 0:
        context = np.concatenate((state.prev_tail, audio))
        processed_context = _apply_profile(context, profile)
        processed = processed_context[state.prev_tail.size :]
        processed = _crossfade_with_hanning(processed_context[: state.prev_tail.size], processed)
    else:
        processed = _apply_profile(audio, profile)

    tail_len = min(state.overlap_size, audio.size)
    state.prev_tail = audio[-tail_len:].copy() if tail_len > 0 else np.empty(0, dtype=np.float32)

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
    stream_state = StreamState()

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
                processed = _process_pcm16_chunk(data, current_profile, stream_state)
                if processed:
                    await websocket.send_bytes(processed)
                continue

            if message.get("type") == "websocket.disconnect":
                break

    except WebSocketDisconnect:
        return
