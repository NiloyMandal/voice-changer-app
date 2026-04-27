from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.audio.io import pitch_shift_numpy
from app.config.settings import settings
from app.inference.engine import InferenceEngine


ProfileName = Literal["natural", "robot", "deep", "chipmunk"]


inference_engine: InferenceEngine | None = None
inference_init_error: str | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global inference_engine, inference_init_error
    inference_init_error = None
    try:
        inference_engine = InferenceEngine(
            backend=settings.inference_backend,
            torch_model_path=settings.torch_model_path,
            onnx_model_path=settings.onnx_model_path,
            device=settings.inference_device,
            model_input_layout=settings.model_input_layout,
            normalize_mode=settings.inference_normalize,
            target_dbfs=settings.target_dbfs,
            restore_level=settings.restore_level,
        )
    except Exception as exc:
        inference_engine = None
        inference_init_error = str(exc)
    yield
    inference_engine = None
    inference_init_error = None


app = FastAPI(title="Voice Changer Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _apply_profile(audio: np.ndarray, profile: ProfileName) -> np.ndarray:
    """Apply realtime profile transforms with AI-first inference routing."""
    audio = audio.astype(np.float32, copy=False)

    if profile == "natural":
        return audio

    # Route transformed profiles through profile-specific inference when available.
    if inference_engine is not None and inference_engine.is_ready(profile):
        inferred = inference_engine.infer(audio, profile=profile)
        return np.clip(inferred.astype(np.float32, copy=False), -1.0, 1.0)

    # Fallback DSP path if model is unavailable or failed to initialize.
    if profile == "deep":
        shifted = pitch_shift_numpy(audio, semitones=-4.0)
        return np.clip(shifted, -1.0, 1.0)
    if profile == "chipmunk":
        shifted = pitch_shift_numpy(audio, semitones=4.0)
        return np.clip(shifted, -1.0, 1.0)

    # Fallback if startup initialization was skipped.
    return np.clip(np.round(audio * 8.0) / 8.0, -1.0, 1.0)


@dataclass
class StreamState:
    overlap_size: int = 512
    prev_input_tail: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    prev_output_tail: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))


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

    # Prepend the previous raw tail to provide continuity for profile processing.
    if state.prev_input_tail.size > 0:
        context = np.concatenate((state.prev_input_tail, audio))
    else:
        context = audio

    processed_context = _apply_profile(context, profile)

    # Blend previous processed tail into the new processed head at the boundary.
    if state.prev_output_tail.size > 0:
        overlap_len = min(state.prev_input_tail.size, processed_context.size)
        head = processed_context[:overlap_len]
        rest = processed_context[overlap_len:]
        blended_head = _crossfade_with_hanning(state.prev_output_tail, head)
        processed = np.concatenate((blended_head, rest))
    else:
        processed = processed_context

    # Keep a raw tail for next context window and hold back output tail for next crossfade.
    tail_len = min(state.overlap_size, audio.size, processed.size)
    if tail_len > 0:
        state.prev_input_tail = audio[-tail_len:].copy()
        state.prev_output_tail = processed[-tail_len:].copy()
        processed = processed[:-tail_len]
    else:
        state.prev_input_tail = np.empty(0, dtype=np.float32)
        state.prev_output_tail = np.empty(0, dtype=np.float32)

    out_pcm = np.clip(processed * 32767.0, -32768, 32767).astype(np.int16)
    return out_pcm.tobytes()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/inference")
async def health_inference() -> dict[str, object]:
    status: dict[str, object] = {
        "initialized": inference_engine is not None,
        "backend": settings.inference_backend,
        "device": settings.inference_device,
        "error": inference_init_error,
    }
    if inference_engine is not None:
        status["engine"] = inference_engine.status()
    return status


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
                processed = await asyncio.to_thread(
                    _process_pcm16_chunk,
                    data,
                    current_profile,
                    stream_state,
                )
                if processed:
                    await websocket.send_bytes(processed)
                continue

            if message.get("type") == "websocket.disconnect":
                break

    except WebSocketDisconnect:
        return
