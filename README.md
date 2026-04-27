# Voice Changer App

Low-latency live voice changer prototype with:

- Python backend audio pipeline and inference abstraction
- FastAPI WebSocket endpoint for browser audio streaming
- React + Vite frontend with Start/Stop and Voice Profile control
- AudioWorklet capture, jitter buffering, and overlap-add backend smoothing

## Current Project Structure

```text
voice-changer-app/
  .github/workflows/ci.yml
  frontend/
    src/
      App.tsx
      components/VoiceChangerControlPanel.tsx
      hooks/useVoiceStream.ts
    public/worklets/pcm-capture-worklet.js
    package.json
  src/app/
    api/server.py
    audio/io.py
    config/settings.py
    inference/engine.py
    pipeline/processor.py
    main.py
  tests/test_pipeline_smoke.py
  requirements.txt
```

## Features

- Real-time audio loopback stream with chunked processing
- NumPy-based low-latency pitch shifting (pitched-down playback path)
- WebSocket audio streaming endpoint (`/ws/audio`) for browser-to-backend live audio
- Voice profile switching over WebSocket control messages
- Profile-aware AI inference routing with optional profile-specific model paths
- Dockerized backend runtime (`Dockerfile`, `docker-compose.yml`)

## Prerequisites

- Python 3.12+
- Linux audio runtime libs:
  - `libportaudio2`
  - `libasound2-dev`

Install system libs (Ubuntu/Debian):

```bash
sudo apt-get update
sudo apt-get install -y libportaudio2 libasound2-dev
```

## Setup (Recommended via pyproject)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Alternative:

```bash
pip install -r requirements.txt
```

## Run Backend (FastAPI WebSocket)

```bash
PYTHONPATH=src .venv/bin/python -m uvicorn app.api.server:app --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Inference status:

```bash
curl http://127.0.0.1:8000/health/inference
```

## Run Frontend

```bash
cd frontend
npm ci
npm run dev
```

Then open the local Vite URL (typically `http://127.0.0.1:5173`).

## Run Local Audio Loopback Demo

```bash
PYTHONPATH=src .venv/bin/python -m app.main
```

Note: The host machine must have available input/output audio devices.

## WebSocket Audio Contract

Endpoint: `ws://localhost:8000/ws/audio`

Client to server:

- Binary: PCM16 mono chunks (`Int16Array` bytes)
- Text JSON control:

```json
{ "type": "set_profile", "profile": "deep" }
```

Supported profiles:

- `natural`
- `robot`
- `deep`
- `chipmunk`

Server to client:

- Binary processed PCM16 chunks
- JSON error messages for invalid control payloads

## Frontend Component

Frontend is a complete minimal React app scaffolded with Vite:

- `frontend/src/components/VoiceChangerControlPanel.tsx`
- `frontend/src/hooks/useVoiceStream.ts`
- `frontend/public/worklets/pcm-capture-worklet.js`

## Environment Variables

Copy `.env.example` to `.env` and customize as needed.

Profile-specific model overrides are supported:

- `VOICE_TORCH_MODEL_PATH_ROBOT`
- `VOICE_TORCH_MODEL_PATH_DEEP`
- `VOICE_TORCH_MODEL_PATH_CHIPMUNK`
- `VOICE_ONNX_MODEL_PATH_ROBOT`
- `VOICE_ONNX_MODEL_PATH_DEEP`
- `VOICE_ONNX_MODEL_PATH_CHIPMUNK`

## Docker

```bash
docker compose up --build
```

Backend will be available on `http://127.0.0.1:8000`.

## Testing

```bash
PYTHONPATH=src .venv/bin/pytest -q
```

Lint and formatting checks:

```bash
ruff check .
black --check .
```

Frontend checks:

```bash
cd frontend
npm run typecheck
npm run build
```

## CI

GitHub Actions workflow in `.github/workflows/ci.yml` runs backend lint/test and frontend typecheck/build on push and pull request.

## License

MIT. See `LICENSE`.
