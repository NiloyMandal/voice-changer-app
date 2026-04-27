# Voice Changer App

Low-latency live voice changer prototype with:
- Python backend audio pipeline and inference abstraction
- FastAPI WebSocket endpoint for browser audio streaming
- React control panel component for Start/Stop and Voice Profile control

## Current Project Structure

```text
voice-changer-app/
  frontend/
    src/components/VoiceChangerControlPanel.tsx
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

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
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
{"type":"set_profile","profile":"deep"}
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

File: `frontend/src/components/VoiceChangerControlPanel.tsx`

Provided UI controls:
- Start/Stop button
- Voice Profiles dropdown

This repository currently includes the component file, not a full frontend app scaffold.
Integrate it into your React app and point `wsUrl` to your backend server.

## Environment Variables

Copy `.env.example` to `.env` and customize as needed.

## Testing

```bash
PYTHONPATH=src .venv/bin/pytest -q
```

## License

No license file has been added yet. Add one if you plan to make this repository public.
