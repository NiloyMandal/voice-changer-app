# Contributing

## Development Setup

1. Create and activate a virtual environment.
2. Install backend dependencies:

```bash
pip install -e .[dev]
```

3. Install frontend dependencies:

```bash
cd frontend
npm ci
```

## Running Locally

1. Start backend:

```bash
PYTHONPATH=src .venv/bin/python -m uvicorn app.api.server:app --host 127.0.0.1 --port 8000
```

2. Start frontend:

```bash
cd frontend
npm run dev
```

## Tests and Checks

Run backend checks:

```bash
ruff check .
black --check .
PYTHONPATH=src pytest -q
```

Run frontend checks:

```bash
cd frontend
npm run typecheck
npm run build
```

## Pull Requests

- Keep PRs focused and small.
- Include tests for behavior changes.
- Update documentation when adding or changing features.
