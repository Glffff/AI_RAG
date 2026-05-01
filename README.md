# RAG Project

PDF-based Retrieval Augmented Generation (RAG) system with Domain Driven Design (DDD) architecture.

## Prerequisites

- Python 3.12+
- UV installed: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Ollama running: `ollama serve`
- Qdrant running: `docker run -p 6333:6333 qdrant/qdrant`

## Setup

**Install dependencies:**
```bash
uv sync
source .venv/bin/activate
```

**Configuration:**
- Settings: `configs/local-settings.toml`
- Secrets: `.env` (in root)

## Run

```bash
uv run main.py
```

API docs: `http://localhost:8000/docs`
Qdrant UI: `http://localhost:6333/dashboard`

## API Endpoints

- `POST /upload` - Upload PDF file
- `POST /ask` - Ask question with citations

