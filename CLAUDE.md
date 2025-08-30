# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Retrieval-Augmented Generation (RAG) system that answers questions about course materials using semantic search and AI-powered responses. The application uses:
- ChromaDB for vector storage
- Anthropic's Claude, OpenAI's GPT, or Google's Gemini for AI generation
- FastAPI for the web backend
- A React frontend (in the frontend/ directory)

## Key Architecture Components

1. **Backend** (`backend/`):
   - `app.py`: Main FastAPI application with endpoints
   - `rag_system.py`: Core orchestrator that coordinates document processing, vector storage, and AI generation
   - `document_processor.py`: Handles parsing and chunking of course documents
   - `vector_store.py`: Manages ChromaDB interactions for storing/retrieving document embeddings
   - `ai_generator.py`: Provider-agnostic AI generation interface supporting multiple LLM providers
   - `config.py`: Configuration management with environment variables
   - `search_tools.py`: Implements semantic search functionality using tools
   - `session_manager.py`: Manages conversation history
   - `models.py`: Data models for courses, lessons, and chunks

2. **Frontend** (`frontend/`): React application for user interface

3. **Documents** (`docs/`): Course materials in PDF, DOCX, or TXT format

## Development Commands

### Setup
```bash
# Install dependencies
uv sync

# Set up environment variables in .env:
# ANTHROPIC_API_KEY=your_key_here
# LLM_PROVIDER=anthropic  # or openai or gemini
```

### Running the Application
```bash
# Quick start (recommended)
chmod +x run.sh
./run.sh

# Manual start
cd backend
uv run uvicorn app:app --reload --port 8000
```

### Adding Course Documents
Place PDF/DOCX/TXT files in the `docs/` folder. The system will automatically load them on startup.

### Testing
Currently, there are no established test patterns in this codebase. If you add tests, follow the existing code structure and patterns.

## Key URLs
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## Multi-Provider LLM Support
The system supports three LLM providers:
- Anthropic Claude (default)
- OpenAI GPT
- Google Gemini

Switch between them by setting the `LLM_PROVIDER` environment variable in `.env`.

# Team Memory

- Always use `uv` for Python env management, installs, and execution — never use `pip` directly.
  - Install/resolve deps: `uv sync`
  - Run server (manual): `cd backend && uv run uvicorn app:app --reload --port 8000`
  - Run tests: `uv run pytest -q`
  - Add deps: `uv add <pkg>` (or edit `pyproject.toml` then `uv sync`)
  - Do not run `pip install` or `python -m pip` in this project.

- The run script already uses `uv`:
  - `./run.sh` starts the API via `uv run uvicorn …`

- If a virtualenv appears broken, prefer `rm -rf .venv && uv sync` over pip.

- make sure to use uv to manage all dependencies

- use uv to run python files