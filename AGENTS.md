# Repository Guidelines

## Project Structure & Module Organization
- `backend/`: FastAPI app and RAG core
  - `app.py`: API routes, CORS, and static UI mount
  - `rag_system.py`: Orchestrates vector store, tools, sessions
  - `vector_store.py`: ChromaDB persistence under `backend/chroma_db`
  - `document_processor.py`: Chunking and course/lesson parsing
  - `ai_generator.py`: Anthropic client + tool calling
  - `search_tools.py`, `models.py`, `session_manager.py`, `config.py`
- `frontend/`: Static UI served at `/`
- `docs/`: Example course text auto‑ingested at startup
- `backend/tests/`: Pytest test files (`test_*.py`)
- Root: `pyproject.toml`, `.env(.example)`, `run.sh`

## Build, Test, and Development Commands
- Install deps: `uv sync` — resolves and installs Python dependencies.
- Run (script): `./run.sh` — starts FastAPI with reload on port 8000.
- Run (manual): `cd backend && uv run uvicorn app:app --reload --port 8000`.
- Env setup: `cp .env.example .env` then set `ANTHROPIC_API_KEY`.
- Tests: `uv run pytest -q` — runs unit tests quietly.

## Coding Style & Naming Conventions
- Language: Python 3.13; follow PEP 8, 4‑space indentation, prefer type hints.
- Names: functions/files use `snake_case`; classes and Pydantic models use `PascalCase`.
- Module focus: keep concerns isolated (e.g., search in `vector_store.py`, parsing in `document_processor.py`).
- Add concise docstrings; prefer small, testable functions.

## Testing Guidelines
- Framework: pytest.
- Location: `backend/tests/test_*.py`.
- What to test: chunking behavior, search filters, and API routes.
- Run locally: `uv run pytest -q` (aim for fast, deterministic tests).

## Commit & Pull Request Guidelines
- Commits: imperative mood, short subject (≤50 chars).
  - Example: `feat: add lesson filter to course search`
- PRs: include clear description, link issues (e.g., `Closes #123`), and steps to test; add screenshots for UI changes.
- Include any config defaults in `.env.example`. Never commit real secrets.

## Security & Configuration Tips
- Secrets only in `.env`; required: `ANTHROPIC_API_KEY`.
- CORS is wide‑open for development; restrict for production.
- Vector DB persists under `backend/chroma_db`; delete this folder to rebuild the index when needed.
- Do not commit `.env` or `backend/chroma_db/` (ensure `.gitignore` covers them).
