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