# Repository Guidelines

## Project Structure & Module Organization
- `notebooks/` — research notebooks; one topic per file; top cell = Summary/Inputs/Outputs.
- `src/reasoning_explain/` — importable Python package backing notebooks.
- `tests/` — unit tests mirroring `src/` (e.g., `tests/test_core.py`).
- `scripts/` — small, idempotent CLIs for data/tasks.
- `data/` — local data only (gitignored). Keep large artifacts out of Git.
- `docs/` — thesis figures/notes generated from notebooks.

Example import: `from reasoning_explain.core import Explainer` (module at `src/reasoning_explain/core.py`).

## Environment (Conda) & Setup
- Authoritative spec: `environment.yml` (update when adding deps).
- Create env: `conda env create -f environment.yml -n reasoning-explain`
- Activate: `conda activate reasoning-explain`
- Dev install: `pip install -e .[dev] && pre-commit install`
- Kernel: `python -m ipykernel install --user --name reasoning-explain`
- Launch: `jupyter lab` (preferred) or `jupyter notebook`.

## Build, Test, and Development Commands
- Lint: `ruff check src tests` and notebooks via `nbqa ruff notebooks/`.
- Format: `ruff format .` (or `black .`) and `nbqa black notebooks/`.
- Tests: `pytest -q`; coverage: `pytest --cov=src --cov-report=term-missing`.
- Run module: `python -m reasoning_explain` (if `__main__.py` exists).

## Coding Style & Naming Conventions
- Python 3.10+; 4-space indent; max line length 100.
- Names: packages/modules `snake_case`; classes `PascalCase`; funcs/vars `snake_case`.
- Public APIs typed; Google-style docstrings. Keep modules cohesive (<400 LOC).
- Notebooks must run top-to-bottom, deterministic; clear outputs before commit (`pre-commit` uses `nbstripout`).

## Testing Guidelines
- Framework: `pytest` (+ `pytest-cov`). Tests mirror `src/` and use `test_*.py`.
- Aim ≥90% coverage on changed lines; include failure-path tests.
- Optional: notebook smoke tests via `pytest --nbmake` if plugin is enabled.

## Commit & Pull Request Guidelines
- Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`, `test:`...).
- Subject ≤72 chars; body explains motivation, approach, validation.
- PRs: link issues (`Closes #123`), include repro steps, screenshots/figures deltas, and notebook/env updates.

## Security & Data Hygiene
- Never commit secrets; provide `.env.example`. Keep `data/` out of Git; prefer small, anonymized samples in `assets/` when needed.

## Agent-Specific Instructions
- Favor minimal, surgical diffs. Keep `environment.yml` and `pre-commit` hooks current. Update Make targets only if present.
