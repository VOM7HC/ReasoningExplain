# Conda-aware developer shortcuts

ENV_NAME ?= reasoning-explain
CONDA ?= conda

.PHONY: help env setup dev kernel lab test cov lint fmt run precommit local-run

help:
	@echo "Available targets:"
	@echo "  env        - Create/update conda env ($(ENV_NAME)) from environment.yml"
	@echo "  setup      - env + dev + kernel"
	@echo "  dev        - Editable install with dev extras and pre-commit"
	@echo "  kernel     - Register Jupyter kernel for the env"
	@echo "  lab        - Launch JupyterLab inside the env"
	@echo "  test       - Run pytest"
	@echo "  cov        - Run pytest with coverage"
	@echo "  lint       - Ruff on code + notebooks via nbqa"
	@echo "  fmt        - Format code + notebooks"
	@echo "  run        - Run module (__main__) inside the env"
	@echo "  precommit  - Run pre-commit on all files"
	@echo "  local-run  - Run module with system Python (no conda)"

env:
	$(CONDA) env create -f environment.yml -n $(ENV_NAME) || \
	$(CONDA) env update -f environment.yml -n $(ENV_NAME)

setup: env dev kernel

dev:
	$(CONDA) run -n $(ENV_NAME) pip install -e .[dev]
	$(CONDA) run -n $(ENV_NAME) pre-commit install

kernel:
	$(CONDA) run -n $(ENV_NAME) python -m ipykernel install --user --name $(ENV_NAME)

lab:
	$(CONDA) run -n $(ENV_NAME) jupyter lab

test:
	$(CONDA) run -n $(ENV_NAME) pytest -q

cov:
	$(CONDA) run -n $(ENV_NAME) pytest --cov=src --cov-report=term-missing

lint:
	$(CONDA) run -n $(ENV_NAME) ruff check src tests
	$(CONDA) run -n $(ENV_NAME) nbqa ruff notebooks/

fmt:
	$(CONDA) run -n $(ENV_NAME) ruff format . || $(CONDA) run -n $(ENV_NAME) black .
	$(CONDA) run -n $(ENV_NAME) nbqa black notebooks/

run:
	$(CONDA) run -n $(ENV_NAME) python -m reasoning_explain

precommit:
	$(CONDA) run -n $(ENV_NAME) pre-commit run -a

# Fallback for environments without conda (e.g., CI smoke test)
local-run:
	PYTHONPATH=src python -m reasoning_explain

