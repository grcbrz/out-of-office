.DEFAULT_GOAL := help

PYTHON := .venv/bin/python
PIP    := .venv/bin/pip

.PHONY: help install test coverage lint format audit run serve-install serve-uninstall train nightly notebook

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install: ## Install all dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt

test: ## Run full test suite
	$(PYTHON) -m pytest tests/

coverage: ## Run tests with coverage report
	$(PYTHON) -m pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=85

lint: ## Run ruff + mypy
	$(PYTHON) -m ruff check src/ tests/
	$(PYTHON) -m mypy src/

format: ## Run black formatter
	$(PYTHON) -m black src/ tests/ scripts/

audit: ## Run pip-audit security check
	$(PYTHON) -m pip_audit

run: ## Start FastAPI server (foreground)
	@set -a && [ -f .env ] && . ./.env; set +a; \
	$(PYTHON) -m uvicorn src.serving.app:app --host 127.0.0.1 --port 8000 --log-level info

serve-install: ## Install API server + nightly scheduler as launchd services
	mkdir -p ~/Library/LaunchAgents
	/bin/cp -f scripts/com.stockrecommender.api.plist ~/Library/LaunchAgents/
	/bin/cp -f scripts/com.stockrecommender.nightly.plist ~/Library/LaunchAgents/
	launchctl bootout gui/$$(id -u)/com.stockrecommender.api 2>/dev/null || true
	launchctl bootout gui/$$(id -u)/com.stockrecommender.nightly 2>/dev/null || true
	sleep 1
	launchctl bootstrap gui/$$(id -u) ~/Library/LaunchAgents/com.stockrecommender.api.plist
	launchctl bootstrap gui/$$(id -u) ~/Library/LaunchAgents/com.stockrecommender.nightly.plist
	@echo "API server: running (port 8000)"
	@echo "Nightly pipeline: scheduled 01:30 UTC (21:30 ET). Logs: /tmp/com.stockrecommender.nightly.stdout.log"

serve-uninstall: ## Unload and remove both launchd services
	launchctl bootout gui/$$(id -u)/com.stockrecommender.api 2>/dev/null || true
	launchctl bootout gui/$$(id -u)/com.stockrecommender.nightly 2>/dev/null || true
	rm -f ~/Library/LaunchAgents/com.stockrecommender.api.plist
	rm -f ~/Library/LaunchAgents/com.stockrecommender.nightly.plist

train: ## Run training harness on existing data/features/
	$(PYTHON) -c "import pandas as pd; from pathlib import Path; from src.models.training_pipeline import TrainingPipeline; \
files = list(Path('data/features').glob('*/*.csv')); \
TrainingPipeline().run(pd.concat([pd.read_csv(f) for f in files], ignore_index=True))"

nightly: ## Run full nightly batch pipeline (optional: START_DATE=YYYY-MM-DD)
	@set -a && [ -f .env ] && . ./.env; set +a; \
	$(PYTHON) scripts/run_nightly.py $(if $(START_DATE),--start-date $(START_DATE),)

notebook: ## Launch Jupyter for EDA
	$(PYTHON) -m jupyter lab notebooks/
