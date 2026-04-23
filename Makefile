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
	$(PYTHON) -m uvicorn src.serving.app:app --host 127.0.0.1 --port 8000 --log-level info

serve-install: ## Install launchd plist for macOS background autostart
	cp scripts/com.stockrecommender.api.plist ~/Library/LaunchAgents/
	launchctl load ~/Library/LaunchAgents/com.stockrecommender.api.plist

serve-uninstall: ## Unload and remove launchd plist
	launchctl unload ~/Library/LaunchAgents/com.stockrecommender.api.plist
	rm -f ~/Library/LaunchAgents/com.stockrecommender.api.plist

train: ## Run training harness
	$(PYTHON) scripts/run_nightly.py --train-only

nightly: ## Run full nightly batch pipeline
	$(PYTHON) scripts/run_nightly.py

notebook: ## Launch Jupyter for EDA
	$(PYTHON) -m jupyter lab notebooks/
