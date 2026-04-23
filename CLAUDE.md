# CLAUDE.md — Stock Recommender App

## Project Overview

**Purpose:** Private-use stock recommender system producing Buy/Hold/Sell signals via time series forecasting.
**Stack:** Python · VS Code · GitHub
**Status:** Greenfield
**Forecast horizon:** Next-day signal (t+1)
**Stock universe:** Top 50 US stocks by volume (dynamically resolved nightly)
**Run mode:** Nightly batch job — signals generated after market close, consumed before open
**Data sources:**
- **Polygon.io** (free tier) — OHLCV + ticker universe; 5 calls/min rate limit, 2 years historical daily data
- **Finnhub** (free tier) — pre-computed news sentiment scores per ticker; 60 calls/min rate limit

---

## Architecture Principles

- **Spec-Driven Development (SDD):** Specs precede implementation. No code without a spec.
- **Test-Driven Development (TDD):** Red → Green → Refactor. Tests are first-class citizens.
- **Harness Engineering:** All model runs, data pipelines, and experiments are wrapped in reproducible harnesses.
- **Clean Code:** Readable > clever. Single responsibility. No magic numbers. No dead code.
- **Modular design:** Strict separation — data layer / feature layer / model layer / evaluation layer / serving layer.

---

## Repository Structure

```
stocks-recommender/
├── specs/                  # SDD specs (Markdown, one per feature/component)
├── data/
│   ├── raw/                # Immutable source data
│   ├── processed/          # Cleaned, validated datasets
│   ├── features/           # Engineered feature sets
│   ├── predictions/        # Nightly signal output (CSV, append-only)
│   └── monitoring/         # Alert files, status.json
├── reports/
│   ├── evaluation/         # Per-run metrics CSV + confusion matrices
│   └── monitoring/         # Evidently HTML reports + monitoring_history.csv
├── models/
│   └── production/         # Latest winning model artifact only
├── src/
│   ├── ingestion/          # Data acquisition — Polygon.io (OHLCV) + Finnhub (sentiment)
│   ├── preprocessing/      # Cleaning, validation, normalisation
│   ├── features/           # Feature engineering pipelines
│   ├── models/             # Model definitions (Autoformer, baselines)
│   ├── evaluation/         # Metrics, explainability, reporting
│   ├── serving/            # API / dashboard / inference endpoint
│   └── monitoring/         # Drift detection, retraining triggers
├── tests/
│   ├── unit/
│   ├── integration/
│   └── acceptance/         # Behaviour-level tests tied to specs
├── notebooks/              # EDA only — not production code
├── configs/                # YAML configs (hyperparams, pipeline settings)
├── scripts/                # CLI entry points, one-off jobs
│   ├── run_nightly.py      # nightly batch orchestrator
│   ├── prediction_client.py
│   └── com.stockrecommender.api.plist  # macOS launchd service definition
├── docker/
├── requirements.txt        # Pinned dependencies
├── requirements-dev.txt
├── .env.example            # Template — never commit .env
├── Makefile                # build, test, lint, run targets
└── CLAUDE.md               # This file
```

---

## Coding Standards

### General

- Python 3.11+
- Type hints on all function signatures (use `from __future__ import annotations` where needed)
- Docstrings on public functions/classes (Google style)
- Max line length: 88 (Black default)
- No commented-out code committed to main

### Naming

- `snake_case` for variables, functions, modules
- `PascalCase` for classes
- `UPPER_SNAKE_CASE` for constants
- Prefix private methods with `_`

### Functions

- Single responsibility
- Max ~20 lines; extract if longer
- No side effects in pure computation functions
- Explicit over implicit — no `**kwargs` abuse

### Error Handling

- Raise specific exceptions, never bare `except:`
- Validate inputs at boundaries (ingestion, API endpoints)
- Log errors with context; do not swallow silently

---

## Testing Requirements

| Layer       | Tooling              | Coverage Target |
|-------------|----------------------|-----------------|
| Unit        | `pytest`             | ≥ 85%           |
| Integration | `pytest` + fixtures  | Key pipelines   |
| Acceptance  | `pytest-bdd` or plain pytest tied to specs | All specs |

- Tests live in `tests/` mirroring `src/` structure
- Use `pytest-cov` for coverage reporting
- Mock external APIs in unit tests (`unittest.mock` or `pytest-mock`)
- No test should touch real market data APIs in CI

---

## Data Pipeline Standards

- **Raw data is immutable.** Never overwrite `data/raw/`.
- Validate schema and dtypes at ingestion boundary using `pydantic` models
- Log data quality issues; fail loudly on critical violations
- Track data lineage in metadata files (source, date pulled, version)
- All pipeline steps must be idempotent
- **Ingestion is a nightly batch job.** Schedule after market close (US Eastern). Do not design for real-time.
- **Rate limiting is mandatory at the ingestion layer:**
  - Polygon.io: max 5 calls/min — use a token bucket or `tenacity` with explicit sleep
  - Finnhub: max 60 calls/min — same pattern, lower pressure
  - Design rate limiter as a reusable utility; never hard-code `time.sleep()` inline
- **Universe resolution:** Top 50 by volume is resolved fresh each night via Polygon's grouped daily endpoint before fetching per-ticker data

---

## Modelling Standards

### Model Selection

- Always benchmark against at least one naive baseline (e.g. last-value, moving average)
- Document justification for chosen architecture in the relevant spec
- Candidate models to evaluate (in complexity order):
  1. Baseline (naive, moving average, linear regression) — mandatory reference point
  2. N-HiTS — efficient, strong on multi-horizon, handles seasonality decomposition natively
  3. PatchTST — patch-based transformer, strong on financial series with local patterns
  4. Autoformer — designed for long horizons; for t+1 signals it is likely overkill; include only if N-HiTS and PatchTST underperform

- **Future experiments only (not current scope):** TimesNet

> Seasonality must always be accounted for. Decompose series (trend + seasonal + residual) before modelling. Use STL or model-native decomposition where available. Validate seasonal periods empirically — do not assume.

### Output

- Final prediction: `BUY / HOLD / SELL` with confidence score
- Probabilistic output preferred over point predictions where feasible
- Never output a signal without an associated uncertainty estimate

### Explainability

- SHAP values for feature importance (use `shap` library)
- Attention weights visualisation for transformer models
- Per-prediction explanation logged alongside signal

### Evaluation Metrics

- **Classification:** Precision, Recall, F1 (macro + per class), MCC, ROC-AUC
- **Forecast quality:** MAE, RMSE, MAPE, Directional Accuracy
- **Financial:** Sharpe ratio, Maximum Drawdown, Hit Rate on directional calls
- **Validation strategy:** Walk-forward (expanding or sliding window) — random splits are forbidden for time series
- Respect temporal order in all train/val/test splits; no data from the future leaks into past windows

---

## Security Requirements

- **No secrets in code.** Use `.env` + `python-dotenv`; `.env` is in `.gitignore`
- API keys, tokens, brokerage credentials → environment variables only
- `.env.example` maintained with all required keys (no values)
- Input validation on all external data before processing
- Dependency audit: run `pip-audit` before any release
- No PII stored; if user data is added later, apply data minimisation

---

## Experiment Tracking

- Use `MLflow` (local) or `Weights & Biases` for run tracking
- Every training run logs: hyperparams, metrics, dataset version, git commit hash
- Model artifacts versioned and stored with metadata
- No "it worked on my machine" — harness must be reproducible from config alone

---

## Monitoring & Retraining

- Feature drift: KS test + PSI as programmatic gates; both must breach threshold to trigger
- Prediction drift: chi-squared test on signal distribution
- Performance degradation: rolling 21-day hit rate; 2 consecutive windows below threshold required
- Evidently AI reports generated weekly (HTML + JSON) for human review
- Retraining triggered on drift/degradation in addition to scheduled 21-day cadence
- All alerts written to `data/monitoring/alerts/{date}.json`; no external notifications
- `data/monitoring/status.json` read by nightly batch to determine if unscheduled retraining is needed

---

## Serving

- FastAPI REST API: `/predict`, `/health`, `/metrics` endpoints
- Bearer token auth on all endpoints; token loaded from `API_TOKEN` env var (≥ 32 chars)
- Server must be running before nightly batch starts — launch as a `launchd` service on macOS
- `launchd` plist included in `scripts/com.stockrecommender.api.plist` for one-time system setup
- Prediction triggered automatically by nightly batch via internal HTTP client after evaluation
- On non-retraining nights: ingestion → preprocessing → features → prediction only
- Per-prediction output includes signal, confidence, SHAP top-5 features, attention weights

---

## Dependency Management

```bash
# Install
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Freeze after adding packages
pip freeze > requirements.txt

# Audit
pip-audit
```

- Pin all versions (`==`) in `requirements.txt`
- Dev tools (pytest, black, ruff, mypy) in `requirements-dev.txt`
- Python version managed via `.python-version` (pyenv)

---

## Tooling

| Purpose        | Tool              |
|----------------|-------------------|
| Formatting     | `black`           |
| Linting        | `ruff`            |
| Type checking  | `mypy`            |
| Testing        | `pytest`          |
| Coverage       | `pytest-cov`      |
| Data validation| `pydantic`        |
| Explainability | `shap`            |
| Experiment tracking | `mlflow`     |
| Serving        | `fastapi` + `uvicorn` |
| Drift monitoring | `evidently`     |
| Statistical drift gates | `scipy` (KS test, chi-squared, PSI) |
| Retry + backoff | `tenacity`       |
| Model architectures | `neuralforecast` |
| Trading calendar | `pandas_market_calendars` |
| Storage format | CSV (raw, processed, features, predictions) |
| Security audit | `pip-audit`       |

---

## Git Workflow

- Branch naming: `feat/`, `fix/`, `refactor/`, `chore/`, `experiment/`
- Commit messages: Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`)
- No direct commits to `main`
- PRs require passing tests + lint (even solo — enforce the habit)
- Tag releases with semantic versioning

---

## Makefile Targets

```makefile
make install         # pip install both requirement files
make test            # run full test suite
make coverage        # test + coverage report
make lint            # ruff + mypy
make format          # black
make audit           # pip-audit
make run             # start FastAPI server (foreground)
make serve-install   # install launchd plist for background autostart on macOS
make serve-uninstall # unload and remove launchd plist
make train           # run training harness
make nightly         # run full nightly batch pipeline
make notebook        # launch Jupyter for EDA
```

---

## Constraints & Scope (Private Use)

- No real-money execution — signal generation only
- Polygon.io and Finnhub free tier ToS permit private, non-commercial use — verify on any tier upgrade
- Do not cache or redistribute raw API data beyond personal use
- This project is **not** financial advice and must not be represented as such

---

## References

- [Autoformer paper](https://arxiv.org/abs/2106.13008) — Wu et al., 2021
- [PatchTST paper](https://arxiv.org/abs/2211.14730) — Nie et al., 2022
- [N-HiTS paper](https://arxiv.org/abs/2201.12886) — Challu et al., 2022
- [NeuralForecast](https://nixtlaverse.nixtla.io/neuralforecast/index.html)
- [Polygon.io REST API docs](https://polygon.io/docs/stocks)
- [Finnhub API docs](https://finnhub.io/docs/api)
- [pandas-market-calendars](https://pandas-market-calendars.readthedocs.io/)
- [tenacity](https://tenacity.readthedocs.io/)
- [SHAP](https://shap.readthedocs.io/)
- [Evidently AI](https://docs.evidentlyai.com/)
- [MLflow](https://mlflow.org/docs/latest/index.html)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Pydantic](https://docs.pydantic.dev/)
- [macOS launchd reference](https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPSystemStartup/Chapters/CreatingLaunchdJobs.html)
- *Advances in Financial Machine Learning* — Marcos López de Prado
- *Time Series Forecasting in Python* — Marco Peixeiro
