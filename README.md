# OOO - learning models, buying time

Private-use nightly stock recommender that produces **BUY / HOLD / SELL** signals for the top 50 US equities by volume. Signals are generated after market close and consumed before open.

---

## Architecture

```
scripts/run_nightly.py
├── Ingestion        — OHLCV + sentiment via Polygon.io & Finnhub
├── Preprocessing    — cleaning, imputation, outlier detection
├── Feature Eng.     — log returns, MACD, OBV, VWAP, lags, seasonality
├── Monitoring       — feature drift (KS+PSI), prediction drift, hit-rate degradation
├── Training         — walk-forward harness; N-HiTS / PatchTST / Autoformer
├── Evaluation       — F1-macro, MCC, Sharpe, quality gate
└── Prediction       — POST /predict via internal HTTP client
```

The FastAPI server runs as a persistent background service (launchd). The nightly batch script calls it via `PredictionClient` once the pipeline completes.

---

## Requirements

- Python 3.11+
- [pyenv](https://github.com/pyenv/pyenv) (recommended; `.python-version` pins 3.12.9)
- A [Polygon.io](https://polygon.io) free-tier API key
- A [Finnhub](https://finnhub.io) free-tier API key
- A bearer token of at least 32 characters for the API (`API_TOKEN`)

---

## Setup

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd out-of-office

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
make install

# 4. Configure environment variables
cp .env.example .env
# Edit .env and fill in:
#   POLYGON_API_KEY=<your key>
#   FINNHUB_API_KEY=<your key>
#   API_TOKEN=<random string, ≥32 chars>
```

---

## Running

### Start the API server (foreground)

```bash
make run
# Server starts at http://127.0.0.1:8000
```

### Install as a macOS background service (launchd)

```bash
make serve-install
# Runs automatically at login; survives reboots
```

```bash
make serve-uninstall   # Remove the service
```

### Run the nightly batch manually

```bash
make nightly
# Or with an explicit start date:
.venv/bin/python scripts/run_nightly.py --start-date 2024-01-02
```

---

## API

All endpoints require a bearer token header:

```
Authorization: Bearer <API_TOKEN>
```

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | `200 OK` when model loaded; `503` in degraded mode |
| `/metrics` | GET | In-memory counters (total predictions, signal distribution) |
| `/predict` | POST | Generate BUY/HOLD/SELL signals |

### Example predict request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["AAPL", "MSFT"], "predict_date": "2026-04-23"}'
```

```json
{
  "run_date": "2026-04-23",
  "model": "nhits",
  "predictions": [
    {
      "ticker": "AAPL",
      "signal": "BUY",
      "confidence": 0.68,
      "explanation": {"top_features": [], "attention_weights": null, "explainer_used": "shap"}
    }
  ],
  "warnings": []
}
```

Omit `tickers` to get signals for the full 50-ticker universe. Omit `predict_date` to default to today.

---

## Development

```bash
make test        # Run full test suite (203 tests)
make coverage    # Tests + coverage report (target ≥85%)
make lint        # ruff + mypy
make format      # black
make audit       # pip-audit security scan
make notebook    # Launch Jupyter for EDA
```

### Project structure

```
src/
├── ingestion/      — Polygon.io + Finnhub clients, rate limiter, pipeline
├── preprocessing/  — Imputation, outlier detection, normalisation, merger
├── features/       — Returns, trend, volume, lags, seasonality, target label
├── models/         — Walk-forward harness, N-HiTS/PatchTST/Autoformer wrappers
├── evaluation/     — Classification metrics, financial metrics, quality gate, SHAP
├── serving/        — FastAPI app, auth, artifact loader, inference engine
└── monitoring/     — Feature drift, prediction drift, degradation, retraining trigger

tests/
├── unit/           — Per-module unit tests (mocked external dependencies)
├── integration/    — Pipeline-level tests
└── acceptance/     — Spec-level behaviour tests

configs/            — YAML configs for all pipeline stages
scripts/            — CLI entry points and launchd plist
specs/              — Spec-driven design documents (01–07)
data/               — Raw, processed, features, predictions, monitoring outputs
models/production/  — Single production artifact (winning model)
reports/            — Evaluation and monitoring reports
```

---

## Data sources

| Source | Data | Rate limit |
|---|---|---|
| [Polygon.io](https://polygon.io/docs/stocks) free tier | OHLCV, universe resolution | 5 calls/min |
| [Finnhub](https://finnhub.io/docs/api) free tier | Pre-computed news sentiment | 60 calls/min |

Raw data is immutable and never overwritten. All pipeline steps are idempotent.

---

## Models

Candidates evaluated in walk-forward cross-validation (252-day train window, 21-day step):

1. **N-HiTS** — default winner; efficient multi-horizon model with seasonality decomposition
2. **PatchTST** — patch-based transformer; strong on local financial patterns
3. **Autoformer** — included when N-HiTS and PatchTST underperform

The model with the highest F1-macro across folds wins. Ties broken in the order above. The winning artifact is saved to `models/production/`.

Quality gate (evaluated after each training run):

| Metric | Threshold |
|---|---|
| F1-macro | ≥ 0.35 |
| MCC | ≥ 0.05 |
| Hit rate | ≥ 0.50 |
| Max signal class share | ≤ 80% |

---

## Monitoring

Run nightly before the training decision:

- **Feature drift** — KS test + PSI per feature; both must breach thresholds to trigger retraining
- **Prediction drift** — chi-squared test on signal distribution; also flags degenerate output (>80% one class)
- **Performance degradation** — rolling 21-day hit rate; two consecutive windows below 45% triggers retraining
- **Evidently reports** — HTML + JSON generated every 7 runs to `reports/monitoring/`

Alert files written to `data/monitoring/alerts/{date}.json`. Retraining state persisted in `data/monitoring/status.json`.

---

## Security

- No secrets in code — use `.env` (never committed)
- Bearer token auth on all API endpoints; token must be ≥ 32 characters
- `pip-audit` run before any release; currently reports no known vulnerabilities
- No PII stored; signals are per-ticker, not per-user

---

## Disclaimer

This project is for private, non-commercial use only. It is not financial advice. Polygon.io and Finnhub free-tier terms of service permit personal use — verify on any tier upgrade.
