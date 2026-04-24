# OOO - learning models, buying time

Private-use nightly stock recommender that produces **BUY / HOLD / SELL** signals for the top 30 US equities by volume. Signals are generated after market close and consumed before open.

---

## Architecture

```
scripts/run_nightly.py
├── Ingestion        — OHLCV via Massive (Polygon.io) + sentiment via Alpha Vantage
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
- A [Massive](https://massive.com) (formerly Polygon.io) Basic-plan API key
- An [Alpha Vantage](https://www.alphavantage.co) API key (premium recommended; free tier allows only 25 calls/day)
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
#   POLYGON_API_KEY=<your Massive key>
#   ALPHA_VANTAGE_API_KEY=<your Alpha Vantage key>
#   API_TOKEN=<random string, ≥32 chars>
```

---

## Running

### Start the API server (foreground)

```bash
make run
# Server starts at http://127.0.0.1:8000
# Returns 503 until a model artifact exists (run make nightly first)
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
make nightly START_DATE=2024-04-23
# Or calling the script directly:
.venv/bin/python scripts/run_nightly.py --start-date 2024-04-23
```

The nightly pipeline should be run **after market close** (US Eastern). Grouped daily data from Massive is typically published 15–30 minutes after 4 PM ET. If run earlier, the pipeline automatically falls back to the previous trading day's universe.

### First run

On a fresh install the model must be trained before the server can serve predictions:

```bash
make run &                               # start server in background (will return 503 initially)
make nightly START_DATE=2024-01-02      # ingest 2 years + train + evaluate
# once training completes, restart the server to pick up the new artifact
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
  -d '{"tickers": ["AAPL", "MSFT"], "predict_date": "2026-04-24"}'
```

```json
{
  "run_date": "2026-04-24",
  "model": "autoformer",
  "predictions": [
    {
      "ticker": "AAPL",
      "signal": "BUY",
      "confidence": 0.62,
      "explanation": {
        "top_features": [
          {"feature": "macd", "shap_value": 0.041},
          {"feature": "log_return", "shap_value": 0.033}
        ],
        "explainer_used": "TreeExplainer"
      }
    }
  ],
  "warnings": []
}
```

Omit `tickers` to get signals for the full 30-ticker universe. Omit `predict_date` to default to today.

Per-prediction explanations use SHAP TreeExplainer for tree-based models (Autoformer/PatchTST) and KernelExplainer fallback for MLP (N-HiTS).

---

## Development

```bash
make test        # Run full test suite (246 tests)
make coverage    # Tests + coverage report (≥85%; currently ~88%)
make lint        # ruff + mypy (zero errors)
make format      # black
make audit       # pip-audit security scan
make train       # Re-run training on existing data/features/
make notebook    # Launch Jupyter for EDA
```

### Project structure

```
src/
├── ingestion/      — Massive + Alpha Vantage clients, rate limiter, pipeline
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
models/production/  — Single production artifact (latest winning model only)
reports/            — Evaluation and monitoring reports
```

---

## Data sources

| Source | Data | Plan |
|---|---|---|
| [Massive](https://massive.com) (formerly Polygon.io) | OHLCV, universe resolution (top 30 by volume) | Basic (100 calls/min) |
| [Alpha Vantage](https://www.alphavantage.co) | News sentiment — aggregated per-ticker score from article feed | Premium recommended |

**Finnhub** was used for sentiment in early development but dropped — the `news-sentiment` endpoint requires a paid plan and is no longer part of this pipeline.

Raw data is immutable and never overwritten. All pipeline steps are idempotent.

### Sentiment aggregation

Alpha Vantage returns a news article feed. For each ticker, the pipeline:
1. Filters articles where the ticker's `relevance_score ≥ 0.1`
2. Computes `bullish_percent` / `bearish_percent` from per-article sentiment labels
3. Computes `company_news_score` as the mean `ticker_sentiment_score` (range −1 to +1)
4. Records `buzz_weekly_average` as the count of relevant articles

---

## Models

Three candidates are evaluated in walk-forward cross-validation (252-day train window, 21-day step, minimum 3 folds). The model with the highest F1-macro on the final fold wins and is written exclusively to `models/production/` — previous winners are evicted automatically.

| Model | Sklearn backend | Notes |
|---|---|---|
| **N-HiTS** | MLPClassifier | Multi-horizon; strong on seasonality decomposition |
| **PatchTST** | GradientBoostingClassifier | Patch-based; strong on local financial patterns |
| **Autoformer** | ExtraTreesClassifier | Included when the above underperform |

> The sklearn backends are a pragmatic substitute until `torch` + `neuralforecast` are installed. The wrapper interface (`train` / `predict` / `predict_proba` / `save` / `load`) is stable — swapping in the real architectures requires only replacing the model internals.

Ties in F1-macro are broken in the order N-HiTS → PatchTST → Autoformer.

### Quality gate

Evaluated after each training run against the production fold:

| Metric | Threshold |
|---|---|
| F1-macro | ≥ 0.35 |
| MCC | ≥ 0.05 |
| Hit rate | ≥ 0.50 |
| Max signal class share | ≤ 80% |

If the gate fails, the `retraining_required` flag stays set in `data/monitoring/status.json` and training is re-attempted on the next nightly run.

---

## Monitoring

Run nightly before the training decision. Results written to `reports/monitoring/monitoring_history.csv`.

| Check | Method | Trigger |
|---|---|---|
| Feature drift | KS test + PSI on continuous features | Both KS p-value < 0.05 **and** PSI ≥ 0.20 |
| Prediction drift | Chi-squared on signal distribution | p-value < 0.05, or any class ≥ 80% |
| Performance degradation | Rolling 21-day hit rate | Two consecutive windows below 45% |

Calendrical and boolean features (`month`, `week_of_year`, `day_of_week`, `is_month_end`, flag columns) are excluded from KS/PSI checks — their distributions shift with the calendar window, not with regime change.

Alert files written to `data/monitoring/alerts/{date}.json`. Retraining state persisted in `data/monitoring/status.json`. Evidently HTML + JSON reports generated every 7 runs to `reports/monitoring/`.

---

## Security

- No secrets in code — use `.env` (never committed)
- Bearer token auth on all API endpoints; token must be ≥ 32 characters
- `pip-audit` run before any release; currently reports no known vulnerabilities
- No PII stored; signals are per-ticker, not per-user

---

## Disclaimer

This project is for private, non-commercial use only. It is not financial advice. Verify the terms of service for Massive and Alpha Vantage on any plan changes.
