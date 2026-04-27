# OOO - learning models, buying time

Private-use nightly stock recommender that produces **BUY / HOLD / SELL** signals for a configurable fixed universe of equities (default: 15 tickers). Signals are generated after market close and consumed before open.

---

## Architecture

```
scripts/run_nightly.py
├── Ingestion        — OHLCV via Massive (Polygon.io) + news sentiment via Polygon + FinBERT
├── Preprocessing    — cleaning, imputation, outlier detection
├── Feature Eng.     — log returns, MACD, OBV, VWAP, lags, seasonality
├── Monitoring       — feature drift (KS+PSI), prediction drift, hit-rate degradation
├── Training         — walk-forward harness; N-HiTS / Autoformer
├── Evaluation       — F1-macro, MCC, Sharpe, quality gate
└── Prediction       — POST /predict via internal HTTP client
```

The FastAPI server runs as a persistent background service (launchd). The nightly batch script calls it via `PredictionClient` once the pipeline completes.

---

## Requirements

- Python 3.11+
- [pyenv](https://github.com/pyenv/pyenv) (recommended; `.python-version` pins 3.12.9)
- A [Massive](https://massive.com) (formerly Polygon.io) Basic-plan API key
- A bearer token of at least 32 characters for the API (`API_TOKEN`)
- `torch` + `transformers` (installed via `make install`; FinBERT downloads ~440 MB on first run)

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

Omit `tickers` to get signals for the full configured universe. Omit `predict_date` to default to today.

Per-prediction explanations use SHAP TreeExplainer for tree-based models (Autoformer) and KernelExplainer fallback for MLP (N-HiTS).

---

## Development

```bash
make test        # Run full test suite (250 tests)
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
├── ingestion/      — Massive OHLCV + Polygon news sentiment (FinBERT-scored), rate limiter, pipeline
├── preprocessing/  — Imputation, outlier detection, normalisation, merger
├── features/       — Returns, trend, volume, lags, seasonality, target label
├── models/         — Walk-forward harness, N-HiTS/Autoformer wrappers
├── evaluation/     — Classification metrics, financial metrics, quality gate, SHAP
├── serving/        — FastAPI app, auth, artifact loader, inference engine
└── monitoring/     — Feature drift, prediction drift, degradation, retraining trigger

tests/
├── unit/           — Per-module unit tests (mocked external dependencies)
├── integration/    — Pipeline-level tests
└── acceptance/     — Spec-level behaviour tests

configs/            — YAML configs for all pipeline stages (incl. fixed_universe list)
scripts/            — CLI entry points and launchd plist
specs/              — Spec-driven design documents (01–07)
notebooks/          — EDA: universe_selection, sentiment_exploration, yfinance_exploration
data/               — Raw, processed, features, predictions, monitoring outputs
models/production/  — Single production artifact (latest winning model only)
reports/            — Evaluation and monitoring reports
```

---

## Data sources

| Source | Data | Notes |
|---|---|---|
| [Massive](https://massive.com) (formerly Polygon.io) | OHLCV + news articles with per-ticker insights | Basic plan (100 calls/min); single API key for both data types |

The universe is **fixed** via `configs/ingestion.yaml` (`fixed_universe` list). When set, the Polygon volume-sort is bypassed entirely — Polygon is only used for OHLCV fetching and the news endpoint. Editing the list and re-running `make nightly` immediately switches to the new set of tickers.

Raw data is immutable and never overwritten. All pipeline steps are idempotent.

### Sentiment pipeline — Polygon news + FinBERT

The nightly ingestion calls `GET /v2/reference/news` for each ticker over a 24h window. Each article in the response includes an `insights[]` array with per-ticker sentiment labels and reasoning text.

For each ticker the pipeline:
1. Extracts all insights where `insight["ticker"] == ticker`
2. Counts `positive_insights` / `negative_insights` / `neutral_insights` from Polygon's pre-computed labels
3. Derives `bullish_percent` = positive / total, `bearish_percent` = negative / total
4. Runs **ProsusAI/FinBERT** on each `sentiment_reasoning` string (e.g. `"Earnings beat expectations significantly"`) to get a confidence score
5. Computes `company_news_score = Σ(sign × finbert_confidence) / n_insights`, clamped to [−1, 1]
6. Records `article_count` as articles mentioning the ticker in the window

FinBERT (`transformers`, `torch`) loads once at ingestion startup (~5–10 s from cache; ~440 MB download on first run). When reasoning text is absent, Polygon's own label is used at a fixed confidence of 0.5.

---

## Models

Two candidates are evaluated in walk-forward cross-validation (252-day train window, 21-day step, minimum 3 folds). The model with the highest **mean F1-macro across all folds** wins and is written exclusively to `models/production/` using its final-fold artifact — previous winners are evicted automatically.

| Model | Sklearn backend | Notes |
|---|---|---|
| **N-HiTS** | MLPClassifier | Multi-horizon; strong on seasonality decomposition |
| **Autoformer** | ExtraTreesClassifier | Strong on irregular patterns; class-balanced training |

> The sklearn backends are a pragmatic substitute until `neuralforecast` is installed. `torch` is already present (added for FinBERT sentiment scoring). All shared logic lives in `BaseModelWrapper` (`src/models/architectures/base.py`); each wrapper implements only `_build_model()`. Swapping in the real architecture requires replacing that one method — nothing else changes.

Ties in F1-macro are broken in the order N-HiTS → Autoformer.

### Quality gate

Evaluated after each training run against the production fold:

| Metric | Threshold |
|---|---|
| F1-macro | ≥ 0.40 |
| MCC | ≥ 0.10 |
| Hit rate | ≥ 0.52 |
| Max signal class share | ≤ 70% |

If the gate fails, the `retraining_required` flag stays set in `data/monitoring/status.json` and training is re-attempted on the next nightly run.

---

## Monitoring

Run nightly before the training decision. Results written to `reports/monitoring/monitoring_history.csv`.

| Check | Method | Trigger |
|---|---|---|
| Feature drift | KS test + PSI on continuous features | Both KS p-value < 0.05 **and** PSI ≥ 0.20 |
| Prediction drift | Chi-squared on signal distribution | p-value < 0.05, or any class ≥ 80% |
| Performance degradation | Rolling 21-day hit rate | Two consecutive windows below 45% |

Excluded from KS/PSI checks:
- **Calendrical** (`month`, `week_of_year`, `day_of_week`, `is_month_end`) — distributions shift with the calendar window, not regime change
- **Sentiment numeric columns** (`bullish_percent`, `bearish_percent`, `company_news_score`, `article_count`, `positive_insights`, `negative_insights`, `neutral_insights`) — null for all historical training data; KS/PSI on mostly-null columns produces meaningless signals
- **Boolean flags** (`close_outlier_flag`, `volume_outlier_flag`, `sentiment_available`) and **categorical** (`ticker_id`)

Alert files written to `data/monitoring/alerts/{date}.json`. Retraining state persisted in `data/monitoring/status.json`. Evidently HTML + JSON reports generated every 7 runs to `reports/monitoring/`.

---

## Security

- No secrets in code — use `.env` (never committed)
- Bearer token auth on all API endpoints; token must be ≥ 32 characters
- `pip-audit` run before any release; currently reports no known vulnerabilities
- No PII stored; signals are per-ticker, not per-user

---

## Disclaimer

This project is for private, non-commercial use only. It is not financial advice. Verify the terms of service for Massive (Polygon.io) on any plan changes.
