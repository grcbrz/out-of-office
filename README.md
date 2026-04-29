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
├── Training         — walk-forward harness; LightGBM vs naive baseline
├── Evaluation       — F1-macro, MCC, Sharpe, confidence threshold calibration, quality gate
└── Prediction       — POST /predict via internal HTTP client (server auto-reloads after training)
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
# server reloads automatically via POST /reload after a quality-gate pass
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
| `/reload` | POST | Hot-reload the production artifact without restarting the server |

### Example predict request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["AAPL", "MSFT"], "predict_date": "2026-04-29"}'
```

```json
{
  "run_date": "2026-04-29",
  "model": "lightgbm",
  "predictions": [
    {
      "ticker": "AAPL",
      "signal": "BUY",
      "confidence": 0.61,
      "explanation": {
        "top_features": [
          {"feature": "log_return_lag1", "shap_value": 0.038},
          {"feature": "macd", "shap_value": 0.027},
          {"feature": "company_news_score", "shap_value": 0.019},
          {"feature": "obv_change", "shap_value": 0.014},
          {"feature": "close_zscore", "shap_value": -0.011}
        ],
        "explainer_used": "TreeExplainer"
      }
    }
  ],
  "warnings": []
}
```

Omit `tickers` to get signals for the full configured universe. Omit `predict_date` to default to today.

Per-prediction explanations use SHAP `TreeExplainer` (exact, no sampling) for LightGBM. Low-confidence signals (max class probability ≤ calibrated τ) are demoted to HOLD before being returned.

---

## Development

```bash
make test        # Run full test suite (290 tests)
make coverage    # Tests + coverage report (≥85%)
make lint        # ruff + mypy (zero errors)
make format      # black
make audit       # pip-audit security scan
make train       # Re-run training on existing data/features/
make notebook    # Launch Jupyter for EDA
```

### Tuning the walk-forward harness

The training harness is configured via `configs/training.yaml`:

```yaml
walk_forward:
  train_window: 120   # trading days per fold (120 ≈ 6 months)
  step_size: 20       # trading days between fold starts
  min_folds: 3        # abort if fewer folds are available
```

**Key trade-off:** a shorter `train_window` fits recent market regime tighter (better in-sample generalisation on a small universe) but increases variance. A longer window sees more history but can degrade when the model trains on stale patterns.

Empirically, `train_window=120 / step_size=20` outperforms `252/21` on this 15-ticker universe (F1 0.42 vs 0.37). If you add more tickers or backfill more history, try larger values.

To retrain immediately with the current config:

```bash
make train
```

To retrain and force-evaluate against the quality gate:

```bash
make nightly START_DATE=<YYYY-MM-DD>
# or with --force-retrain to bypass the 21-day cadence check:
.venv/bin/python scripts/run_nightly.py --force-retrain
```

### Project structure

```
src/
├── ingestion/      — Massive OHLCV + Polygon news sentiment (FinBERT-scored), rate limiter, pipeline
├── preprocessing/  — Imputation, outlier detection, normalisation, merger
├── features/       — Returns, trend, volume, lags, seasonality, target label
├── models/         — Walk-forward harness, LightGBM + naive baseline
├── evaluation/     — Classification metrics, financial metrics, quality gate, SHAP
├── serving/        — FastAPI app, auth, artifact loader, inference engine
└── monitoring/     — Feature drift, prediction drift, degradation, retraining trigger

tests/
├── unit/           — Per-module unit tests (mocked external dependencies)
├── integration/    — Pipeline-level tests
└── acceptance/     — Spec-level behaviour tests

configs/
├── training.yaml       — Walk-forward harness settings (train_window, step_size)
├── ingestion.yaml      — Polygon rate limits, fixed universe ticker list
├── evaluation.yaml     — Quality gate thresholds (F1, MCC, hit rate)
├── monitoring.yaml     — Drift detection thresholds
└── models/             — Per-model hyperparameters (lightgbm.yaml)
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

Each nightly training run evaluates **LightGBM** against a **naive last-direction baseline** using walk-forward cross-validation (configurable via `configs/training.yaml`; default 120-day train window, 20-day step, minimum 3 folds). LightGBM is the sole production candidate — the baseline exists only as a quality floor that the production model must beat.

| Role | Implementation | Notes |
|---|---|---|
| **Production candidate** | `lightgbm.LGBMClassifier` | Multi-class (SELL/HOLD/BUY); exact SHAP via TreeExplainer; handles mixed-scale features natively |
| **Naive baseline** | Last-direction rule (percentile thresholds on `log_return_lag1`) | Never promoted to production; sets the minimum acceptable F1 delta |

The production model is selected by **mean F1-macro across all folds**, written to `models/production/lightgbm/`, and previous artifacts are evicted automatically.

### Confidence thresholding

After each fold, a confidence threshold τ is calibrated by grid-searching validation Sharpe over candidate values (0.34–0.70). At inference time, any prediction whose top class probability ≤ τ is demoted to HOLD. This concentrates trading on high-conviction signals.

τ is stored in the artifact metadata and loaded by the server — no manual configuration required.

### Quality gate

Evaluated after each training run against the production fold. Training must pass **all** absolute floors **and** beat the baseline:

| Check | Threshold |
|---|---|
| F1-macro (absolute) | ≥ 0.40 |
| MCC | ≥ 0.10 |
| Hit rate | ≥ 0.52 |
| Max signal class share | ≤ 70% |
| F1-macro delta over baseline | ≥ 0.02 |

If the gate fails, the `retraining_required` flag stays set in `data/monitoring/status.json` and training is re-attempted on the next nightly run. The server is **not** reloaded on a gate failure — the previous production artifact continues serving.

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
