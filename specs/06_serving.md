# Spec 06 — Serving

**Status:** Draft
**Created:** 2026-04-23
**Depends on:** Spec 05 (Evaluation & Explainability)
**Consumed by:** Spec 07 (Monitoring & Retraining)

---

## 1. Problem Statement

Expose next-day BUY/HOLD/SELL signals via a FastAPI REST endpoint for private local use. The serving layer loads the production model artifact, runs inference on the latest feature data, attaches SHAP-based explanations, and persists predictions to CSV. Prediction is triggered automatically at the end of the nightly batch pipeline after training completes and quality gates pass.

---

## 2. Scope

**In scope:**
- FastAPI application with `/predict`, `/health`, `/metrics` endpoints
- Production model loading from `models/production/`
- Inference on latest feature data for all 50 tickers
- Per-prediction explanation (SHAP top-5 + attention weights)
- Request validation via Pydantic
- Bearer token authentication (local, static token)
- Prediction persistence to CSV
- Automatic trigger from nightly batch pipeline
- Graceful error handling and structured logging

**Out of scope:**
- Streamlit dashboard (future increment)
- Multi-user access or external deployment
- Real-time streaming or WebSocket endpoints
- Model retraining from serving layer

---

## 3. Endpoints

### 3.1 `POST /predict`

Run inference for one or more tickers using the production model and latest available features.

**Request:**
```json
{
  "tickers": ["AAPL", "MSFT"],
  "date": "2026-04-24"
}
```

- `tickers`: list of 1–50 valid ticker strings; if omitted, defaults to full universe
- `date`: ISO date string for which to generate next-day signal; defaults to today

**Response:**
```json
{
  "run_date": "2026-04-24",
  "model": "nhits",
  "predictions": [
    {
      "ticker": "AAPL",
      "signal": "BUY",
      "confidence": 0.71,
      "explanation": {
        "top_features": [
          {"feature": "close_zscore", "shap_value": 0.18},
          {"feature": "macd_hist", "shap_value": 0.12},
          {"feature": "bullish_percent", "shap_value": 0.09},
          {"feature": "close_lag1", "shap_value": -0.07},
          {"feature": "volume_zscore", "shap_value": 0.06}
        ],
        "attention_weights": [0.02, 0.03, 0.15],
        "explainer_used": "DeepExplainer"
      }
    }
  ],
  "warnings": []
}
```

- `confidence`: softmax probability of predicted class
- `attention_weights`: null for N-HiTS
- `warnings`: list of non-fatal issues (e.g. sentiment unavailable for a ticker)

**Error responses:**

| Status | Condition |
|---|---|
| 400 | Invalid ticker, invalid date format, date in future |
| 401 | Missing or invalid bearer token |
| 422 | Pydantic validation failure on request body |
| 503 | No production model artifact found |
| 500 | Inference failure — logged, structured error returned |

### 3.2 `GET /health`

Returns service status and production model metadata.

```json
{
  "status": "ok",
  "model": "nhits",
  "model_date": "2026-04-23",
  "production_fold_f1_macro": 0.41,
  "quality_gate_passed": true
}
```

Returns `503` with `"status": "degraded"` if no valid production artifact is loaded.

### 3.3 `GET /metrics`

Returns prediction statistics since last model load.

```json
{
  "total_predictions": 150,
  "signal_distribution": {"BUY": 0.31, "HOLD": 0.38, "SELL": 0.31},
  "tickers_served": 50,
  "last_prediction_at": "2026-04-24T06:00:12Z",
  "model": "nhits"
}
```

In-memory counters only — reset on server restart. Persisted signal history is in the prediction CSV.

---

## 4. Authentication

Static bearer token loaded from environment variable `API_TOKEN`.

```
Authorization: Bearer <token>
```

- Applied to all endpoints including `/health` and `/metrics`
- Token validated on every request via FastAPI dependency
- If `API_TOKEN` is not set at startup, server refuses to start and logs a critical error
- Token must be ≥ 32 characters (enforced at startup)

---

## 5. Model Loading

On startup, the server loads the production artifact from `models/production/`:

```
models/production/
└── {model_name}/
    ├── model.pt
    ├── config.yaml
    ├── imputation_params.json
    ├── ticker_map.json
    ├── class_weights.json
    └── metadata.json
```

- Model loaded once at startup, held in memory
- If artifact directory is missing or incomplete, server starts in degraded mode — `/health` returns `503`, `/predict` returns `503`
- Hot reload not supported in v1 — restart server after retraining to load new model

---

## 6. Inference Pipeline

Per `/predict` request:

```
1. Validate request (Pydantic)
2. Authenticate bearer token
3. Load latest feature rows for requested tickers from data/features/{ticker}/{date}.csv
4. Apply imputation (using params from artifact — no recomputation)
5. Encode tickers via ticker_map
6. Run model forward pass → softmax probabilities → argmax → signal
7. Compute SHAP explanation (DeepExplainer or KernelExplainer fallback)
8. Extract attention weights if applicable
9. Build response
10. Persist prediction to data/predictions/{date}.csv
11. Update in-memory metrics counters
12. Return response
```

Steps 6–8 run per ticker. SHAP is computed individually — not batched — to produce per-prediction explanations.

---

## 7. Prediction Persistence

**File:** `data/predictions/{date}.csv`
Appended on every `/predict` call. Never overwritten.

| Column | Type | Notes |
|---|---|---|
| `run_date` | `date` | |
| `ticker` | `str` | |
| `signal` | `str` | BUY / HOLD / SELL |
| `confidence` | `float` | softmax probability |
| `model` | `str` | model architecture name |
| `top_feature_1` … `top_feature_5` | `str` | feature name |
| `shap_1` … `shap_5` | `float` | SHAP value |
| `explainer_used` | `str` | DeepExplainer / KernelExplainer |
| `sentiment_available` | `bool` | |
| `predicted_at` | `datetime` | UTC |

---

## 8. Nightly Batch Integration

The nightly batch pipeline (orchestrated via `scripts/run_nightly.py`) calls the serving layer automatically after training and evaluation complete:

```
Nightly batch sequence:
1. scripts/run_nightly.py --start-date {date}
   ├── IngestionPipeline.run()
   ├── PreprocessingPipeline.run()
   ├── FeaturePipeline.run()
   ├── TrainingPipeline.run()         ← retrains if it is a retraining day (every 21 days)
   ├── EvaluationPipeline.run()       ← skipped if training skipped
   └── PredictionClient.run()         ← calls POST /predict for full universe
```

`PredictionClient` is a thin internal HTTP client that calls the running FastAPI server. The server must already be running before the nightly batch starts (launched as a background process or system service).

**Retraining cadence:** Training and evaluation run every 21 trading days (Spec 04). On non-retraining nights, ingestion → preprocessing → features → prediction runs only.

---

## 9. Structured Logging

All requests and inference steps logged as structured JSON to stdout:

```json
{
  "timestamp": "2026-04-24T06:00:05Z",
  "level": "INFO",
  "event": "prediction_complete",
  "ticker": "AAPL",
  "signal": "BUY",
  "confidence": 0.71,
  "latency_ms": 312,
  "explainer": "DeepExplainer"
}
```

Log levels: `DEBUG` (feature loading), `INFO` (predictions, startup), `WARNING` (sentiment missing, fallback explainer), `ERROR` (inference failure), `CRITICAL` (model not found, token not set).

---

## 10. Module Structure

```
src/serving/
├── __init__.py
├── app.py              # FastAPI app — router registration, lifespan
├── routers/
│   ├── __init__.py
│   ├── predict.py      # POST /predict
│   ├── health.py       # GET /health
│   └── metrics.py      # GET /metrics
├── auth.py             # bearer token dependency
├── inference.py        # InferenceEngine — model forward pass, signal decoding
├── explainer.py        # ServingExplainer — SHAP + attention at inference time
├── loader.py           # ArtifactLoader — load model + metadata from disk
├── persistence.py      # append_prediction_csv
├── schemas.py          # PredictRequest, PredictResponse, PredictionRecord (Pydantic)
└── metrics_store.py    # in-memory counters for /metrics endpoint
```

```
scripts/
├── run_nightly.py      # nightly batch orchestrator
└── prediction_client.py  # internal HTTP client for automated prediction trigger
```

---

## 11. Configuration

```yaml
# configs/serving.yaml
host: 127.0.0.1
port: 8000
log_level: info

model:
  artifact_dir: models/production

predictions:
  output_dir: data/predictions
  default_universe: all     # or list of tickers

security:
  token_min_length: 32
```

`API_TOKEN` loaded exclusively from environment — never in config files.

---

## 12. Acceptance Criteria

- [ ] `POST /predict` returns BUY/HOLD/SELL signal with confidence and SHAP explanation for each requested ticker
- [ ] `GET /health` returns model metadata and quality gate status; `503` if no artifact loaded
- [ ] `GET /metrics` returns signal distribution and prediction count since last restart
- [ ] All endpoints reject requests without valid bearer token (`401`)
- [ ] Server refuses to start if `API_TOKEN` env var is missing or < 32 characters
- [ ] Server starts in degraded mode (503 on predict/health) if production artifact is missing or incomplete
- [ ] Pydantic validation rejects malformed requests (`422`)
- [ ] Future dates rejected with `400`
- [ ] Unknown tickers rejected with `400`
- [ ] Predictions appended to `data/predictions/{date}.csv`; file never overwritten
- [ ] Imputation uses artifact params — no recomputation at inference time
- [ ] SHAP computed per ticker; fallback to KernelExplainer logged in response and CSV
- [ ] Attention weights null for N-HiTS; populated for PatchTST and Autoformer
- [ ] Nightly batch calls `/predict` automatically after evaluation completes
- [ ] On non-retraining nights, pipeline skips training/evaluation and runs prediction only
- [ ] All requests logged as structured JSON with latency

---

## 13. Tests Required

| Test | Type | Notes |
|---|---|---|
| `test_predict_valid_request` | Unit | Mock model; assert signal + explanation structure |
| `test_predict_missing_token` | Unit | No auth header → 401 |
| `test_predict_invalid_token` | Unit | Wrong token → 401 |
| `test_predict_future_date` | Unit | Date > today → 400 |
| `test_predict_unknown_ticker` | Unit | Unknown ticker → 400 |
| `test_predict_invalid_body` | Unit | Malformed JSON → 422 |
| `test_predict_no_artifact` | Unit | Missing production dir → 503 |
| `test_health_ok` | Unit | Artifact loaded → 200 with metadata |
| `test_health_degraded` | Unit | No artifact → 503 |
| `test_metrics_counters` | Unit | Two predict calls; assert count = 2 |
| `test_token_too_short_startup` | Unit | Token < 32 chars → startup error |
| `test_imputation_uses_artifact_params` | Unit | Assert inference uses stored median, not recomputed |
| `test_prediction_appended_to_csv` | Unit | Two calls → two rows in CSV |
| `test_csv_not_overwritten` | Unit | Existing file; new call appends |
| `test_nightly_batch_sequence` | Integration | Mock all pipelines; assert call order |
| `test_nightly_skips_training_on_non_retraining_day` | Integration | Day 10 of cycle; training not called |
| `test_structured_log_output` | Unit | Assert log event keys present on prediction |
