# Spec 03 — Feature Engineering

**Status:** Draft
**Created:** 2026-04-23
**Depends on:** Spec 02 (Preprocessing)
**Consumed by:** Spec 04 (Model Training)

---

## 1. Problem Statement

Transform processed OHLCV and sentiment data into a model-ready feature matrix per ticker. This layer computes technical indicators, lag features, return-based features, and the target label. Output is a single feature CSV per ticker, covering the full available history, ready for walk-forward model training and evaluation.

---

## 2. Scope

**In scope:**
- Daily log returns
- Trend indicators: SMA, EMA, MACD
- Volume indicators: OBV, VWAP ratio
- Lag features: close and volume, lags 1, 2, 3 days
- Seasonality features: day-of-week, week-of-year, month, is_month_end
- Sentiment passthrough features (with availability flag)
- Target label: BUY / HOLD / SELL via per-ticker percentile thresholding
- Lookahead bias guard: target computed from t+1 close, all features from ≤ t
- Feature-level null audit before write
- Persist feature matrix as CSV

**Out of scope:**
- Model-specific transformations or scaling beyond Spec 02
- Cross-ticker features or portfolio-level signals
- Momentum and volatility indicators (Spec 03 v2)

---

## 3. Input

**File:** `data/processed/{ticker}/{date}.csv`
**Schema:** `ProcessedRecord` (Spec 02)
Full historical window loaded per ticker, sorted ascending by `date`.

---

## 4. Output

**File:** `data/features/{ticker}/{date}.csv`
**One row per trading day** (rows with insufficient history for any feature are dropped)
**Schema model:** `FeatureRecord` (Pydantic)

### 4.1 Feature Columns

#### Returns
| Feature | Definition |
|---|---|
| `log_return` | `ln(close_t / close_t-1)` |
| `log_return_lag1` | `log_return` shifted 1 day |
| `log_return_lag2` | `log_return` shifted 2 days |
| `log_return_lag3` | `log_return` shifted 3 days |

#### Trend
| Feature | Definition | Params |
|---|---|---|
| `sma_10` | Simple moving average of close | window=10 |
| `sma_20` | Simple moving average of close | window=20 |
| `ema_10` | Exponential moving average of close | span=10 |
| `ema_20` | Exponential moving average of close | span=20 |
| `macd` | EMA(12) − EMA(26) | standard |
| `macd_signal` | EMA(9) of macd | standard |
| `macd_hist` | macd − macd_signal | |
| `close_to_sma20` | `close / sma_20 − 1` | ratio, mean-reversion signal |

#### Volume
| Feature | Definition | Params |
|---|---|---|
| `obv` | Cumulative OBV: `+volume` if close > prev_close, else `−volume` | running cumsum |
| `obv_lag1` | OBV shifted 1 day | |
| `vwap_ratio` | `close / vwap` | null if `vwap` is null |

#### Lags
| Feature | Definition |
|---|---|
| `close_lag1` | `close` shifted 1 day |
| `close_lag2` | `close` shifted 2 days |
| `close_lag3` | `close` shifted 3 days |
| `volume_lag1` | `volume` shifted 1 day |
| `volume_lag2` | `volume` shifted 2 days |
| `volume_lag3` | `volume` shifted 3 days |

#### Seasonality
| Feature | Definition | Encoding |
|---|---|---|
| `day_of_week` | 0=Monday … 4=Friday | int |
| `week_of_year` | ISO week number | int |
| `month` | 1–12 | int |
| `is_month_end` | Last trading day of calendar month | bool |

> Seasonality features are always included. Do not treat them as optional. Weekly and monthly cycles are empirically present in US equity volume and return patterns.

#### Sentiment Passthrough
| Feature | Source field | Notes |
|---|---|---|
| `bullish_percent` | `ProcessedRecord.bullish_percent` | null if unavailable |
| `bearish_percent` | `ProcessedRecord.bearish_percent` | null if unavailable |
| `company_news_score` | `ProcessedRecord.company_news_score` | null if unavailable |
| `buzz_weekly_average` | `ProcessedRecord.buzz_weekly_average` | null if unavailable |
| `sentiment_available` | `ProcessedRecord.sentiment_available` | bool; always present |

#### Outlier Flags Passthrough
| Feature | Source |
|---|---|
| `close_outlier_flag` | `ProcessedRecord.close_outlier_flag` |
| `volume_outlier_flag` | `ProcessedRecord.volume_outlier_flag` |

### 4.2 Target Label

| Column | Type | Values |
|---|---|---|
| `target` | `str` | `BUY`, `HOLD`, `SELL` |
| `forward_return` | `float` | `ln(close_t+1 / close_t)` — stored for evaluation, not used as model input |

**Definition:**

```
forward_return = ln(close_t+1 / close_t)

rolling_window = 60 trading days of forward_return per ticker

p30 = 30th percentile of rolling_window
p70 = 70th percentile of rolling_window

if forward_return > p70  → BUY
if forward_return < p30  → SELL
else                     → HOLD
```

**Lookahead bias rule:** `target` at row `t` uses `close_t+1`. All features at row `t` use data from `≤ t`. The final row of each ticker's history has no valid target — drop it from the feature file. `forward_return` is stored as a column but must never be used as a model input feature.

**Edge case:** If fewer than 60 days of forward return history are available, compute percentiles over available window (minimum 5 days). Below 5 days, drop the row entirely.

---

## 5. Lookahead Bias Guard

This is non-negotiable. Before writing the feature file, assert:

- No feature column references data from `t+1` or later
- `forward_return` column is present but excluded from the feature list exported to the model
- `target` is derived exclusively from `forward_return`
- OBV is computed as a running cumsum — verify it uses only past values at each step

Any violation must raise a `LookaheadBiasError` and halt the pipeline.

---

## 6. Null Audit

Before writing output, run a null audit across all feature columns:

- Log null count and null rate per column per ticker
- If any non-nullable feature (e.g. `log_return`, `sma_10`, `day_of_week`) has nulls beyond the warm-up period, raise a warning
- Rows where any core feature is null after the warm-up period are dropped and logged
- Sentiment nulls are expected and not flagged as warnings

**Warm-up period:** First 26 trading days per ticker are dropped from output (minimum history required for EMA(26) / MACD). This is the minimum row count to produce valid features — rows before this are structurally incomplete.

---

## 7. Module Structure

```
src/features/
├── __init__.py
├── returns.py          # compute_log_returns, compute_forward_return
├── trend.py            # compute_sma, compute_ema, compute_macd, compute_close_to_sma
├── volume.py           # compute_obv, compute_vwap_ratio
├── lags.py             # compute_lag_features
├── seasonality.py      # compute_seasonality_features
├── sentiment.py        # passthrough_sentiment
├── target.py           # compute_target_label (percentile-based, per ticker)
├── audit.py            # null_audit, lookahead_bias_guard
├── schema.py           # FeatureRecord (Pydantic)
└── pipeline.py         # FeaturePipeline — orchestrates full run per ticker
```

---

## 8. Processing Metadata

Each run writes `data/features/runs/{date}.json`:

```json
{
  "run_date": "2026-04-23",
  "tickers_processed": 48,
  "tickers_skipped": 0,
  "warmup_rows_dropped": 1248,
  "target_distribution": {
    "BUY": 0.30,
    "HOLD": 0.40,
    "SELL": 0.30
  },
  "null_rates": {
    "vwap_ratio": 0.03,
    "sentiment_fields": 0.12
  },
  "started_at": "2026-04-23T21:18:00Z",
  "completed_at": "2026-04-23T21:19:30Z"
}
```

Target distribution logged to detect label imbalance before training.

---

## 9. Acceptance Criteria

- [ ] All features computed exclusively from data at time `≤ t`
- [ ] `forward_return` present as column but absent from model feature list
- [ ] Final row per ticker (no valid `t+1`) dropped from output
- [ ] `LookaheadBiasError` raised if guard detects any violation
- [ ] First 26 rows per ticker dropped (MACD warm-up)
- [ ] Log returns computed as `ln(close_t / close_t-1)`; first row null (dropped)
- [ ] SMA(10), SMA(20), EMA(10), EMA(20) computed correctly per ticker
- [ ] MACD = EMA(12) − EMA(26); signal = EMA(9) of MACD; hist = MACD − signal
- [ ] OBV computed as running cumsum with correct sign logic; no lookahead
- [ ] `vwap_ratio` is null when `vwap` source field is null
- [ ] Lag features (close and volume, lags 1–3) computed with correct shift
- [ ] Seasonality features present on every row; no nulls
- [ ] Sentiment columns passed through; nulls accepted; `sentiment_available` always present
- [ ] Target label: BUY/HOLD/SELL assigned via 30th/70th percentile of 60-day rolling forward return per ticker
- [ ] Rows with fewer than 5 days of return history for percentile computation dropped
- [ ] Null audit runs before write; warnings logged for unexpected nulls in core features
- [ ] Output validated against `FeatureRecord` Pydantic model before write
- [ ] Existing feature files not overwritten (idempotent)
- [ ] `data/features/runs/{date}.json` written on every run with target distribution

---

## 10. Tests Required

| Test | Type | Notes |
|---|---|---|
| `test_log_return_computation` | Unit | Spot-check formula; first row null |
| `test_sma_ema_values` | Unit | Verify against known reference values |
| `test_macd_components` | Unit | MACD, signal, hist correct for synthetic series |
| `test_obv_sign_logic` | Unit | Close up → +volume; close down → −volume |
| `test_vwap_ratio_null_when_vwap_null` | Unit | vwap=None → vwap_ratio=None |
| `test_lag_features_shift` | Unit | close_lag1[t] == close[t-1] |
| `test_seasonality_completeness` | Unit | No nulls in any seasonality column |
| `test_target_buy_sell_hold_distribution` | Unit | Synthetic series; assert ~30/40/30 split |
| `test_target_uses_forward_return` | Unit | target[t] derived from close[t+1] |
| `test_lookahead_bias_guard_passes` | Unit | Valid feature set; no error raised |
| `test_lookahead_bias_guard_fails` | Unit | Inject t+1 feature; assert `LookaheadBiasError` |
| `test_warmup_rows_dropped` | Unit | First 26 rows absent from output |
| `test_final_row_dropped` | Unit | Last row (no t+1) absent from output |
| `test_null_audit_warns_on_core_nulls` | Unit | Unexpected null in `log_return`; warning logged |
| `test_null_audit_accepts_sentiment_nulls` | Unit | Null sentiment; no warning |
| `test_output_schema_valid` | Unit | All rows pass `FeatureRecord` validation |
| `test_idempotency` | Unit | Run twice same date; file not overwritten |
| `test_pipeline_full_run` | Integration | 5 mock tickers; assert feature files and metadata |
