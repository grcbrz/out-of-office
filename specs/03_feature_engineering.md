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

> **Stationarity rule.** Every model-input feature must be in return-scale, ratio,
> z-score, or bounded categorical form. Absolute price-level columns (raw SMA, EMA,
> MACD, OBV, close lags, volume lags) are intermediate computations only — they
> must not appear in `FEATURE_COLUMNS`. Rationale: a global model trained across
> 50 tickers cannot learn from features whose scale is dominated by ticker
> identity.

#### Trend (return-scale only)
| Feature | Definition | Params |
|---|---|---|
| `close_to_sma_10` | `close / sma_10 − 1` | mean-reversion signal vs short SMA |
| `close_to_sma_20` | `close / sma_20 − 1` | mean-reversion signal vs medium SMA |
| `close_to_ema_10` | `close / ema_10 − 1` | mean-reversion signal vs short EMA |
| `close_to_ema_20` | `close / ema_20 − 1` | mean-reversion signal vs medium EMA |
| `macd_norm` | `macd / close` | MACD divided by close — return-scale |
| `macd_signal_norm` | `macd_signal / close` | |
| `macd_hist_norm` | `macd_hist / close` | |

`sma_*`, `ema_*`, `macd`, `macd_signal`, `macd_hist` are computed as intermediate
columns to derive the features above. They are **not** persisted in
`FeatureRecord` and **not** in `FEATURE_COLUMNS`.

#### Volume (return-scale only)
| Feature | Definition | Params |
|---|---|---|
| `obv_pct_change_20` | `obv.pct_change(20)` | 20-day OBV pct change — stationary derivative |
| `volume_log_ratio_20` | `ln(volume / volume.rolling(20).mean())` | volume vs its own 20-day mean |
| `vwap_ratio` | `close / vwap` | null if `vwap` is null |

`obv` is computed internally as a cumulative cumsum to derive `obv_pct_change_20`,
then dropped from output.

#### Volatility & Momentum (new)
| Feature | Definition | Params |
|---|---|---|
| `realised_vol_20` | `log_return.rolling(20).std()` | rolling realised volatility |
| `momentum_20` | `log_return.rolling(20).sum()` | cumulative log-return over 20 days |
| `atr_14` | `ATR(14) / close` | Average True Range divided by close |
| `log_return_zscore_60` | `(log_return − rolling_mean_60) / rolling_std_60` | min_periods=20 |

#### Lags (returns only — no price-level lags)
| Feature | Definition |
|---|---|
| `log_return_lag1` | `log_return` shifted 1 day |
| `log_return_lag2` | `log_return` shifted 2 days |
| `log_return_lag3` | `log_return` shifted 3 days |

`close_lag*` and `volume_lag*` removed — non-stationary, redundant with
`log_return_lag*` and `volume_log_ratio_20`.

#### Seasonality (cyclic encoded)
| Feature | Definition | Encoding |
|---|---|---|
| `dow_sin` | `sin(2π · dayofweek / 5)` | float, trading week period |
| `dow_cos` | `cos(2π · dayofweek / 5)` | float |
| `month_sin` | `sin(2π · month / 12)` | float, yearly period |
| `month_cos` | `cos(2π · month / 12)` | float |
| `is_month_end` | Last trading day of calendar month | bool |

Cyclic encoding (sin/cos pair) avoids treating Monday=0 and Friday=4 as ordinal
distance "4". Tree models can split on the pair; linear and neural models can
learn the cycle natively. `week_of_year`, raw `day_of_week` and raw `month`
removed — redundant under cyclic encoding.

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

**Warm-up period:** First **60 trading days** per ticker are dropped from output (minimum history required for `log_return_zscore_60`, the longest-window feature). EMA(26) / MACD warmup is shorter (26 days) but no longer the binding constraint.

---

## 7. Module Structure

```
src/features/
├── __init__.py
├── returns.py          # compute_log_returns, compute_forward_return
├── trend.py            # compute_sma, compute_ema, compute_macd (intermediate)
│                       # compute_close_to_ma_ratios, compute_macd_normalised (model inputs)
├── volume.py           # compute_obv (intermediate), compute_obv_pct_change,
│                       # compute_volume_log_ratio, compute_vwap_ratio
├── volatility.py       # compute_realised_volatility, compute_momentum,
│                       # compute_atr, compute_return_zscore
├── lags.py             # compute_log_return_lags
├── seasonality.py      # compute_seasonality_features (cyclic encoded)
├── sentiment.py        # passthrough_sentiment
├── target.py           # compute_target_label (percentile-based, per ticker)
├── audit.py            # null_audit, lookahead_bias_guard
├── schema.py           # FeatureRecord (Pydantic), FEATURE_COLUMNS
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
- [ ] `forward_return` present as column but absent from `FEATURE_COLUMNS`
- [ ] No absolute price-level column appears in `FEATURE_COLUMNS` (stationarity rule)
- [ ] Final row per ticker (no valid `t+1`) dropped from output
- [ ] `LookaheadBiasError` raised if guard detects any violation
- [ ] First 60 rows per ticker dropped (`log_return_zscore_60` warm-up)
- [ ] Log returns computed as `ln(close_t / close_t-1)`; first row null (dropped)
- [ ] `close_to_sma_10/20`, `close_to_ema_10/20` computed correctly per ticker
- [ ] `macd_norm`, `macd_signal_norm`, `macd_hist_norm` = MACD components / close
- [ ] `obv_pct_change_20` is `obv.pct_change(20)`; raw `obv` not in output
- [ ] `volume_log_ratio_20` = `ln(volume / 20-day mean volume)`
- [ ] `realised_vol_20`, `momentum_20`, `atr_14`, `log_return_zscore_60` present
- [ ] `vwap_ratio` is null when `vwap` source field is null
- [ ] `log_return_lag1/2/3` computed with correct shift; no `close_lag*` or `volume_lag*`
- [ ] Cyclic seasonality (`dow_sin/cos`, `month_sin/cos`, `is_month_end`) present on every row
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
| `test_close_to_ma_ratios_returns_scale` | Unit | `close_to_sma_20[t] == close[t]/sma_20[t] - 1` |
| `test_macd_normalised_returns_scale` | Unit | `macd_norm == macd / close` for known synthetic series |
| `test_obv_pct_change_no_cumulative_in_output` | Unit | `obv` absent from FEATURE_COLUMNS; `obv_pct_change_20` present |
| `test_volume_log_ratio` | Unit | `volume_log_ratio_20 == ln(volume / rolling_mean_20)` |
| `test_realised_volatility_correct` | Unit | Against `log_return.rolling(20).std()` reference |
| `test_momentum_correct` | Unit | Against `log_return.rolling(20).sum()` reference |
| `test_atr_normalised_by_close` | Unit | `atr_14[t] == TR.rolling(14).mean()[t] / close[t]` |
| `test_return_zscore_correct` | Unit | Hand-computed reference on small series |
| `test_vwap_ratio_null_when_vwap_null` | Unit | vwap=None → vwap_ratio=None |
| `test_log_return_lags_shift` | Unit | `log_return_lag1[t] == log_return[t-1]` |
| `test_seasonality_cyclic_completeness` | Unit | dow_sin/cos, month_sin/cos in [-1,1]; no nulls |
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
