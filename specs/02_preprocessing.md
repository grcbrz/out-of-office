# Spec 02 — Preprocessing

**Status:** Draft
**Created:** 2026-04-23
**Depends on:** Spec 01 (Data Ingestion)
**Consumed by:** Spec 03 (Feature Engineering)

---

## 1. Problem Statement

Transform raw OHLCV and sentiment CSVs into clean, validated, normalised datasets ready for feature engineering. This layer enforces dtype contracts, handles missing values, detects and flags outliers, applies z-score normalisation per ticker, and produces a single merged processed file per ticker per run date. Raw files are never modified.

---

## 2. Scope

**In scope:**
- Load and dtype-cast raw CSVs (OHLCV + sentiment)
- Re-validate schema via Pydantic on load (CSV has no native type safety)
- Missing value detection and imputation strategy per field
- Outlier detection and flagging (OHLCV price/volume)
- Z-score normalisation per ticker over 60-trading-day rolling window
- Sentiment null handling
- Merge OHLCV and sentiment into single processed record per ticker per date
- Persist processed output as CSV
- Processing metadata logging

**Out of scope:**
- Feature engineering (Spec 03)
- Technical indicators
- Model-specific transformations
- Any modification of raw files

---

## 3. Inputs

| File pattern | Source spec | Schema model |
|---|---|---|
| `data/raw/ohlcv/{ticker}/{date}.csv` | Spec 01 | `OHLCVRecord` |
| `data/raw/sentiment/{ticker}/{date}.csv` | Spec 01 | `SentimentRecord` |
| `data/raw/universe/{date}.csv` | Spec 01 | ticker list |

Processing is triggered per ticker in the resolved universe for the given run date.

---

## 4. Output

**File:** `data/processed/{ticker}/{date}.csv`
**One row per trading day** (historical window + current date)
**Schema model:** `ProcessedRecord` (Pydantic)

| Field | Type | Notes |
|---|---|---|
| `ticker` | `str` | |
| `date` | `date` | |
| `open` | `float` | raw, unscaled |
| `high` | `float` | raw, unscaled |
| `low` | `float` | raw, unscaled |
| `close` | `float` | raw, unscaled |
| `volume` | `int` | raw, unscaled |
| `vwap` | `float \| None` | raw, unscaled |
| `close_zscore` | `float \| None` | z-score over 60-day rolling window |
| `volume_zscore` | `float \| None` | z-score over 60-day rolling window |
| `close_outlier_flag` | `bool` | True if `abs(close_zscore) > 3.0` |
| `volume_outlier_flag` | `bool` | True if `abs(volume_zscore) > 3.0` |
| `bullish_percent` | `float \| None` | from sentiment; null if unavailable |
| `bearish_percent` | `float \| None` | from sentiment; null if unavailable |
| `company_news_score` | `float \| None` | from sentiment; null if unavailable |
| `buzz_weekly_average` | `float \| None` | from sentiment; null if unavailable |
| `sentiment_available` | `bool` | False if all sentiment fields are null |
| `is_trading_day` | `bool` | False for holidays/weekends (from Spec 01 calendar) |
| `imputed_close` | `bool` | True if close was forward-filled |
| `imputed_volume` | `bool` | True if volume was zero-filled |

Raw price/volume fields are always preserved unscaled. Scaled values are additive columns.

---

## 5. Dtype Casting on Load

CSV round-trip loses dtype information. On load, cast explicitly before any processing:

| Column | Target dtype | Cast failure action |
|---|---|---|
| `date` | `datetime.date` | reject row, log |
| `open`, `high`, `low`, `close`, `vwap` | `float` | reject row, log |
| `volume` | `int` | reject row, log |
| `bullish_percent`, `bearish_percent` | `float \| None` | set to `None`, log |
| `company_news_score`, `buzz_weekly_average` | `float \| None` | set to `None`, log |

Casting is performed inside a Pydantic validator — not as ad-hoc `pd.to_numeric` calls scattered through the pipeline.

---

## 6. Missing Value Strategy

### OHLCV

| Field | Strategy | Rationale |
|---|---|---|
| `close` | Forward-fill up to 2 consecutive days; beyond that reject | Handles occasional data gaps; avoids stale prices propagating too far |
| `open`, `high`, `low` | Forward-fill alongside `close` | Consistency within a row |
| `volume` | Fill with 0; set `imputed_volume = True` | Missing volume is assumed zero-trade day, not a data error |
| `vwap` | Leave as `None` | Optional field; downstream handles nulls |

Forward-fill respects trading calendar — gaps on non-trading days are not counted toward the 2-day limit.

### Sentiment

All sentiment fields are nullable by design (Spec 01). No imputation applied. Set `sentiment_available = False` when all four fields are null for a given ticker-date. Downstream features must handle this flag.

---

## 7. Outlier Detection

**Method:** Rolling z-score over a 60-trading-day window, computed per ticker.

```
z = (x - rolling_mean(60)) / rolling_std(60)
```

**Threshold:** `abs(z) > 3.0` → set `{field}_outlier_flag = True`

**Applied to:** `close`, `volume`

**Action:** Flag and keep. Records are not removed or clipped. The model receives both the raw value and the flag — it decides how to handle it.

**Edge case:** If fewer than 60 trading days of history are available (e.g. early in backfill), compute z-score over available window. If fewer than 5 days available, set `close_zscore = None` and `volume_zscore = None` — do not produce a spurious z-score from insufficient data.

---

## 8. Z-Score Normalisation

**Scope:** `close` and `volume` only in the preprocessing layer. Additional features normalised in Spec 03 as needed.

**Window:** 60 trading days, expanding from day 5 (minimum), rolling thereafter.

**Per-ticker:** Parameters (mean, std) computed independently per ticker. No cross-ticker normalisation.

**Std = 0 edge case:** If rolling std is zero (e.g. trading halted, repeated identical prices), set zscore to `0.0` and log a warning. Do not divide by zero.

**Persistence:** Rolling stats (mean, std per ticker per date) are **not** persisted separately in this spec. They are recomputed from the processed window on each run. If caching becomes a performance concern, address in a future spec.

---

## 9. Merge Logic

For each ticker, join OHLCV and sentiment on `(ticker, date)`:

- **Join type:** Left join on OHLCV — every trading day with OHLCV data gets a row
- **Sentiment missing:** If no sentiment row exists for a date, all sentiment fields set to `None`, `sentiment_available = False`
- **No OHLCV row:** Cannot occur by construction (OHLCV drives the join)

Output is sorted ascending by `date`.

---

## 10. Module Structure

```
src/preprocessing/
├── __init__.py
├── loader.py           # load_ohlcv, load_sentiment — CSV load + dtype cast
├── validator.py        # ProcessedRecord (Pydantic), row-level validation
├── imputer.py          # forward_fill_close, fill_volume — missing value logic
├── outlier.py          # flag_outliers — rolling z-score flagging
├── normaliser.py       # compute_zscore — rolling z-score normalisation
├── merger.py           # merge_ohlcv_sentiment — left join logic
└── pipeline.py         # PreprocessingPipeline — orchestrates full run per ticker
```

---

## 11. Processing Metadata

Each run appends to `data/processed/runs/{date}.json`:

```json
{
  "run_date": "2026-04-23",
  "tickers_processed": 48,
  "tickers_skipped": 2,
  "skipped_detail": [
    {"ticker": "TICKER_A", "reason": "no raw OHLCV file found"},
    {"ticker": "TICKER_B", "reason": "forward-fill exceeded 2-day limit on 3 rows"}
  ],
  "outliers_flagged": {
    "close": 4,
    "volume": 7
  },
  "imputed_rows": {
    "close": 2,
    "volume": 11
  },
  "started_at": "2026-04-23T21:15:00Z",
  "completed_at": "2026-04-23T21:17:45Z"
}
```

---

## 12. Acceptance Criteria

- [ ] Raw CSV files are never modified
- [ ] Dtypes are explicitly cast on load via Pydantic; cast failures reject the row and log field context
- [ ] `close` forward-filled up to 2 consecutive trading days; rows exceeding limit are rejected and logged
- [ ] `volume` zero-filled; `imputed_volume = True` set on affected rows
- [ ] `close_zscore` and `volume_zscore` computed over 60-trading-day rolling window per ticker
- [ ] Z-score is `None` when fewer than 5 trading days of history are available
- [ ] `close_outlier_flag` and `volume_outlier_flag` set to `True` when `abs(zscore) > 3.0`
- [ ] Outlier-flagged records are kept, not removed or clipped
- [ ] Rolling std = 0 sets zscore to `0.0` and logs a warning — no division by zero
- [ ] Sentiment joined via left join on OHLCV; missing sentiment produces `sentiment_available = False`
- [ ] `ProcessedRecord` Pydantic model validates every output row before write
- [ ] Processed CSV written to `data/processed/{ticker}/{date}.csv`; existing files not overwritten (idempotent)
- [ ] `data/processed/runs/{date}.json` written on every run

---

## 13. Tests Required

| Test | Type | Notes |
|---|---|---|
| `test_dtype_cast_valid` | Unit | All fields cast correctly from raw CSV strings |
| `test_dtype_cast_failure_rejects_row` | Unit | Bad dtype on `close`; row rejected, others pass |
| `test_forward_fill_within_limit` | Unit | 1–2 day gap filled; `imputed_close = True` |
| `test_forward_fill_exceeds_limit` | Unit | 3-day gap; row rejected, logged |
| `test_volume_zero_fill` | Unit | Missing volume → 0, flag set |
| `test_zscore_sufficient_history` | Unit | ≥ 60 days; correct mean/std used |
| `test_zscore_insufficient_history` | Unit | < 5 days; zscore is `None` |
| `test_zscore_std_zero` | Unit | Constant prices; zscore = 0.0, warning logged |
| `test_outlier_flag_set` | Unit | `abs(z) > 3.0`; flag True, record kept |
| `test_outlier_record_not_removed` | Unit | Flagged record present in output |
| `test_sentiment_merge_present` | Unit | Matching sentiment row; fields populated |
| `test_sentiment_merge_missing` | Unit | No sentiment row; all null, `sentiment_available = False` |
| `test_output_schema_valid` | Unit | Output rows pass `ProcessedRecord` validation |
| `test_idempotency` | Unit | Run twice same date; file not overwritten |
| `test_pipeline_full_run` | Integration | Mock raw files for 5 tickers; assert output files and metadata |
