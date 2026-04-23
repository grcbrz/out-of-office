# Spec 01 — Data Ingestion

**Status:** Draft
**Created:** 2026-04-23
**Depends on:** None
**Consumed by:** Spec 02 (Preprocessing)

---

## 1. Problem Statement

Nightly batch job that fetches, validates, and persists raw market and sentiment data for the top 50 US stocks by volume. Must run reliably after market close, tolerate partial API failures gracefully, and produce immutable raw files as the single source of truth for all downstream processing.

---

## 2. Scope

**In scope:**
- Universe resolution (top 50 by volume via Polygon)
- OHLCV ingestion per ticker (Polygon)
- Sentiment score ingestion per ticker (Finnhub)
- Schema validation at ingestion boundary
- Rate limiting for both APIs
- Retry logic with partial failure handling and alerting
- Raw file persistence (CSV)
- Run metadata and lineage logging

**Out of scope:**
- Data cleaning or normalisation (Spec 02)
- Feature engineering (Spec 03)
- Any model input preparation

---

## 3. Data Sources

| Source | Purpose | Rate limit | Auth |
|---|---|---|---|
| Polygon.io REST API (free) | OHLCV + universe resolution | 5 calls/min | `POLYGON_API_KEY` env var |
| Finnhub REST API (free) | News sentiment score per ticker | 60 calls/min | `FINNHUB_API_KEY` env var |

Both keys loaded exclusively from environment variables via `python-dotenv`. Never hardcoded.

---

## 4. Universe Resolution

**Endpoint:** `GET /v2/aggs/grouped/locale/us/market/stocks/{date}`
**Logic:**
1. Fetch previous trading day's grouped daily aggregates
2. Filter to common stocks (exclude ETFs, warrants, preferred)
3. Sort descending by volume (`v` field)
4. Take top 50 tickers
5. Persist universe list to `data/raw/universe/{date}.csv`

**Counts as 1 API call.** Must complete before per-ticker fetches begin.

**Edge case:** If the grouped endpoint returns < 50 valid tickers (e.g. holiday, data gap), log a warning and proceed with what is available. Do not fail the run.

---

## 5. OHLCV Ingestion

**Endpoint:** `GET /v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}`
**Parameters:** `adjusted=true`, `sort=asc`, `limit=730` (2 years daily)
**Called once per ticker on first run; incremental (last date + 1 day) on subsequent runs.**

**Schema contract (Pydantic model `OHLCVRecord`):**

| Field | Type | Constraints |
|---|---|---|
| `ticker` | `str` | non-empty, uppercase |
| `date` | `date` | no future dates |
| `open` | `float` | > 0 |
| `high` | `float` | >= open |
| `low` | `float` | > 0, <= open |
| `close` | `float` | > 0 |
| `volume` | `int` | >= 0 |
| `vwap` | `float \| None` | > 0 if present |

Validation failures: log field-level error, reject the record, continue. Do not write corrupt records to raw storage.

**Output:** `data/raw/ohlcv/{ticker}/{date}.csv` (one file per ticker per run date)

> **Note:** CSV has no native dtype enforcement. Pydantic validation on write is the schema contract. Downstream readers must re-cast dtypes explicitly on load.

---

## 6. Sentiment Ingestion

**Endpoint:** `GET /news-sentiment?symbol={ticker}`
**Returns:** Pre-computed sentiment score (`buzz`, `sentiment.bullishPercent`, `sentiment.bearishPercent`, `companyNewsScore`)

**Schema contract (Pydantic model `SentimentRecord`):**

| Field | Type | Constraints |
|---|---|---|
| `ticker` | `str` | non-empty, uppercase |
| `date` | `date` | ingestion date |
| `bullish_percent` | `float \| None` | 0.0–1.0 if present |
| `bearish_percent` | `float \| None` | 0.0–1.0 if present |
| `company_news_score` | `float \| None` | >= 0 if present |
| `buzz_weekly_average` | `float \| None` | >= 0 if present |

All fields are nullable. Missing sentiment for a ticker is a valid state — not a pipeline error.

**Output:** `data/raw/sentiment/{ticker}/{date}.csv`

---

## 7. Rate Limiting

A shared `RateLimiter` utility (token bucket) wraps all outbound API calls.

**Contract:**
```python
class RateLimiter:
    def __init__(self, calls_per_minute: int) -> None: ...
    def acquire(self) -> None: ...  # blocks until a token is available
```

- Instantiated once per API client, injected as dependency
- No `time.sleep()` calls outside this utility
- Polygon client: `RateLimiter(calls_per_minute=5)`
- Finnhub client: `RateLimiter(calls_per_minute=60)`

---

## 8. Retry and Failure Handling

**Strategy:** Exponential backoff via `tenacity`, max 3 attempts per ticker per source.

```
Attempt 1 → immediate
Attempt 2 → wait 2s
Attempt 3 → wait 4s
→ On final failure: mark ticker as failed for that source
```

**Partial failure behaviour:**

| Scenario | Action |
|---|---|
| OHLCV fetch fails after 3 retries | Skip ticker entirely for this run; log error; include in alert |
| Sentiment fetch fails after 3 retries | Write OHLCV as normal; set all sentiment fields to `null`; log warning; include in alert |
| Universe resolution fails | Abort run immediately; raise critical alert |

**Alert mechanism (v1):** Write a structured JSON alert file to `data/raw/alerts/{date}.json` containing failed tickers, source, error type, and timestamp. No external notification in scope yet.

---

## 9. Run Metadata and Lineage

Each run writes `data/raw/runs/{date}.json`:

```json
{
  "run_date": "2026-04-23",
  "universe_size": 50,
  "ohlcv_success": 48,
  "ohlcv_failed": ["TICKER_A", "TICKER_B"],
  "sentiment_success": 46,
  "sentiment_null": ["TICKER_C", "TICKER_D", "TICKER_E", "TICKER_F"],
  "sentiment_failed": [],
  "polygon_api_version": "v2",
  "finnhub_api_version": "v1",
  "started_at": "2026-04-23T21:00:05Z",
  "completed_at": "2026-04-23T21:14:32Z"
}
```

---

## 10. File Structure

```
data/raw/
├── universe/
│   └── 2026-04-23.csv
├── ohlcv/
│   └── AAPL/
│       └── 2026-04-23.csv
├── sentiment/
│   └── AAPL/
│       └── 2026-04-23.csv
├── alerts/
│   └── 2026-04-23.json          # only written if failures occurred
└── runs/
    └── 2026-04-23.json
```

---

## 11. Module Structure

```
src/ingestion/
├── __init__.py
├── clients/
│   ├── __init__.py
│   ├── polygon.py          # PolygonClient — OHLCV + universe
│   └── finnhub.py          # FinnhubClient — sentiment
├── models/
│   ├── __init__.py
│   ├── ohlcv.py            # OHLCVRecord (Pydantic)
│   └── sentiment.py        # SentimentRecord (Pydantic)
├── rate_limiter.py         # RateLimiter (token bucket)
├── persistence.py          # write_csv, write_json helpers
├── alerts.py               # AlertWriter
└── pipeline.py             # IngestionPipeline — orchestrates full run
```

---

## 12. Acceptance Criteria

- [ ] Universe resolves exactly 50 tickers (or < 50 with warning logged) from Polygon grouped daily endpoint
- [ ] OHLCV data written as CSV for each ticker in the resolved universe
- [ ] Sentiment data written as CSV for each ticker; missing scores stored as empty string / blank field, not omitted
- [ ] A ticker failing OHLCV after 3 retries is excluded from output and appears in the alert file
- [ ] A ticker failing sentiment after 3 retries has all sentiment fields set to `null`; OHLCV is unaffected
- [ ] Universe resolution failure aborts the run before any ticker fetch begins
- [ ] No API call is made without passing through `RateLimiter.acquire()`
- [ ] `runs/{date}.json` is written on every completed run (including partial)
- [ ] `alerts/{date}.json` is written only when at least one failure occurred
- [ ] Raw files are never overwritten; re-running on the same date skips already-written files (idempotent)
- [ ] All Pydantic validation errors are logged with ticker and field context; invalid records are rejected
- [ ] No secrets appear in logs or output files
- [ ] `--start-date` CLI argument accepted in `YYYY-MM-DD` format; used as fetch start for all tickers
- [ ] Non-trading days (weekends, US market holidays) detected via `pandas_market_calendars`; gaps on these dates logged as expected, not as errors

---

## 13. Tests Required

| Test | Type | Notes |
|---|---|---|
| `test_universe_resolution` | Unit | Mock Polygon response; assert top 50 by volume, correct filtering |
| `test_universe_resolution_partial` | Unit | < 50 valid tickers returned; assert warning logged, run continues |
| `test_ohlcv_schema_validation` | Unit | Invalid records rejected; valid records pass |
| `test_sentiment_schema_validation` | Unit | Null fields accepted; out-of-range values rejected |
| `test_rate_limiter_blocks` | Unit | Assert `acquire()` enforces call spacing |
| `test_retry_ohlcv_failure` | Unit | Mock 3 consecutive failures; assert ticker excluded, alert written |
| `test_retry_sentiment_failure` | Unit | Mock 3 consecutive failures; assert nulls written, OHLCV unaffected |
| `test_universe_failure_aborts` | Unit | Mock universe endpoint failure; assert no ticker fetches attempted |
| `test_idempotency` | Unit | Run twice for same date; assert files not overwritten |
| `test_start_date_cli_arg` | Unit | Valid and invalid `--start-date` inputs; assert correct parsing and rejection |
| `test_trading_calendar_gap` | Unit | Gap on known holiday; assert logged as expected, not as error |
| `test_run_metadata_written` | Integration | Full mock run; assert `runs/{date}.json` schema correct |
| `test_alert_written_on_failure` | Integration | Partial failure scenario; assert alert file content |
| `test_no_real_api_calls_in_tests` | Unit | Assert all clients are mocked; no live network calls |

---

## 14. Decisions

- **Incremental fetch:** Start date passed explicitly via `--start-date YYYY-MM-DD` CLI argument. Pipeline does not auto-detect last stored date.
- **Trading calendar:** Non-trading days validated via `pandas_market_calendars` (US market calendar). Missing data on holidays/weekends is expected and logged as such — not treated as a pipeline error.
