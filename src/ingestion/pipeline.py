from __future__ import annotations

import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path

import pandas_market_calendars as mcal

from src.ingestion.alerts import AlertWriter
from src.ingestion.clients.finnhub import FinnhubClient
from src.ingestion.clients.polygon import PolygonClient
from src.ingestion.models.ohlcv import OHLCVRecord
from src.ingestion.models.sentiment import SentimentRecord
from src.ingestion.persistence import write_csv, write_json
from src.ingestion.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

_RAW_DIR = Path("data/raw")
_POLYGON_CALLS_PER_MIN = 5
_FINNHUB_CALLS_PER_MIN = 60


class IngestionPipeline:
    """Orchestrates the full nightly ingestion run.

    Resolves the top-50 universe, fetches OHLCV and sentiment for each ticker,
    validates, persists raw CSVs, and writes run metadata + alerts.
    """

    def __init__(
        self,
        polygon_api_key: str,
        finnhub_api_key: str,
        raw_dir: Path = _RAW_DIR,
    ) -> None:
        polygon_limiter = RateLimiter(_POLYGON_CALLS_PER_MIN)
        finnhub_limiter = RateLimiter(_FINNHUB_CALLS_PER_MIN)
        self._polygon = PolygonClient(polygon_api_key, polygon_limiter)
        self._finnhub = FinnhubClient(finnhub_api_key, finnhub_limiter)
        self._raw_dir = raw_dir
        self._alert_writer = AlertWriter(raw_dir / "alerts")
        self._calendar = mcal.get_calendar("NYSE")

    def run(self, run_date: date, start_date: date) -> None:
        """Execute a full ingestion run for run_date.

        Args:
            run_date: The date identifying this run (usually today).
            start_date: First date of the OHLCV fetch window (--start-date CLI arg).
        """
        started_at = datetime.now(timezone.utc)
        logger.info("ingestion starting for %s", run_date)

        universe = self._resolve_universe(run_date)
        logger.info("universe: %d tickers", len(universe))

        ohlcv_success: list[str] = []
        ohlcv_failed: list[str] = []
        sentiment_success: list[str] = []
        sentiment_null: list[str] = []
        sentiment_failed: list[str] = []

        for ticker in universe:
            ohlcv_ok = self._ingest_ohlcv(ticker, run_date, start_date)
            if ohlcv_ok:
                ohlcv_success.append(ticker)
            else:
                ohlcv_failed.append(ticker)
                continue  # skip sentiment if OHLCV failed

            sent_status = self._ingest_sentiment(ticker, run_date)
            if sent_status == "ok":
                sentiment_success.append(ticker)
            elif sent_status == "null":
                sentiment_null.append(ticker)
            else:
                sentiment_failed.append(ticker)

        completed_at = datetime.now(timezone.utc)
        run_meta = {
            "run_date": str(run_date),
            "universe_size": len(universe),
            "ohlcv_success": len(ohlcv_success),
            "ohlcv_failed": ohlcv_failed,
            "sentiment_success": len(sentiment_success),
            "sentiment_null": sentiment_null,
            "sentiment_failed": sentiment_failed,
            "polygon_api_version": "v2",
            "finnhub_api_version": "v1",
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
        }
        write_json(self._raw_dir / "runs" / f"{run_date}.json", run_meta)

        self._alert_writer.write(
            run_date,
            {"ohlcv_failed": ohlcv_failed, "sentiment_failed": sentiment_failed},
        )
        logger.info(
            "ingestion complete: ohlcv ok=%d failed=%d, sentiment ok=%d null=%d failed=%d",
            len(ohlcv_success), len(ohlcv_failed),
            len(sentiment_success), len(sentiment_null), len(sentiment_failed),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_universe(self, run_date: date) -> list[str]:
        try:
            return self._polygon.resolve_universe(run_date)
        except Exception as exc:
            logger.critical("universe resolution failed: %s", exc)
            raise RuntimeError(f"universe resolution failed for {run_date}") from exc

    def _ingest_ohlcv(self, ticker: str, run_date: date, start_date: date) -> bool:
        dest = self._raw_dir / "ohlcv" / ticker / f"{run_date}.csv"
        if dest.exists():
            logger.debug("ohlcv already exists for %s %s, skipping", ticker, run_date)
            return True
        try:
            records = self._polygon.fetch_ohlcv(ticker, start_date, run_date)
            rows = [self._ohlcv_to_dict(r) for r in records]
            write_csv(dest, rows)
            return True
        except Exception as exc:
            logger.error("ohlcv fetch failed for %s: %s", ticker, exc)
            return False

    def _ingest_sentiment(self, ticker: str, run_date: date) -> str:
        dest = self._raw_dir / "sentiment" / ticker / f"{run_date}.csv"
        if dest.exists():
            logger.debug("sentiment already exists for %s %s, skipping", ticker, run_date)
            return "ok"
        try:
            record = self._finnhub.fetch_sentiment(ticker, run_date)
            row = self._sentiment_to_dict(record)
            write_csv(dest, [row])
            all_null = all(
                row[f] in (None, "") for f in
                ["bullish_percent", "bearish_percent", "company_news_score", "buzz_weekly_average"]
            )
            return "null" if all_null else "ok"
        except Exception as exc:
            logger.error("sentiment fetch failed for %s: %s", ticker, exc)
            null_record = SentimentRecord(ticker=ticker, date=run_date)
            write_csv(dest, [self._sentiment_to_dict(null_record)])
            return "failed"

    def _is_trading_day(self, d: date) -> bool:
        schedule = self._calendar.schedule(
            start_date=str(d), end_date=str(d)
        )
        return not schedule.empty

    @staticmethod
    def _ohlcv_to_dict(r: OHLCVRecord) -> dict:
        return {
            "ticker": r.ticker,
            "date": str(r.date),
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "volume": r.volume,
            "vwap": r.vwap if r.vwap is not None else "",
        }

    @staticmethod
    def _sentiment_to_dict(r: SentimentRecord) -> dict:
        return {
            "ticker": r.ticker,
            "date": str(r.date),
            "bullish_percent": r.bullish_percent if r.bullish_percent is not None else "",
            "bearish_percent": r.bearish_percent if r.bearish_percent is not None else "",
            "company_news_score": r.company_news_score if r.company_news_score is not None else "",
            "buzz_weekly_average": r.buzz_weekly_average if r.buzz_weekly_average is not None else "",
        }
