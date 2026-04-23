from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.ingestion.models.ohlcv import OHLCVRecord
from src.ingestion.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.polygon.io"
_COMMON_STOCK_TYPES = {"CS"}  # common stock only


class PolygonClient:
    """Fetches universe and OHLCV data from the Polygon.io REST API."""

    def __init__(self, api_key: str, rate_limiter: RateLimiter) -> None:
        self._api_key = api_key
        self._rate_limiter = rate_limiter
        self._http = httpx.Client(base_url=_BASE_URL, timeout=30)

    def close(self) -> None:
        self._http.close()

    # ------------------------------------------------------------------
    # Universe resolution
    # ------------------------------------------------------------------

    def resolve_universe(self, trade_date: date) -> list[str]:
        """Return top-50 common-stock tickers by volume for trade_date."""
        raw = self._fetch_grouped_daily(trade_date)
        results = raw.get("results") or []
        common = [r for r in results if r.get("T") and self._is_common_stock(r)]
        common.sort(key=lambda r: r.get("v", 0), reverse=True)
        universe = [r["T"] for r in common[:50]]
        if len(universe) < 50:
            logger.warning(
                "universe resolved to %d tickers (< 50) for %s", len(universe), trade_date
            )
        return universe

    def _is_common_stock(self, result: dict[str, Any]) -> bool:
        return result.get("otc") is False or result.get("otc") is None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=4))
    def _fetch_grouped_daily(self, trade_date: date) -> dict[str, Any]:
        self._rate_limiter.acquire()
        url = f"/v2/aggs/grouped/locale/us/market/stocks/{trade_date}"
        resp = self._http.get(url, params={"apiKey": self._api_key, "adjusted": "true"})
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # OHLCV per ticker
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self, ticker: str, start_date: date, end_date: date
    ) -> list[OHLCVRecord]:
        """Fetch daily OHLCV bars for ticker between start_date and end_date."""
        raw = self._fetch_aggs(ticker, start_date, end_date)
        results = raw.get("results") or []
        records: list[OHLCVRecord] = []
        for r in results:
            try:
                record = OHLCVRecord(
                    ticker=ticker,
                    date=date.fromtimestamp(r["t"] / 1000),
                    open=r["o"],
                    high=r["h"],
                    low=r["l"],
                    close=r["c"],
                    volume=int(r["v"]),
                    vwap=r.get("vw"),
                )
                records.append(record)
            except Exception as exc:
                logger.error("validation error for %s row %s: %s", ticker, r, exc)
        return records

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=4))
    def _fetch_aggs(
        self, ticker: str, start_date: date, end_date: date
    ) -> dict[str, Any]:
        self._rate_limiter.acquire()
        url = f"/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
        resp = self._http.get(
            url,
            params={
                "apiKey": self._api_key,
                "adjusted": "true",
                "sort": "asc",
                "limit": 730,
            },
        )
        resp.raise_for_status()
        return resp.json()
