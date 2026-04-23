from __future__ import annotations

import logging
from datetime import date
from typing import Any

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.ingestion.models.sentiment import SentimentRecord
from src.ingestion.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

_BASE_URL = "https://finnhub.io/api/v1"


class FinnhubClient:
    """Fetches pre-computed sentiment scores from the Finnhub REST API."""

    def __init__(self, api_key: str, rate_limiter: RateLimiter) -> None:
        self._api_key = api_key
        self._rate_limiter = rate_limiter
        self._http = httpx.Client(base_url=_BASE_URL, timeout=30)

    def close(self) -> None:
        self._http.close()

    def fetch_sentiment(self, ticker: str, fetch_date: date) -> SentimentRecord:
        """Return a SentimentRecord for ticker on fetch_date.

        Missing or invalid fields are left as None — that is a valid state.
        """
        raw = self._fetch_raw_sentiment(ticker)
        return self._parse(ticker, fetch_date, raw)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=4),
        retry=retry_if_exception(
            lambda exc: not (
                isinstance(exc, httpx.HTTPStatusError)
                and exc.response.status_code in {401, 403}
            )
        ),
    )
    def _fetch_raw_sentiment(self, ticker: str) -> dict[str, Any]:
        self._rate_limiter.acquire()
        resp = self._http.get(
            "/news-sentiment",
            params={"symbol": ticker, "token": self._api_key},
        )
        resp.raise_for_status()
        return resp.json()

    def _parse(
        self, ticker: str, fetch_date: date, raw: dict[str, Any]
    ) -> SentimentRecord:
        sentiment = raw.get("sentiment") or {}
        try:
            return SentimentRecord(
                ticker=ticker,
                date=fetch_date,
                bullish_percent=sentiment.get("bullishPercent"),
                bearish_percent=sentiment.get("bearishPercent"),
                company_news_score=raw.get("companyNewsScore"),
                buzz_weekly_average=(raw.get("buzz") or {}).get("weeklyAverage"),
            )
        except Exception as exc:
            logger.error("sentiment validation error for %s: %s", ticker, exc)
            return SentimentRecord(ticker=ticker, date=fetch_date)
