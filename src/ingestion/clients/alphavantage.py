from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.ingestion.models.sentiment import SentimentRecord
from src.ingestion.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.alphavantage.co"
_BULLISH_LABELS = {"Bullish", "Somewhat-Bullish"}
_BEARISH_LABELS = {"Somewhat-Bearish", "Bearish"}
_MIN_RELEVANCE = 0.1


class AlphaVantageClient:
    """Fetches news sentiment from the Alpha Vantage News Sentiment API.

    Aggregates per-article ticker sentiment scores into a single SentimentRecord
    compatible with the rest of the pipeline.
    """

    def __init__(self, api_key: str, rate_limiter: RateLimiter) -> None:
        self._api_key = api_key
        self._rate_limiter = rate_limiter
        self._http = httpx.Client(base_url=_BASE_URL, timeout=30)

    def close(self) -> None:
        self._http.close()

    def fetch_sentiment(self, ticker: str, fetch_date: date) -> SentimentRecord:
        """Return aggregated sentiment for ticker over the 24h window ending at fetch_date."""
        raw = self._fetch_raw(ticker, fetch_date)
        return self._aggregate(ticker, fetch_date, raw)

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
    def _fetch_raw(self, ticker: str, fetch_date: date) -> dict[str, Any]:
        self._rate_limiter.acquire()
        time_from = (fetch_date - timedelta(days=1)).strftime("%Y%m%dT0000")
        time_to = fetch_date.strftime("%Y%m%dT2359")
        resp = self._http.get(
            "/query",
            params={
                "function": "NEWS_SENTIMENT",
                "tickers": ticker,
                "time_from": time_from,
                "time_to": time_to,
                "limit": 50,
                "apikey": self._api_key,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        # Alpha Vantage returns 200 with error text on bad key or rate limit
        if "Error Message" in data or "Information" in data:
            msg = data.get("Error Message") or data.get("Information")
            raise RuntimeError(f"Alpha Vantage API error: {msg}")
        return data

    def _aggregate(self, ticker: str, fetch_date: date, raw: dict[str, Any]) -> SentimentRecord:
        """Aggregate per-article ticker sentiment scores into a single record."""
        feed = raw.get("feed") or []
        scores: list[float] = []
        bullish_count = 0
        bearish_count = 0

        for article in feed:
            ts = self._find_ticker_sentiment(article, ticker)
            if ts is None:
                continue
            try:
                if float(ts["relevance_score"]) < _MIN_RELEVANCE:
                    continue
                score = float(ts["ticker_sentiment_score"])
                label = ts.get("ticker_sentiment_label", "")
                scores.append(score)
                if label in _BULLISH_LABELS:
                    bullish_count += 1
                elif label in _BEARISH_LABELS:
                    bearish_count += 1
            except (ValueError, KeyError) as exc:
                logger.debug("skipping article for %s: %s", ticker, exc)

        if not scores:
            return SentimentRecord(ticker=ticker, date=fetch_date)

        total = len(scores)
        return SentimentRecord(
            ticker=ticker,
            date=fetch_date,
            bullish_percent=round(bullish_count / total, 4),
            bearish_percent=round(bearish_count / total, 4),
            company_news_score=round(sum(scores) / total, 4),
            buzz_weekly_average=float(total),
        )

    @staticmethod
    def _find_ticker_sentiment(article: dict[str, Any], ticker: str) -> dict[str, Any] | None:
        for ts in article.get("ticker_sentiment") or []:
            if ts.get("ticker") == ticker:
                return ts
        return None
