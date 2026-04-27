from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.ingestion.models.ohlcv import OHLCVRecord
from src.ingestion.models.sentiment import SentimentRecord
from src.ingestion.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.massive.com"  # formerly api.polygon.io (rebranded Oct 2025)


class PolygonClient:
    """Fetches universe and OHLCV data from the Massive (formerly Polygon.io) REST API."""

    def __init__(self, api_key: str, rate_limiter: RateLimiter) -> None:
        self._api_key = api_key
        self._rate_limiter = rate_limiter
        self._http = httpx.Client(base_url=_BASE_URL, timeout=30)

    def close(self) -> None:
        self._http.close()

    # ------------------------------------------------------------------
    # Universe resolution
    # ------------------------------------------------------------------

    def resolve_universe(self, trade_date: date, universe_size: int = 50) -> list[str]:
        """Return top-N common-stock tickers by volume for trade_date."""
        raw = self._fetch_grouped_daily(trade_date)
        results = raw.get("results") or []
        common = [r for r in results if r.get("T") and self._is_common_stock(r)]
        common.sort(key=lambda r: r.get("v", 0), reverse=True)
        universe = [r["T"] for r in common[:universe_size]]
        if len(universe) < universe_size:
            logger.warning(
                "universe resolved to %d tickers (< %d) for %s", len(universe), universe_size, trade_date
            )
        return universe

    def _is_common_stock(self, result: dict[str, Any]) -> bool:
        # The grouped daily endpoint does not include a "type" field — only "otc".
        # Non-OTC is the best available proxy for exchange-listed equities from this endpoint.
        return result.get("otc") is False or result.get("otc") is None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=4))
    def _fetch_grouped_daily(self, trade_date: date) -> dict[str, Any]:
        self._rate_limiter.acquire()
        url = f"/v2/aggs/grouped/locale/us/market/stocks/{trade_date}"
        all_results: list[dict[str, Any]] = []
        params: dict[str, Any] = {"apiKey": self._api_key, "adjusted": "true"}
        next_url: str | None = None

        while True:
            if next_url:
                resp = self._http.get(next_url, params={"apiKey": self._api_key})
            else:
                resp = self._http.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") == "ERROR":
                raise RuntimeError(f"Massive API error: {data.get('error')}")
            all_results.extend(data.get("results") or [])
            next_url = data.get("next_url")
            if not next_url:
                break

        return {"results": all_results}

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

    # ------------------------------------------------------------------
    # News sentiment (Polygon /v2/reference/news + FinBERT)
    # ------------------------------------------------------------------

    def fetch_news_sentiment(self, ticker: str, fetch_date: date, finbert: Any) -> SentimentRecord:
        """Aggregate Polygon news insights for ticker over a 24h window using FinBERT scoring."""
        date_from = (fetch_date - timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
        date_to = fetch_date.strftime("%Y-%m-%dT23:59:59Z")
        articles = self._fetch_news_pages(ticker, date_from, date_to)
        return self._aggregate_insights(ticker, fetch_date, articles, finbert)

    def _fetch_news_pages(self, ticker: str, date_from: str, date_to: str) -> list[dict[str, Any]]:
        self._rate_limiter.acquire()
        params: dict[str, Any] = {
            "ticker": ticker,
            "published_utc.gte": date_from,
            "published_utc.lte": date_to,
            "limit": 50,
            "sort": "published_utc",
            "apiKey": self._api_key,
        }
        all_articles: list[dict[str, Any]] = []
        next_url: str | None = None

        while True:
            if next_url:
                resp = self._http.get(next_url, params={"apiKey": self._api_key})
            else:
                resp = self._http.get("/v2/reference/news", params=params)
            resp.raise_for_status()
            data = resp.json()
            all_articles.extend(data.get("results") or [])
            next_url = data.get("next_url")
            if not next_url:
                break

        return all_articles

    def _aggregate_insights(
        self, ticker: str, fetch_date: date, articles: list[dict[str, Any]], finbert: Any
    ) -> SentimentRecord:
        _SIGN = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
        positive = negative = neutral = 0
        weighted_sum = 0.0
        n_scored = 0

        for article in articles:
            for insight in (article.get("insights") or []):
                if (insight.get("ticker") or "").upper() != ticker.upper():
                    continue
                polygon_label = (insight.get("sentiment") or "neutral").lower()
                positive += int(polygon_label == "positive")
                negative += int(polygon_label == "negative")
                neutral += int(polygon_label == "neutral")

                reasoning = (insight.get("sentiment_reasoning") or "").strip()
                if reasoning and finbert is not None:
                    try:
                        result = finbert(reasoning[:512])[0]
                        fb_label = result["label"].lower()
                        confidence = float(result["score"])
                    except Exception as exc:
                        logger.debug("FinBERT failed on reasoning text: %s", exc)
                        fb_label = polygon_label
                        confidence = 0.5
                else:
                    fb_label = polygon_label
                    confidence = 0.5

                weighted_sum += _SIGN.get(fb_label, 0.0) * confidence
                n_scored += 1

        total_insights = positive + negative + neutral
        if total_insights == 0:
            return SentimentRecord(ticker=ticker, date=fetch_date)

        article_count = float(len(articles))
        bullish_pct = round(positive / total_insights, 4)
        bearish_pct = round(negative / total_insights, 4)
        score = max(-1.0, min(1.0, round(weighted_sum / max(n_scored, 1), 4)))

        return SentimentRecord(
            ticker=ticker,
            date=fetch_date,
            bullish_percent=bullish_pct,
            bearish_percent=bearish_pct,
            company_news_score=score,
            article_count=article_count,
            positive_insights=positive,
            negative_insights=negative,
            neutral_insights=neutral,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=4))
    def _fetch_aggs(
        self, ticker: str, start_date: date, end_date: date
    ) -> dict[str, Any]:
        self._rate_limiter.acquire()
        url = f"/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
        all_results: list[dict[str, Any]] = []
        params: dict[str, Any] = {
            "apiKey": self._api_key,
            "adjusted": "true",
            "sort": "asc",
            "limit": 730,
        }
        next_url: str | None = None

        while True:
            if next_url:
                resp = self._http.get(next_url, params={"apiKey": self._api_key})
            else:
                resp = self._http.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") == "ERROR":
                raise RuntimeError(f"Massive API error for {ticker}: {data.get('error')}")
            all_results.extend(data.get("results") or [])
            next_url = data.get("next_url")
            if not next_url:
                break

        return {"results": all_results}
