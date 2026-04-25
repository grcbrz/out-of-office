from __future__ import annotations

import csv
import logging
from pathlib import Path

from src.ingestion.models.ohlcv import OHLCVRecord
from src.ingestion.models.sentiment import SentimentRecord

logger = logging.getLogger(__name__)

_OHLCV_FIELDS = ["ticker", "date", "open", "high", "low", "close", "volume", "vwap"]
_SENTIMENT_FIELDS = [
    "ticker", "date", "bullish_percent", "bearish_percent",
    "company_news_score", "article_count",
]


def load_ohlcv(path: Path) -> list[OHLCVRecord]:
    """Load and validate OHLCV records from a raw CSV file.

    Rows that fail Pydantic validation are rejected and logged; valid rows are returned.
    """
    records: list[OHLCVRecord] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                records.append(OHLCVRecord(**_coerce_ohlcv(row)))
            except Exception as exc:
                logger.error("ohlcv row rejected from %s: %s — %s", path, row, exc)
    return records


def load_sentiment(path: Path) -> list[SentimentRecord]:
    """Load and validate sentiment records from a raw CSV file."""
    records: list[SentimentRecord] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                records.append(SentimentRecord(**_coerce_sentiment(row)))
            except Exception as exc:
                logger.error("sentiment row rejected from %s: %s — %s", path, row, exc)
    return records


# ------------------------------------------------------------------
# Type coercion helpers — CSV is string-only
# ------------------------------------------------------------------

def _coerce_ohlcv(row: dict) -> dict:
    return {
        "ticker": row["ticker"],
        "date": row["date"],
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": int(float(row["volume"])),
        "vwap": float(row["vwap"]) if row.get("vwap") not in (None, "", "None") else None,
    }


def _coerce_sentiment(row: dict) -> dict:
    def _opt_float(val: str | None) -> float | None:
        if val is None or val in ("", "None"):
            return None
        return float(val)

    return {
        "ticker": row["ticker"],
        "date": row["date"],
        "bullish_percent": _opt_float(row.get("bullish_percent")),
        "bearish_percent": _opt_float(row.get("bearish_percent")),
        "company_news_score": _opt_float(row.get("company_news_score")),
        "article_count": _opt_float(row.get("article_count")),
    }
