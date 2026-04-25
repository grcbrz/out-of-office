from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.preprocessing.loader import load_ohlcv, load_sentiment


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_dtype_cast_valid(tmp_path):
    path = tmp_path / "ohlcv.csv"
    _write_csv(path, [{
        "ticker": "AAPL", "date": "2024-01-02",
        "open": "100.0", "high": "110.0", "low": "90.0",
        "close": "105.0", "volume": "1000000", "vwap": "103.5",
    }])
    records = load_ohlcv(path)
    assert len(records) == 1
    assert records[0].open == 100.0
    assert records[0].volume == 1000000


def test_dtype_cast_failure_rejects_row(tmp_path, caplog):
    path = tmp_path / "ohlcv.csv"
    _write_csv(path, [
        {"ticker": "AAPL", "date": "2024-01-02",
         "open": "NOT_A_FLOAT", "high": "110.0", "low": "90.0",
         "close": "105.0", "volume": "1000000", "vwap": ""},
        {"ticker": "MSFT", "date": "2024-01-02",
         "open": "200.0", "high": "210.0", "low": "190.0",
         "close": "205.0", "volume": "500000", "vwap": ""},
    ])
    import logging
    with caplog.at_level(logging.ERROR):
        records = load_ohlcv(path)
    assert len(records) == 1
    assert records[0].ticker == "MSFT"


def test_sentiment_load_valid(tmp_path):
    path = tmp_path / "sentiment.csv"
    _write_csv(path, [{
        "ticker": "AAPL", "date": "2024-01-02",
        "bullish_percent": "0.6", "bearish_percent": "0.4",
        "company_news_score": "1.2", "article_count": "0.8",
    }])
    records = load_sentiment(path)
    assert len(records) == 1
    assert records[0].bullish_percent == pytest.approx(0.6)


def test_sentiment_load_null_fields(tmp_path):
    path = tmp_path / "sentiment.csv"
    _write_csv(path, [{
        "ticker": "AAPL", "date": "2024-01-02",
        "bullish_percent": "", "bearish_percent": "",
        "company_news_score": "", "article_count": "",
    }])
    records = load_sentiment(path)
    assert len(records) == 1
    assert records[0].bullish_percent is None
