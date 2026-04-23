from __future__ import annotations

from datetime import date


from src.preprocessing.validator import ProcessedRecord


def _base() -> dict:
    return {
        "ticker": "AAPL",
        "date": date(2026, 1, 2),
        "open": 100.0,
        "high": 110.0,
        "low": 95.0,
        "close": 105.0,
        "volume": 1_000_000,
    }


def test_valid_record_accepted():
    rec = ProcessedRecord(**_base())
    assert rec.ticker == "AAPL"
    assert rec.sentiment_available is False


def test_defaults():
    rec = ProcessedRecord(**_base())
    assert rec.close_outlier_flag is False
    assert rec.imputed_close is False
    assert rec.is_trading_day is True
    assert rec.vwap is None


def test_optional_sentiment_fields_nullable():
    rec = ProcessedRecord(**_base(), bullish_percent=0.6, sentiment_available=True)
    assert rec.bullish_percent == 0.6
    assert rec.sentiment_available is True


def test_outlier_flags_settable():
    rec = ProcessedRecord(**_base(), close_outlier_flag=True, volume_outlier_flag=True)
    assert rec.close_outlier_flag is True
    assert rec.volume_outlier_flag is True
