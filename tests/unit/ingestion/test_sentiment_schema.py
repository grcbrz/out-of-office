from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from src.ingestion.models.sentiment import SentimentRecord


def _valid() -> dict:
    return {
        "ticker": "MSFT",
        "date": date(2024, 1, 2),
        "bullish_percent": 0.6,
        "bearish_percent": 0.4,
        "company_news_score": 1.2,
        "buzz_weekly_average": 0.8,
    }


def test_valid_record_passes():
    rec = SentimentRecord(**_valid())
    assert rec.ticker == "MSFT"


def test_all_nullable_fields_none():
    rec = SentimentRecord(ticker="X", date=date(2024, 1, 2))
    assert rec.bullish_percent is None
    assert rec.bearish_percent is None
    assert rec.company_news_score is None
    assert rec.buzz_weekly_average is None


def test_bullish_percent_out_of_range_rejected():
    with pytest.raises(ValidationError):
        SentimentRecord(**{**_valid(), "bullish_percent": 1.1})


def test_bearish_percent_negative_rejected():
    with pytest.raises(ValidationError):
        SentimentRecord(**{**_valid(), "bearish_percent": -0.1})


def test_company_news_score_signed_accepted():
    # Alpha Vantage sentiment scores are signed in [-1, 1] (negative = bearish)
    rec = SentimentRecord(**{**_valid(), "company_news_score": -0.5})
    assert rec.company_news_score == -0.5


def test_buzz_weekly_average_zero_accepted():
    rec = SentimentRecord(**{**_valid(), "buzz_weekly_average": 0.0})
    assert rec.buzz_weekly_average == 0.0
