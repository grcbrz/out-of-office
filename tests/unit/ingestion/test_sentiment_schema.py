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
        "article_count": 0.8,
    }


def test_valid_record_passes():
    rec = SentimentRecord(**_valid())
    assert rec.ticker == "MSFT"


def test_all_nullable_fields_none():
    rec = SentimentRecord(ticker="X", date=date(2024, 1, 2))
    assert rec.bullish_percent is None
    assert rec.bearish_percent is None
    assert rec.company_news_score is None
    assert rec.article_count is None
    assert rec.positive_insights is None
    assert rec.negative_insights is None
    assert rec.neutral_insights is None


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


def test_article_count_zero_accepted():
    rec = SentimentRecord(**{**_valid(), "article_count": 0.0})
    assert rec.article_count == 0.0


def test_insight_counts_accepted():
    rec = SentimentRecord(**{**_valid(), "positive_insights": 3, "negative_insights": 1, "neutral_insights": 2})
    assert rec.positive_insights == 3
    assert rec.negative_insights == 1
    assert rec.neutral_insights == 2


def test_negative_insight_count_rejected():
    with pytest.raises(ValidationError):
        SentimentRecord(**{**_valid(), "positive_insights": -1})
