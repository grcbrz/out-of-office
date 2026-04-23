from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing.merger import merge_ohlcv_sentiment


def _ohlcv_df():
    return pd.DataFrame({
        "ticker": ["AAPL", "AAPL"],
        "date": [pd.Timestamp("2024-01-02").date(), pd.Timestamp("2024-01-03").date()],
        "close": [100.0, 101.0],
    })


def _sentiment_df(date_str="2024-01-02"):
    return pd.DataFrame({
        "ticker": ["AAPL"],
        "date": [pd.Timestamp(date_str).date()],
        "bullish_percent": [0.6],
        "bearish_percent": [0.4],
        "company_news_score": [1.2],
        "buzz_weekly_average": [0.8],
    })


def test_sentiment_merge_present():
    result = merge_ohlcv_sentiment(_ohlcv_df(), _sentiment_df("2024-01-02"))
    row = result[result["date"] == pd.Timestamp("2024-01-02").date()].iloc[0]
    assert row["bullish_percent"] == pytest.approx(0.6)
    assert row["sentiment_available"] == True  # noqa: E712


def test_sentiment_merge_missing_produces_false():
    result = merge_ohlcv_sentiment(_ohlcv_df(), pd.DataFrame())
    assert not result["sentiment_available"].any()


def test_partial_sentiment_only_available_on_matched_date():
    result = merge_ohlcv_sentiment(_ohlcv_df(), _sentiment_df("2024-01-02"))
    row_jan3 = result[result["date"] == pd.Timestamp("2024-01-03").date()].iloc[0]
    assert row_jan3["sentiment_available"] == False  # noqa: E712


def test_output_sorted_by_date():
    ohlcv = pd.DataFrame({
        "ticker": ["AAPL", "AAPL"],
        "date": [pd.Timestamp("2024-01-03").date(), pd.Timestamp("2024-01-02").date()],
        "close": [101.0, 100.0],
    })
    result = merge_ohlcv_sentiment(ohlcv, pd.DataFrame())
    dates = list(result["date"])
    assert dates == sorted(dates)
