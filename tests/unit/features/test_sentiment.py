from __future__ import annotations

import pandas as pd

from src.features.sentiment import passthrough_sentiment


def test_passthrough_adds_missing_columns():
    df = pd.DataFrame({"close": [100.0]})
    result = passthrough_sentiment(df)
    assert "bullish_percent" in result.columns
    assert "sentiment_available" in result.columns
    assert result["sentiment_available"].iloc[0] == False


def test_passthrough_preserves_existing_columns():
    df = pd.DataFrame({"close": [100.0], "bullish_percent": [0.6], "sentiment_available": [True]})
    result = passthrough_sentiment(df)
    assert result["bullish_percent"].iloc[0] == 0.6
    assert result["sentiment_available"].iloc[0] == True


def test_passthrough_does_not_mutate():
    df = pd.DataFrame({"close": [100.0]})
    original_cols = set(df.columns)
    passthrough_sentiment(df)
    assert set(df.columns) == original_cols
