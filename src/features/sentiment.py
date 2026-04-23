from __future__ import annotations

import pandas as pd

_SENTIMENT_COLS = [
    "bullish_percent", "bearish_percent", "company_news_score",
    "buzz_weekly_average", "sentiment_available",
]


def passthrough_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure sentiment columns exist; fill missing ones with None/False."""
    df = df.copy()
    for col in _SENTIMENT_COLS:
        if col not in df.columns:
            df[col] = None if col != "sentiment_available" else False
    return df
