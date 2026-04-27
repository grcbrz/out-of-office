from __future__ import annotations

import pandas as pd


_SENTIMENT_COLS = [
    "bullish_percent", "bearish_percent", "company_news_score", "article_count",
    "positive_insights", "negative_insights", "neutral_insights",
]


def merge_ohlcv_sentiment(ohlcv_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join OHLCV and sentiment on (ticker, date). OHLCV drives the join.

    Missing sentiment rows produce None for all sentiment fields and
    sentiment_available=False. Output is sorted ascending by date.
    """
    if sentiment_df.empty or not all(c in sentiment_df.columns for c in ["ticker", "date"]):
        merged = ohlcv_df.copy()
        for col in _SENTIMENT_COLS:
            merged[col] = None
        merged["sentiment_available"] = False
        return merged.sort_values("date").reset_index(drop=True)

    merged = ohlcv_df.merge(
        sentiment_df,
        on=["ticker", "date"],
        how="left",
        suffixes=("", "_sent"),
    )
    for col in _SENTIMENT_COLS:
        if col not in merged.columns:
            merged[col] = None

    merged["sentiment_available"] = merged[_SENTIMENT_COLS].notna().any(axis=1)
    return merged.sort_values("date").reset_index(drop=True)
