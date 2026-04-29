from __future__ import annotations

import pandas as pd


_SENTIMENT_COLS = [
    "bullish_percent", "bearish_percent", "company_news_score", "article_count",
    "positive_insights", "negative_insights", "neutral_insights",
]


def merge_ohlcv_sentiment(ohlcv_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join OHLCV and sentiment on (ticker, date). OHLCV drives the join.

    If a sentiment row's date is later than the most recent OHLCV date (e.g.
    today's news arrives before today's market close data is published), the
    sentiment is backfilled onto the most recent OHLCV row so it is never lost.

    Missing sentiment rows produce None for all sentiment fields and
    sentiment_available=False. Output is sorted ascending by date.
    """
    if sentiment_df.empty or not all(c in sentiment_df.columns for c in ["ticker", "date"]):
        merged = ohlcv_df.copy()
        for col in _SENTIMENT_COLS:
            merged[col] = None
        merged["sentiment_available"] = False
        return merged.sort_values("date").reset_index(drop=True)

    # Backfill any sentiment rows whose date exceeds the last OHLCV date.
    max_ohlcv_date = ohlcv_df["date"].max()
    sentiment_df = sentiment_df.copy()
    sentiment_df.loc[sentiment_df["date"] > max_ohlcv_date, "date"] = max_ohlcv_date

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
