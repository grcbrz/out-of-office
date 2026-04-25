from __future__ import annotations

from datetime import date

from pydantic import BaseModel

# Authoritative list of columns passed to the model.
# forward_return must NEVER appear here.
FEATURE_COLUMNS: list[str] = [
    "log_return", "log_return_lag1", "log_return_lag2", "log_return_lag3",
    "sma_10", "sma_20", "ema_10", "ema_20",
    "macd", "macd_signal", "macd_hist", "close_to_sma20",
    "obv", "obv_lag1", "vwap_ratio",
    "close_lag1", "close_lag2", "close_lag3",
    "volume_lag1", "volume_lag2", "volume_lag3",
    "close_zscore", "volume_zscore",
    "day_of_week", "week_of_year", "month", "is_month_end",
    "bullish_percent", "bearish_percent", "company_news_score", "article_count",
    "sentiment_available",
    "close_outlier_flag", "volume_outlier_flag",
    "ticker_id",
]

# Continuous-valued features only — used by KS/PSI drift gates.
# Excludes calendricals (cycle weekly/yearly, not regime drift), boolean flags,
# and the ticker_id categorical encoding (PSI on integer IDs is meaningless).
_NON_CONTINUOUS_FEATURES = {
    "day_of_week", "week_of_year", "month", "is_month_end",
    # Sentiment columns: null for all historical data and sparse in recent data.
    # KS/PSI on mostly-null columns produces unstable, meaningless drift signals.
    "bullish_percent", "bearish_percent", "company_news_score", "article_count",
    "sentiment_available", "close_outlier_flag", "volume_outlier_flag",
    "ticker_id",
}
CONTINUOUS_FEATURE_COLUMNS: list[str] = [c for c in FEATURE_COLUMNS if c not in _NON_CONTINUOUS_FEATURES]


class FeatureRecord(BaseModel):
    ticker: str
    date: date
    # Returns
    log_return: float | None = None
    log_return_lag1: float | None = None
    log_return_lag2: float | None = None
    log_return_lag3: float | None = None
    # Trend
    sma_10: float | None = None
    sma_20: float | None = None
    ema_10: float | None = None
    ema_20: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_hist: float | None = None
    close_to_sma20: float | None = None
    # Volume
    obv: float | None = None
    obv_lag1: float | None = None
    vwap_ratio: float | None = None
    # Lags
    close_lag1: float | None = None
    close_lag2: float | None = None
    close_lag3: float | None = None
    volume_lag1: float | None = None
    volume_lag2: float | None = None
    volume_lag3: float | None = None
    # Z-scores (from preprocessing)
    close_zscore: float | None = None
    volume_zscore: float | None = None
    # Seasonality
    day_of_week: int | None = None
    week_of_year: int | None = None
    month: int | None = None
    is_month_end: bool | None = None
    # Sentiment passthrough
    bullish_percent: float | None = None
    bearish_percent: float | None = None
    company_news_score: float | None = None
    article_count: float | None = None
    sentiment_available: bool = False
    # Outlier flags passthrough
    close_outlier_flag: bool = False
    volume_outlier_flag: bool = False
    # Target (not a model input)
    forward_return: float | None = None
    target: str | None = None
    # Ticker encoding (added by training pipeline)
    ticker_id: int | None = None
