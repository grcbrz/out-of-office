from __future__ import annotations

from datetime import date

from pydantic import BaseModel

# Authoritative list of columns passed to the model.
# - Must contain only stationary, return-scale, ratio, z-score, or bounded
#   categorical features.
# - forward_return MUST NEVER appear here (target-construction column).
# - Raw price-level columns (sma_*, ema_*, macd, obv, close_lag*, volume_lag*)
#   are intermediates only; they MUST NEVER appear here either.
FEATURE_COLUMNS: list[str] = [
    # Returns
    "log_return", "log_return_lag1", "log_return_lag2", "log_return_lag3",
    "log_return_zscore_60",
    # Volatility & momentum
    "realised_vol_20", "momentum_20", "atr_14",
    # Trend (return-scale)
    "close_to_sma_10", "close_to_sma_20",
    "close_to_ema_10", "close_to_ema_20",
    "macd_norm", "macd_signal_norm", "macd_hist_norm",
    # Volume (return-scale)
    "obv_pct_change_20", "volume_log_ratio_20", "vwap_ratio",
    # Z-scores from preprocessing
    "close_zscore", "volume_zscore",
    # Cyclic seasonality
    "dow_sin", "dow_cos", "month_sin", "month_cos", "is_month_end",
    # Sentiment passthrough
    "bullish_percent", "bearish_percent", "company_news_score", "article_count",
    "positive_insights", "negative_insights", "neutral_insights",
    "sentiment_available",
    # Outlier flags
    "close_outlier_flag", "volume_outlier_flag",
    # Ticker identity (integer encoded)
    "ticker_id",
]

# Continuous-valued features used by KS / PSI drift gates.
# Excludes cyclic encodings (sin/cos already bounded — KS is misleading on them),
# boolean flags, sentiment columns (mostly null in historical data), and the
# ticker_id categorical encoding.
_NON_CONTINUOUS_FEATURES = {
    "dow_sin", "dow_cos", "month_sin", "month_cos", "is_month_end",
    "bullish_percent", "bearish_percent", "company_news_score", "article_count",
    "positive_insights", "negative_insights", "neutral_insights",
    "sentiment_available",
    "close_outlier_flag", "volume_outlier_flag",
    "ticker_id",
}
CONTINUOUS_FEATURE_COLUMNS: list[str] = [
    c for c in FEATURE_COLUMNS if c not in _NON_CONTINUOUS_FEATURES
]


class FeatureRecord(BaseModel):
    ticker: str
    date: date
    # Returns
    log_return: float | None = None
    log_return_lag1: float | None = None
    log_return_lag2: float | None = None
    log_return_lag3: float | None = None
    log_return_zscore_60: float | None = None
    # Volatility & momentum
    realised_vol_20: float | None = None
    momentum_20: float | None = None
    atr_14: float | None = None
    # Trend (return-scale)
    close_to_sma_10: float | None = None
    close_to_sma_20: float | None = None
    close_to_ema_10: float | None = None
    close_to_ema_20: float | None = None
    macd_norm: float | None = None
    macd_signal_norm: float | None = None
    macd_hist_norm: float | None = None
    # Volume (return-scale)
    obv_pct_change_20: float | None = None
    volume_log_ratio_20: float | None = None
    vwap_ratio: float | None = None
    # Z-scores (from preprocessing)
    close_zscore: float | None = None
    volume_zscore: float | None = None
    # Cyclic seasonality
    dow_sin: float | None = None
    dow_cos: float | None = None
    month_sin: float | None = None
    month_cos: float | None = None
    is_month_end: bool | None = None
    # Sentiment passthrough
    bullish_percent: float | None = None
    bearish_percent: float | None = None
    company_news_score: float | None = None
    article_count: float | None = None
    positive_insights: int | None = None
    negative_insights: int | None = None
    neutral_insights: int | None = None
    sentiment_available: bool = False
    # Outlier flags passthrough
    close_outlier_flag: bool = False
    volume_outlier_flag: bool = False
    # Target (not a model input)
    forward_return: float | None = None
    target: str | None = None
    # Ticker encoding (added by training pipeline)
    ticker_id: int | None = None
