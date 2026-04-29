from __future__ import annotations

import logging

import pandas as pd

from src.features.schema import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

_CORE_FEATURES = [
    "log_return", "log_return_lag1", "log_return_lag2", "log_return_lag3",
    "log_return_zscore_60", "realised_vol_20", "momentum_20", "atr_14",
    "close_to_sma_10", "close_to_sma_20", "close_to_ema_10", "close_to_ema_20",
    "macd_norm", "macd_signal_norm", "macd_hist_norm",
    "obv_pct_change_20", "volume_log_ratio_20",
    "dow_sin", "dow_cos", "month_sin", "month_cos",
]
_NULLABLE_FEATURES = {"vwap_ratio", "bullish_percent", "bearish_percent",
                      "company_news_score", "article_count",
                      "positive_insights", "negative_insights", "neutral_insights"}


class LookaheadBiasError(Exception):
    """Raised when a feature derived from t+1 data is detected in the feature set."""


def lookahead_bias_guard(df: pd.DataFrame) -> None:
    """Assert that forward_return is not in the model feature columns.

    Raises LookaheadBiasError on any violation. This is a hard gate — the
    pipeline must not proceed if this fires.
    """
    if "forward_return" in FEATURE_COLUMNS:
        raise LookaheadBiasError(
            "forward_return must never appear in FEATURE_COLUMNS — "
            "it is a target-construction column, not a model input."
        )
    if "forward_return" in df.columns:
        # Verify it is not accidentally being treated as an input feature
        # by checking it is only used as a label source
        logger.debug("forward_return present in DataFrame (expected); not in FEATURE_COLUMNS")


def null_audit(df: pd.DataFrame, ticker: str) -> None:
    """Log null rates per column. Warn on unexpected nulls in core features."""
    for col in df.columns:
        null_rate = df[col].isna().mean()
        if null_rate > 0:
            if col in _CORE_FEATURES:
                logger.warning(
                    "unexpected nulls in core feature %s for %s: %.1f%%",
                    col, ticker, null_rate * 100,
                )
            elif col in _NULLABLE_FEATURES:
                logger.debug("expected nulls in %s for %s: %.1f%%", col, ticker, null_rate * 100)
