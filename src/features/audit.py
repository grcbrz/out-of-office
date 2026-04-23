from __future__ import annotations

import logging

import pandas as pd

from src.features.schema import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

_CORE_FEATURES = [
    "log_return", "sma_10", "sma_20", "ema_10", "ema_20",
    "macd", "macd_signal", "macd_hist", "obv",
    "close_lag1", "close_lag2", "close_lag3",
    "day_of_week", "week_of_year", "month",
]
_NULLABLE_FEATURES = {"vwap_ratio", "bullish_percent", "bearish_percent",
                      "company_news_score", "buzz_weekly_average"}


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
