from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_ZSCORE_WINDOW = 60
_MIN_HISTORY = 5
_OUTLIER_THRESHOLD = 3.0


def flag_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Add close_zscore, volume_zscore, close_outlier_flag, volume_outlier_flag columns.

    Z-score is computed over a 60-trading-day rolling window per ticker.
    If fewer than 5 days of history are available, zscore is None.
    Std = 0 produces zscore = 0.0 (logged as warning).
    Records are never removed — flags are additive columns.
    """
    df = df.copy()
    df["close_zscore"] = _rolling_zscore(df["close"], "close")
    df["volume_zscore"] = _rolling_zscore(df["volume"].astype(float), "volume")
    df["close_outlier_flag"] = df["close_zscore"].abs() > _OUTLIER_THRESHOLD
    df["volume_outlier_flag"] = df["volume_zscore"].abs() > _OUTLIER_THRESHOLD
    # None/NaN zscores → False flag (not enough history to judge)
    df["close_outlier_flag"] = df["close_outlier_flag"].fillna(False)
    df["volume_outlier_flag"] = df["volume_outlier_flag"].fillna(False)
    return df


def _rolling_zscore(series: pd.Series, name: str) -> pd.Series:
    rolling = series.rolling(window=_ZSCORE_WINDOW, min_periods=_MIN_HISTORY)
    mean = rolling.mean()
    std = rolling.std()

    std_zero = (std == 0) & series.notna()
    if std_zero.any():
        logger.warning("rolling std=0 for %s on %d rows; zscore set to 0.0", name, std_zero.sum())

    zscore = (series - mean) / std
    zscore = zscore.where(~std_zero, other=0.0)

    # Rows with insufficient history (< MIN_HISTORY) → None
    has_history = rolling.count() >= _MIN_HISTORY
    zscore = zscore.where(has_history)
    return zscore
