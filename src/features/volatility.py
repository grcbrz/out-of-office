"""Volatility, momentum, and ATR features.

All features in this module are stationary by construction: standard deviations,
sums of returns, and a normalised ATR. They generalise across tickers regardless
of absolute price scale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_VOL_WINDOW: int = 20
_MOMENTUM_WINDOW: int = 20
_ATR_WINDOW: int = 14
_RETURN_ZSCORE_WINDOW: int = 60
_RETURN_ZSCORE_MIN_PERIODS: int = 20


def compute_realised_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling 20-day standard deviation of `log_return`."""
    df = df.copy()
    df["realised_vol_20"] = df["log_return"].rolling(window=_VOL_WINDOW).std()
    return df


def compute_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Sum of log returns over a 20-day window — cumulative log-momentum."""
    df = df.copy()
    df["momentum_20"] = df["log_return"].rolling(window=_MOMENTUM_WINDOW).sum()
    return df


def compute_atr(df: pd.DataFrame) -> pd.DataFrame:
    """Average True Range over 14 days, divided by close.

    True Range = max(high - low, |high - prev_close|, |low - prev_close|)
    ATR_14 = mean of True Range over 14 days
    Output normalised by close so it is comparable across tickers.
    """
    df = df.copy()
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=_ATR_WINDOW).mean()
    df["atr_14"] = atr / df["close"].replace(0, np.nan)
    return df


def compute_return_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score of `log_return` over a 60-day rolling window (min_periods=20)."""
    df = df.copy()
    rolling = df["log_return"].rolling(
        window=_RETURN_ZSCORE_WINDOW,
        min_periods=_RETURN_ZSCORE_MIN_PERIODS,
    )
    mean = rolling.mean()
    std = rolling.std()
    df["log_return_zscore_60"] = (df["log_return"] - mean) / std.replace(0, np.nan)
    return df
