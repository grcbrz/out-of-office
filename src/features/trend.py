from __future__ import annotations

import pandas as pd

# Period parameters — single source of truth.
_SMA_PERIODS: tuple[int, ...] = (10, 20)
_EMA_PERIODS: tuple[int, ...] = (10, 20)
_MACD_FAST: int = 12
_MACD_SLOW: int = 26
_MACD_SIGNAL: int = 9


def compute_sma(df: pd.DataFrame) -> pd.DataFrame:
    """Add `sma_{p}` columns. Intermediate — not a model input. See ratios below."""
    df = df.copy()
    for p in _SMA_PERIODS:
        df[f"sma_{p}"] = df["close"].rolling(window=p).mean()
    return df


def compute_ema(df: pd.DataFrame) -> pd.DataFrame:
    """Add `ema_{p}` columns. Intermediate — not a model input. See ratios below."""
    df = df.copy()
    for p in _EMA_PERIODS:
        df[f"ema_{p}"] = df["close"].ewm(span=p, adjust=False).mean()
    return df


def compute_macd(df: pd.DataFrame) -> pd.DataFrame:
    """Add `macd`, `macd_signal`, `macd_hist`. Intermediate — feed into compute_macd_normalised."""
    df = df.copy()
    fast = df["close"].ewm(span=_MACD_FAST, adjust=False).mean()
    slow = df["close"].ewm(span=_MACD_SLOW, adjust=False).mean()
    df["macd"] = fast - slow
    df["macd_signal"] = df["macd"].ewm(span=_MACD_SIGNAL, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def compute_close_to_ma_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Return-scale trend features: ``close / MA - 1``.

    Replaces absolute-level SMA/EMA features which do not generalise across
    tickers with different price scales. Requires `compute_sma` and
    `compute_ema` to have been called first so `sma_*` / `ema_*` columns exist.
    """
    df = df.copy()
    for p in _SMA_PERIODS:
        df[f"close_to_sma_{p}"] = df["close"] / df[f"sma_{p}"] - 1
    for p in _EMA_PERIODS:
        df[f"close_to_ema_{p}"] = df["close"] / df[f"ema_{p}"] - 1
    return df


def compute_macd_normalised(df: pd.DataFrame) -> pd.DataFrame:
    """MACD components divided by close — return-scale, comparable across tickers.

    Requires `compute_macd` to have been called first.
    """
    df = df.copy()
    df["macd_norm"] = df["macd"] / df["close"]
    df["macd_signal_norm"] = df["macd_signal"] / df["close"]
    df["macd_hist_norm"] = df["macd_hist"] / df["close"]
    return df
