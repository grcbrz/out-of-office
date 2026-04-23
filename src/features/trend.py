from __future__ import annotations

import pandas as pd


def compute_sma(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sma_10"] = df["close"].rolling(window=10).mean()
    df["sma_20"] = df["close"].rolling(window=20).mean()
    return df


def compute_ema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    return df


def compute_macd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def compute_close_to_sma(df: pd.DataFrame) -> pd.DataFrame:
    """close_to_sma20 = close / sma_20 - 1. Requires sma_20 already computed."""
    df = df.copy()
    df["close_to_sma20"] = df["close"] / df["sma_20"] - 1
    return df
