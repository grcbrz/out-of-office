from __future__ import annotations

import numpy as np
import pandas as pd


def compute_obv(df: pd.DataFrame) -> pd.DataFrame:
    """On-Balance Volume: cumulative sum of +volume on up days, -volume on down days."""
    df = df.copy()
    direction = np.sign(df["close"].diff())
    direction.iloc[0] = 0  # first row has no prior close
    df["obv"] = (direction * df["volume"]).cumsum()
    df["obv_lag1"] = df["obv"].shift(1)
    return df


def compute_vwap_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """vwap_ratio = close / vwap. Null when vwap is null."""
    df = df.copy()
    df["vwap_ratio"] = df["close"] / df["vwap"].replace(0, float("nan"))
    return df
