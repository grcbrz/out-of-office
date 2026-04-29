from __future__ import annotations

import numpy as np
import pandas as pd

_OBV_PCT_WINDOW: int = 20
_VOLUME_RATIO_WINDOW: int = 20


def compute_obv(df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative OBV. Intermediate — feed into compute_obv_pct_change.

    OBV cumulative magnitude depends on history length and is not stationary
    across tickers, so it is not a model input on its own.
    """
    df = df.copy()
    direction = np.sign(df["close"].diff())
    direction.iloc[0] = 0  # first row has no prior close
    df["obv"] = (direction * df["volume"]).cumsum()
    return df


def compute_obv_pct_change(df: pd.DataFrame) -> pd.DataFrame:
    """`obv.pct_change(20)` — stationary derivative of OBV.

    Requires `compute_obv` to have been called first.
    """
    df = df.copy()
    df["obv_pct_change_20"] = df["obv"].pct_change(periods=_OBV_PCT_WINDOW)
    # Replace inf (when prior obv was zero) with NaN so imputation handles it cleanly.
    df["obv_pct_change_20"] = df["obv_pct_change_20"].replace([np.inf, -np.inf], np.nan)
    return df


def compute_volume_log_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """`ln(volume / rolling_mean(volume, 20))` — return-scale volume signal.

    Replaces absolute volume lags which are non-stationary across tickers.
    """
    df = df.copy()
    rolling_mean = df["volume"].rolling(window=_VOLUME_RATIO_WINDOW).mean()
    df["volume_log_ratio_20"] = np.log(
        df["volume"].astype(float) / rolling_mean.replace(0, np.nan)
    )
    df["volume_log_ratio_20"] = df["volume_log_ratio_20"].replace(
        [np.inf, -np.inf], np.nan
    )
    return df


def compute_vwap_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """`close / vwap`. Null when vwap is null or zero."""
    df = df.copy()
    df["vwap_ratio"] = df["close"] / df["vwap"].replace(0, float("nan"))
    return df
