from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add log_return = ln(close_t / close_t-1). First row is NaN."""
    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


def compute_forward_return(df: pd.DataFrame) -> pd.DataFrame:
    """Add forward_return = ln(close_t+1 / close_t).

    Used exclusively for target label computation — never as a model input.
    The last row will have NaN (no t+1 available) and must be dropped downstream.
    """
    df = df.copy()
    df["forward_return"] = np.log(df["close"].shift(-1) / df["close"])
    return df
