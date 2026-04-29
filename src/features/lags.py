from __future__ import annotations

import pandas as pd

_LOG_RETURN_LAGS: tuple[int, ...] = (1, 2, 3)


def compute_log_return_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Add `log_return_lag1/2/3`. Replaces removed close_lag* and volume_lag*.

    Price-level lags are non-stationary across tickers; log-return lags are
    stationary and carry the same short-horizon information.
    """
    df = df.copy()
    for lag in _LOG_RETURN_LAGS:
        df[f"log_return_lag{lag}"] = df["log_return"].shift(lag)
    return df
