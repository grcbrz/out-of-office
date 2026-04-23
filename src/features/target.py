from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_ROLLING_WINDOW = 60
_MIN_HISTORY = 5
_BUY_PERCENTILE = 70
_SELL_PERCENTILE = 30


def compute_target_label(df: pd.DataFrame) -> pd.DataFrame:
    """Assign BUY / HOLD / SELL target based on forward_return percentile thresholds.

    Percentiles computed on a rolling 60-day window of forward_return per ticker.
    Rows with fewer than MIN_HISTORY days of return history are dropped.
    The final row (no valid forward_return) must already be NaN — it is dropped here.
    """
    df = df.copy()
    rolling = df["forward_return"].rolling(window=_ROLLING_WINDOW, min_periods=_MIN_HISTORY)
    p70 = rolling.quantile(_BUY_PERCENTILE / 100)
    p30 = rolling.quantile(_SELL_PERCENTILE / 100)

    def _label(row_idx: int) -> str | None:
        fwd = df.at[row_idx, "forward_return"]
        if pd.isna(fwd) or pd.isna(p70.iloc[row_idx]) or pd.isna(p30.iloc[row_idx]):
            return None
        if fwd > p70.iloc[row_idx]:
            return "BUY"
        if fwd < p30.iloc[row_idx]:
            return "SELL"
        return "HOLD"

    df["target"] = [_label(i) for i in range(len(df))]

    # Drop rows with insufficient history (percentiles are NaN) or no forward_return
    before = len(df)
    df = df[df["target"].notna()].reset_index(drop=True)
    logger.debug("target computation dropped %d rows", before - len(df))
    return df
