from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_MAX_FORWARD_FILL_DAYS = 2


def forward_fill_close(df: pd.DataFrame, calendar_schedule: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill close (and open, high, low) for up to 2 consecutive trading days.

    Rows that require more than 2 consecutive fills are rejected (dropped + logged).
    calendar_schedule is the NYSE schedule DataFrame used to count only trading-day gaps.
    Returns a new DataFrame with imputed_close flag set where fills were applied.
    """
    df = df.copy()
    df["imputed_close"] = False
    trading_days = set(calendar_schedule.index.date)

    filled_streak = 0
    prev_valid_idx = None

    for i in df.index:
        row_date = df.at[i, "date"]
        if pd.isna(df.at[i, "close"]):
            if prev_valid_idx is None:
                logger.error("no prior close available to fill at %s; dropping row", row_date)
                df = df.drop(i)
                filled_streak = 0
                continue
            if row_date in trading_days:
                filled_streak += 1
            if filled_streak > _MAX_FORWARD_FILL_DAYS:
                logger.error(
                    "forward-fill exceeded %d-day limit at %s; dropping row",
                    _MAX_FORWARD_FILL_DAYS, row_date,
                )
                df = df.drop(i)
            else:
                for col in ["close", "open", "high", "low"]:
                    df.at[i, col] = df.at[prev_valid_idx, col]
                df.at[i, "imputed_close"] = True
        else:
            filled_streak = 0
            prev_valid_idx = i

    return df


def fill_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Zero-fill missing volume and set imputed_volume flag."""
    df = df.copy()
    df["imputed_volume"] = False
    missing = df["volume"].isna()
    df.loc[missing, "volume"] = 0
    df.loc[missing, "imputed_volume"] = True
    return df
