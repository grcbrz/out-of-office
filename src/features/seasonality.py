from __future__ import annotations

import numpy as np
import pandas as pd

_TRADING_DAYS_PER_WEEK: int = 5
_MONTHS_PER_YEAR: int = 12


def compute_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclic-encoded calendar features.

    Trading-week period is 5 (Mon-Fri); year period is 12 months. Cyclic
    encoding (sin/cos pair) avoids treating Monday=0 and Friday=4 as ordinal
    distance "4" — both linear and tree models can use the encoded pair, while
    neural nets and linear models can learn the cycle natively. ``is_month_end``
    is kept as a boolean event flag.
    """
    df = df.copy()
    dates = pd.to_datetime(df["date"])
    dow = dates.dt.dayofweek.astype(float)
    month = dates.dt.month.astype(float)

    df["dow_sin"] = np.sin(2 * np.pi * dow / _TRADING_DAYS_PER_WEEK)
    df["dow_cos"] = np.cos(2 * np.pi * dow / _TRADING_DAYS_PER_WEEK)
    df["month_sin"] = np.sin(2 * np.pi * month / _MONTHS_PER_YEAR)
    df["month_cos"] = np.cos(2 * np.pi * month / _MONTHS_PER_YEAR)
    df["is_month_end"] = dates.dt.is_month_end
    return df
