from __future__ import annotations

import pandas as pd


def compute_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day_of_week, week_of_year, month, is_month_end. Never null."""
    df = df.copy()
    dates = pd.to_datetime(df["date"])
    df["day_of_week"] = dates.dt.dayofweek          # 0=Monday, 4=Friday
    df["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    df["month"] = dates.dt.month
    df["is_month_end"] = dates.dt.is_month_end
    return df
