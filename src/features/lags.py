from __future__ import annotations

import pandas as pd


def compute_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for lag in [1, 2, 3]:
        df[f"close_lag{lag}"] = df["close"].shift(lag)
        df[f"volume_lag{lag}"] = df["volume"].shift(lag)
    return df
