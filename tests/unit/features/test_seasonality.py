from __future__ import annotations

import pandas as pd

from src.features.seasonality import compute_seasonality_features


def test_seasonality_no_nulls():
    df = pd.DataFrame({"date": pd.date_range("2024-01-02", periods=10).date})
    result = compute_seasonality_features(df)
    for col in ["day_of_week", "week_of_year", "month", "is_month_end"]:
        assert result[col].isna().sum() == 0


def test_day_of_week_values():
    df = pd.DataFrame({"date": [pd.Timestamp("2024-01-02").date()]})  # Tuesday
    result = compute_seasonality_features(df)
    assert result["day_of_week"].iloc[0] == 1  # 0=Mon, 1=Tue


def test_month_end_flag():
    df = pd.DataFrame({"date": [
        pd.Timestamp("2024-01-31").date(),
        pd.Timestamp("2024-01-30").date(),
    ]})
    result = compute_seasonality_features(df)
    assert result["is_month_end"].iloc[0] == True  # noqa: E712
    assert result["is_month_end"].iloc[1] == False  # noqa: E712
