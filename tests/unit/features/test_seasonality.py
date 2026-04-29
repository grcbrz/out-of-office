from __future__ import annotations

import math

import pandas as pd

from src.features.seasonality import compute_seasonality_features


def test_seasonality_cyclic_completeness():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=10, freq="B")})
    out = compute_seasonality_features(df)

    for col in ("dow_sin", "dow_cos", "month_sin", "month_cos"):
        assert not out[col].isna().any()
        assert ((out[col] >= -1.0) & (out[col] <= 1.0)).all()

    # is_month_end is a clean boolean.
    assert out["is_month_end"].dtype == bool


def test_seasonality_cycle_values_for_monday():
    # 2024-01-01 is a Monday → dayofweek=0 → sin=0, cos=1.
    df = pd.DataFrame({"date": [pd.Timestamp("2024-01-01")]})
    out = compute_seasonality_features(df)
    assert math.isclose(out["dow_sin"].iloc[0], 0.0, abs_tol=1e-9)
    assert math.isclose(out["dow_cos"].iloc[0], 1.0, abs_tol=1e-9)


def test_month_end_flag():
    df = pd.DataFrame({"date": [
        pd.Timestamp("2024-01-31"),
        pd.Timestamp("2024-01-30"),
    ]})
    out = compute_seasonality_features(df)
    assert out["is_month_end"].iloc[0]
    assert not out["is_month_end"].iloc[1]
