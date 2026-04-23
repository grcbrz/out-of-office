from __future__ import annotations

import pandas as pd
import pytest

from src.features.lags import compute_lag_features


def _df(closes, volumes):
    return pd.DataFrame({"close": closes, "volume": volumes})


def test_lag_shift_correct():
    df = compute_lag_features(_df([10.0, 20.0, 30.0], [100, 200, 300]))
    assert df["close_lag1"].iloc[1] == pytest.approx(10.0)
    assert df["close_lag2"].iloc[2] == pytest.approx(10.0)
    assert df["volume_lag1"].iloc[1] == pytest.approx(100)


def test_lag_first_rows_null():
    df = compute_lag_features(_df([10.0, 20.0, 30.0], [100, 200, 300]))
    assert pd.isna(df["close_lag1"].iloc[0])
    assert pd.isna(df["close_lag2"].iloc[0])
    assert pd.isna(df["close_lag3"].iloc[0])
