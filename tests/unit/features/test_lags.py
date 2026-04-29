from __future__ import annotations

import pandas as pd
import pytest

from src.features.lags import compute_log_return_lags


def _df(returns):
    return pd.DataFrame({"log_return": returns})


def test_log_return_lag_shift_correct():
    df = compute_log_return_lags(_df([0.01, -0.02, 0.03]))
    assert df["log_return_lag1"].iloc[1] == pytest.approx(0.01)
    assert df["log_return_lag2"].iloc[2] == pytest.approx(0.01)
    assert df["log_return_lag3"].iloc[2] != df["log_return_lag3"].iloc[2]  # NaN at idx 2


def test_log_return_lag_first_rows_null():
    df = compute_log_return_lags(_df([0.01, -0.02, 0.03]))
    assert pd.isna(df["log_return_lag1"].iloc[0])
    assert pd.isna(df["log_return_lag2"].iloc[0])
    assert pd.isna(df["log_return_lag3"].iloc[0])
    assert pd.isna(df["log_return_lag2"].iloc[1])
