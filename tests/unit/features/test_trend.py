from __future__ import annotations

import pandas as pd
import pytest

from src.features.trend import compute_close_to_sma, compute_ema, compute_macd, compute_sma


def _df(n=30, base=100.0):
    return pd.DataFrame({"close": [base + i for i in range(n)]})


def test_sma_10_correct():
    df = compute_sma(_df(20))
    # At index 9 (10th row), sma_10 = mean of closes[0:10]
    expected = sum(range(10)) / 10 + 100.0
    assert df["sma_10"].iloc[9] == pytest.approx(expected)


def test_sma_warm_up_nan():
    df = compute_sma(_df(5))
    assert df["sma_10"].isna().all()


def test_ema_10_produces_values():
    df = compute_ema(_df(30))
    assert not df["ema_10"].isna().any()


def test_macd_components():
    df = _df(50)
    df = compute_ema(df)
    result = compute_macd(df)
    assert "macd" in result.columns
    assert "macd_signal" in result.columns
    assert "macd_hist" in result.columns
    # hist = macd - signal
    row = result.iloc[40]
    assert row["macd_hist"] == pytest.approx(row["macd"] - row["macd_signal"])


def test_close_to_sma20_ratio():
    df = compute_sma(_df(30))
    result = compute_close_to_sma(df)
    row = result.iloc[25]
    expected = row["close"] / row["sma_20"] - 1
    assert result.iloc[25]["close_to_sma20"] == pytest.approx(expected)
