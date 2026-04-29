from __future__ import annotations

import pandas as pd
import pytest

from src.features.trend import (
    compute_close_to_ma_ratios,
    compute_ema,
    compute_macd,
    compute_macd_normalised,
    compute_sma,
)


def _df(n: int = 30, base: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame({"close": [base + i for i in range(n)]})


# ----- intermediate (still public for use inside the pipeline) -----

def test_sma_10_correct():
    df = compute_sma(_df(20))
    expected = sum(range(10)) / 10 + 100.0
    assert df["sma_10"].iloc[9] == pytest.approx(expected)


def test_sma_warm_up_nan():
    df = compute_sma(_df(5))
    assert df["sma_10"].isna().all()


def test_ema_10_produces_values():
    df = compute_ema(_df(30))
    assert not df["ema_10"].isna().any()


def test_macd_components():
    df = compute_macd(_df(50))
    row = df.iloc[40]
    assert row["macd_hist"] == pytest.approx(row["macd"] - row["macd_signal"])


# ----- model-input ratios -----

def test_close_to_sma_ratios_returns_scale():
    df = compute_sma(_df(30))
    df = compute_close_to_ma_ratios(compute_ema(df))
    row = df.iloc[25]
    assert row["close_to_sma_20"] == pytest.approx(row["close"] / row["sma_20"] - 1)
    assert row["close_to_sma_10"] == pytest.approx(row["close"] / row["sma_10"] - 1)
    assert row["close_to_ema_10"] == pytest.approx(row["close"] / row["ema_10"] - 1)
    assert row["close_to_ema_20"] == pytest.approx(row["close"] / row["ema_20"] - 1)


def test_macd_normalised_returns_scale():
    df = compute_macd(_df(50))
    out = compute_macd_normalised(df)
    row = out.iloc[40]
    assert row["macd_norm"] == pytest.approx(row["macd"] / row["close"])
    assert row["macd_signal_norm"] == pytest.approx(row["macd_signal"] / row["close"])
    assert row["macd_hist_norm"] == pytest.approx(row["macd_hist"] / row["close"])
