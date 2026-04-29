from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.volatility import (
    compute_atr,
    compute_momentum,
    compute_realised_volatility,
    compute_return_zscore,
)


def _returns_df(returns):
    return pd.DataFrame({"log_return": returns})


def _ohlc_df(highs, lows, closes):
    return pd.DataFrame({"high": highs, "low": lows, "close": closes})


def test_realised_volatility_correct():
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.01, size=30).tolist()
    out = compute_realised_volatility(_returns_df(returns))
    expected = pd.Series(returns).rolling(20).std().iloc[20]
    assert out["realised_vol_20"].iloc[20] == pytest.approx(expected)
    # Insufficient history → NaN.
    assert pd.isna(out["realised_vol_20"].iloc[18])


def test_momentum_correct():
    returns = [0.01] * 25
    out = compute_momentum(_returns_df(returns))
    assert out["momentum_20"].iloc[19] == pytest.approx(0.01 * 20)
    assert pd.isna(out["momentum_20"].iloc[18])


def test_atr_normalised_by_close():
    highs = [102.0] * 20
    lows = [98.0] * 20
    closes = [100.0] * 20
    out = compute_atr(_ohlc_df(highs, lows, closes))
    # All TR = 4 (high-low) — prev_close == 100 so |high-prev|=2, |low-prev|=2.
    # max(4, 2, 2) = 4. ATR_14 = 4. atr_14 = 4 / 100 = 0.04.
    assert out["atr_14"].iloc[14] == pytest.approx(0.04)


def test_return_zscore_correct():
    rng = np.random.default_rng(1)
    returns = rng.normal(0, 0.01, size=80).tolist()
    out = compute_return_zscore(_returns_df(returns))
    # Manual reference at idx 79 over the trailing 60-day window.
    series = pd.Series(returns)
    window = series.iloc[20:80]
    expected = (returns[79] - window.mean()) / window.std()
    assert out["log_return_zscore_60"].iloc[79] == pytest.approx(expected, rel=1e-6)
