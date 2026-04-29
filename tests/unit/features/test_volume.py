from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.volume import (
    compute_obv,
    compute_obv_pct_change,
    compute_volume_log_ratio,
    compute_vwap_ratio,
)


def _df(closes, volumes, vwaps=None):
    d = {"close": closes, "volume": volumes}
    if vwaps is not None:
        d["vwap"] = vwaps
    return pd.DataFrame(d)


def test_obv_sign_logic():
    df = _df([100.0, 110.0, 105.0], [1000, 2000, 1500])
    result = compute_obv(df)
    # row 0: direction=0, obv=0
    # row 1: close went up, obv += 2000 → 2000
    # row 2: close went down, obv -= 1500 → 500
    assert result["obv"].iloc[0] == 0
    assert result["obv"].iloc[1] == 2000
    assert result["obv"].iloc[2] == 500


def test_obv_pct_change_against_reference():
    closes = [100 + i for i in range(25)]
    volumes = [1000] * 25
    df = compute_obv_pct_change(compute_obv(_df(closes, volumes)))
    # Manual: obv is monotonic increasing (all up days), so the 20-day pct change
    # at index 20 = obv[20] / obv[0] - 1, but obv[0] == 0 → NaN expected.
    assert pd.isna(df["obv_pct_change_20"].iloc[20])
    # Index 21 onwards should be finite.
    assert np.isfinite(df["obv_pct_change_20"].iloc[21])


def test_volume_log_ratio_against_reference():
    volumes = [1000] * 19 + [2000]  # last day is double the rolling mean
    closes = [100.0] * 20
    df = compute_volume_log_ratio(_df(closes, volumes))
    # mean over the 20-day window ending at idx 19 is (19*1000 + 2000) / 20 = 1050
    expected = np.log(2000 / 1050)
    assert df["volume_log_ratio_20"].iloc[19] == pytest.approx(expected)


def test_vwap_ratio_null_when_vwap_null():
    df = _df([100.0], [1000], vwaps=[None])
    result = compute_vwap_ratio(df)
    assert pd.isna(result["vwap_ratio"].iloc[0])


def test_vwap_ratio_computed():
    df = _df([105.0], [1000], vwaps=[100.0])
    result = compute_vwap_ratio(df)
    assert result["vwap_ratio"].iloc[0] == pytest.approx(105.0 / 100.0)
