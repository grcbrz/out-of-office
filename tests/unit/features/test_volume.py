from __future__ import annotations

import pandas as pd
import pytest

from src.features.volume import compute_obv, compute_vwap_ratio


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


def test_obv_lag1():
    df = _df([100.0, 110.0], [1000, 2000])
    result = compute_obv(df)
    assert pd.isna(result["obv_lag1"].iloc[0])
    assert result["obv_lag1"].iloc[1] == result["obv"].iloc[0]


def test_vwap_ratio_null_when_vwap_null():
    df = _df([100.0], [1000], vwaps=[None])
    result = compute_vwap_ratio(df)
    assert pd.isna(result["vwap_ratio"].iloc[0])


def test_vwap_ratio_computed():
    df = _df([105.0], [1000], vwaps=[100.0])
    result = compute_vwap_ratio(df)
    assert result["vwap_ratio"].iloc[0] == pytest.approx(105.0 / 100.0)
