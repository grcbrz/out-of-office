from __future__ import annotations

import pandas as pd

from src.preprocessing.normaliser import compute_zscore


def _make_df(close_vals, volume_vals):
    return pd.DataFrame({"close": close_vals, "volume": volume_vals})


def test_zscore_columns_added():
    df = _make_df([100.0] * 70, [1000] * 70)
    result = compute_zscore(df)
    assert "close_zscore" in result.columns
    assert "volume_zscore" in result.columns


def test_zscore_does_not_mutate_input():
    df = _make_df([100.0] * 70, [1000] * 70)
    original = df.copy()
    compute_zscore(df)
    pd.testing.assert_frame_equal(df, original)


def test_zscore_constant_series_is_zero():
    df = _make_df([50.0] * 70, [500] * 70)
    result = compute_zscore(df)
    # std = 0 → z-score should be 0.0 (not NaN) after the guard
    non_null = result["close_zscore"].dropna()
    assert (non_null == 0.0).all()
