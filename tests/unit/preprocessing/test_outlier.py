from __future__ import annotations

import pandas as pd

from src.preprocessing.outlier import flag_outliers


def _df_constant(n=10, close=100.0, volume=1000):
    return pd.DataFrame({
        "ticker": ["AAPL"] * n,
        "date": pd.date_range("2024-01-01", periods=n).date,
        "close": [close] * n,
        "volume": [volume] * n,
    })


def _df_with_spike(n=65, spike_idx=64, spike_value=999.0):
    closes = [100.0] * n
    closes[spike_idx] = spike_value
    return pd.DataFrame({
        "ticker": ["AAPL"] * n,
        "date": pd.bdate_range("2023-01-02", periods=n).date,
        "close": closes,
        "volume": [1000.0] * n,
    })


def test_zscore_sufficient_history():
    df = _df_with_spike()
    result = flag_outliers(df)
    # Spike at index 64 should be flagged
    assert result.loc[64, "close_outlier_flag"] == True  # noqa: E712


def test_zscore_insufficient_history():
    """< 5 days → zscore must be None."""
    df = _df_constant(n=4)
    result = flag_outliers(df)
    assert result["close_zscore"].isna().all()


def test_outlier_flag_set():
    df = _df_with_spike()
    result = flag_outliers(df)
    assert result.loc[64, "close_outlier_flag"] == True  # noqa: E712


def test_outlier_record_not_removed():
    """Flagged records must remain in the output."""
    df = _df_with_spike()
    result = flag_outliers(df)
    assert len(result) == len(df)


def test_zscore_std_zero_produces_zero(caplog):
    """Constant prices → std=0 → zscore=0.0, no division by zero."""
    import logging
    df = _df_constant(n=70)
    with caplog.at_level(logging.WARNING):
        result = flag_outliers(df)
    # After warm-up (5 days), zscore should be 0.0 (not NaN)
    post_warmup = result["close_zscore"].dropna()
    assert (post_warmup == 0.0).all()
    assert any("std=0" in r.message for r in caplog.records)
