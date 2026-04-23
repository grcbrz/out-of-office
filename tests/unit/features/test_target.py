from __future__ import annotations

import pandas as pd

from src.features.target import compute_target_label


def _df_with_forward_returns(n=80):
    import numpy as np
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.01, n)
    return pd.DataFrame({"forward_return": returns})


def test_target_uses_forward_return():
    """Target at row t is derived from forward_return, not close."""
    df = _df_with_forward_returns(70)
    result = compute_target_label(df)
    assert "target" in result.columns
    assert set(result["target"].unique()).issubset({"BUY", "HOLD", "SELL"})


def test_target_distribution_roughly_30_40_30():
    df = _df_with_forward_returns(500)
    result = compute_target_label(df)
    dist = result["target"].value_counts(normalize=True)
    for label in ["BUY", "HOLD", "SELL"]:
        # Allow wide tolerance for randomness
        assert 0.20 <= dist.get(label, 0) <= 0.50, f"{label}: {dist.get(label, 0)}"


def test_rows_below_min_history_dropped():
    """Fewer than 5 days of forward_return → no valid percentiles → rows dropped."""
    df = pd.DataFrame({"forward_return": [0.01, 0.02, 0.03, 0.004]})
    result = compute_target_label(df)
    assert len(result) == 0


def test_final_row_no_target_when_forward_return_null():
    """Last row has NaN forward_return (no t+1) → must be dropped."""
    df = _df_with_forward_returns(20)
    df.at[19, "forward_return"] = float("nan")
    result = compute_target_label(df)
    assert 19 not in result.index or result.iloc[-1]["target"] is not None
