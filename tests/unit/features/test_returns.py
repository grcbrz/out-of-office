from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.returns import compute_forward_return, compute_log_returns


def _df(closes):
    return pd.DataFrame({"close": closes})


def test_log_return_first_row_null():
    df = compute_log_returns(_df([100.0, 110.0]))
    assert pd.isna(df["log_return"].iloc[0])


def test_log_return_computation():
    df = compute_log_returns(_df([100.0, 110.0]))
    assert df["log_return"].iloc[1] == pytest.approx(np.log(110 / 100))


def test_forward_return_last_row_null():
    df = compute_forward_return(_df([100.0, 110.0]))
    assert pd.isna(df["forward_return"].iloc[-1])


def test_forward_return_uses_next_close():
    df = compute_forward_return(_df([100.0, 110.0, 120.0]))
    assert df["forward_return"].iloc[0] == pytest.approx(np.log(110 / 100))
    assert df["forward_return"].iloc[1] == pytest.approx(np.log(120 / 110))
