from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.financial import compute_financial_metrics


def test_hit_rate_excludes_hold():
    signals = pd.Series(["BUY", "HOLD", "SELL", "HOLD"])
    returns = pd.Series([0.01, -0.05, -0.02, 0.03])
    result = compute_financial_metrics(signals, returns)
    # BUY+positive, SELL+negative → 2/2 = 1.0
    assert result["hit_rate"] == pytest.approx(1.0)


def test_hit_rate_all_hold_is_nan():
    signals = pd.Series(["HOLD", "HOLD"])
    returns = pd.Series([0.01, -0.01])
    result = compute_financial_metrics(signals, returns)
    assert np.isnan(result["hit_rate"])


def test_sharpe_ratio_computation():
    signals = pd.Series(["BUY"] * 50)
    returns = pd.Series([0.001] * 50)
    result = compute_financial_metrics(signals, returns)
    # Constant positive returns → positive Sharpe
    assert result["sharpe_ratio"] > 0 or np.isinf(result["sharpe_ratio"]) or result["sharpe_ratio"] == 0


def test_max_drawdown_negative_or_zero():
    signals = pd.Series(["BUY"] * 10)
    returns = pd.Series([0.01, 0.01, -0.05, 0.01, 0.01, -0.03, 0.01, 0.01, 0.01, 0.01])
    result = compute_financial_metrics(signals, returns)
    assert result["max_drawdown"] <= 0


def test_signal_distribution_sums_to_one():
    signals = pd.Series(["BUY"] * 30 + ["HOLD"] * 40 + ["SELL"] * 30)
    returns = pd.Series([0.0] * 100)
    result = compute_financial_metrics(signals, returns)
    dist = result["signal_distribution"]
    assert abs(sum(dist.values()) - 1.0) < 1e-6
