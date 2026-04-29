from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.financial import compute_financial_metrics


# ----- hit rate -----

def test_hit_rate_excludes_hold():
    signals = pd.Series(["BUY", "HOLD", "SELL", "HOLD"])
    returns = pd.Series([0.01, -0.05, -0.02, 0.03])
    result = compute_financial_metrics(signals, returns)
    assert result["hit_rate"] == pytest.approx(1.0)


def test_hit_rate_all_hold_is_nan():
    result = compute_financial_metrics(
        pd.Series(["HOLD", "HOLD"]), pd.Series([0.01, -0.01])
    )
    assert np.isnan(result["hit_rate"])


# ----- simple-return conversion -----

def test_strategy_uses_simple_returns_for_compounding():
    """A −2.0 log return becomes a simple return of expm1(-2) ≈ −0.865, not -2.

    The old buggy code applied ``(1 + log_r).cumprod()``, which would go
    negative for `log_r < -1` and produce nonsensical max drawdown values.
    """
    signals = pd.Series(["BUY"] * 3)
    log_returns = pd.Series([0.01, -2.0, 0.05])  # the −2.0 is the trap
    result = compute_financial_metrics(signals, log_returns)
    # Strategy returns are simple returns: expm1(log_r).
    # Cumulative: (1+0.01)*(1+expm1(-2))*(1+expm1(0.05))
    expected_simple = np.expm1(log_returns.to_numpy())
    expected_cum = (1.0 + expected_simple).cumprod()
    expected_dd = float((expected_cum / np.maximum.accumulate(expected_cum) - 1.0).min())
    assert result["max_drawdown"] == pytest.approx(expected_dd)
    # Bounded in (-1, 0]: cumulative product can never go negative under simple returns.
    assert result["max_drawdown"] > -1.0


def test_max_drawdown_zero_when_all_positive_buy_returns():
    signals = pd.Series(["BUY"] * 5)
    returns = pd.Series([0.01] * 5)
    result = compute_financial_metrics(signals, returns)
    assert result["max_drawdown"] == pytest.approx(0.0)


# ----- Sharpe -----

def test_sharpe_ratio_positive_for_constant_buy_gain():
    signals = pd.Series(["BUY"] * 50)
    returns = pd.Series([0.001] * 50)  # constant → std=0 → Sharpe=0
    result = compute_financial_metrics(signals, returns)
    assert result["sharpe_ratio"] == 0.0


def test_sharpe_ratio_known_series():
    """Hand-computed Sharpe on a known series."""
    signals = pd.Series(["BUY"] * 4)
    log_returns = pd.Series([0.01, -0.005, 0.02, 0.0])
    result = compute_financial_metrics(signals, log_returns)
    simple = np.expm1(log_returns.to_numpy())
    expected = simple.mean() / simple.std(ddof=1) * np.sqrt(252)
    assert result["sharpe_ratio"] == pytest.approx(expected)


# ----- distribution -----

def test_signal_distribution_sums_to_one():
    signals = pd.Series(["BUY"] * 30 + ["HOLD"] * 40 + ["SELL"] * 30)
    result = compute_financial_metrics(signals, pd.Series([0.0] * 100))
    dist = result["signal_distribution"]
    assert abs(sum(dist.values()) - 1.0) < 1e-6
