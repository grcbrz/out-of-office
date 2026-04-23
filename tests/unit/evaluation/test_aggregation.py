from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.aggregation import aggregate_across_folds, per_ticker_breakdown


def test_aggregate_empty_returns_empty():
    result = aggregate_across_folds([])
    assert result == {}


def test_aggregate_computes_mean_and_std():
    folds = [{"f1_macro": 0.40, "mcc": 0.1}, {"f1_macro": 0.60, "mcc": 0.3}]
    result = aggregate_across_folds(folds)
    assert abs(result["f1_macro_mean"] - 0.50) < 1e-6
    assert "f1_macro_std" in result
    assert "mcc_mean" in result


def test_per_ticker_breakdown():
    y_true = pd.Series(["BUY", "HOLD", "SELL", "BUY"])
    y_pred = pd.Series(["BUY", "HOLD", "SELL", "HOLD"])
    tickers = pd.Series(["AAPL", "AAPL", "MSFT", "MSFT"])
    fwd = pd.Series([0.02, -0.01, -0.03, 0.01])

    results = per_ticker_breakdown(y_true, y_pred, tickers, fwd)
    assert len(results) == 2
    ticker_names = {r["ticker"] for r in results}
    assert ticker_names == {"AAPL", "MSFT"}
