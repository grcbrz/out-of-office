from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.models.preparation import DataPreparer, TrainingDataError


def _df(tickers=None, n=10):
    if tickers is None:
        tickers = ["AAPL"]
    rows = []
    for t in tickers:
        for i in range(n):
            rows.append({"ticker": t, "target": "HOLD", "bullish_percent": None, "close_zscore": 1.0})
    return pd.DataFrame(rows)


def test_ticker_encoding_stable(tmp_path):
    preparer = DataPreparer(tmp_path / "ticker_map.json")
    df = _df(["AAPL", "MSFT"])
    df1 = preparer.encode_tickers(df)
    df2 = preparer.encode_tickers(df)
    # Same tickers → same IDs across two calls
    assert list(df1["ticker_id"]) == list(df2["ticker_id"])


def test_ticker_encoding_append(tmp_path):
    preparer = DataPreparer(tmp_path / "ticker_map.json")
    df1 = _df(["AAPL"])
    df2 = _df(["MSFT"])
    preparer.encode_tickers(df1)
    aapl_id = preparer._ticker_map["AAPL"]
    preparer.encode_tickers(df2)
    # AAPL ID must not change after adding MSFT
    assert preparer._ticker_map["AAPL"] == aapl_id
    assert "MSFT" in preparer._ticker_map


def test_imputation_train_only(tmp_path):
    """Median imputed from train fold; applied to val without re-computing."""
    preparer = DataPreparer(tmp_path / "ticker_map.json")
    train_df = pd.DataFrame({"bullish_percent": [0.5, 0.6, None, 0.7], "target": ["HOLD"] * 4})
    val_df = pd.DataFrame({"bullish_percent": [None, 0.4], "target": ["HOLD"] * 2})

    preparer.fit_imputation(train_df)
    result = preparer.apply_imputation(val_df)
    # Val null filled with train median
    expected_median = pd.Series([0.5, 0.6, 0.7]).median()
    assert result["bullish_percent"].iloc[0] == pytest.approx(expected_median)


def test_nan_after_imputation_raises(tmp_path):
    preparer = DataPreparer(tmp_path / "ticker_map.json")
    # No imputation fitted → NaN remains in feature columns
    from src.features.schema import FEATURE_COLUMNS
    df = pd.DataFrame({col: [None] for col in ["bullish_percent"]})
    # fit with nothing
    preparer._imputation_params = {}
    # don't mark bullish_percent as needing imputation → no fill → NaN remains
    # but FEATURE_COLUMNS includes bullish_percent
    if "bullish_percent" in FEATURE_COLUMNS:
        with pytest.raises(TrainingDataError):
            preparer.apply_imputation(df)


def test_class_weights_per_fold(tmp_path):
    preparer = DataPreparer(tmp_path / "ticker_map.json")
    targets = pd.Series(["BUY"] * 30 + ["HOLD"] * 40 + ["SELL"] * 30)
    weights = preparer.compute_class_weights(targets)
    # Weights should be inverse to frequency
    assert weights[1] < weights[0]  # HOLD (freq 40%) should weigh less than BUY (30%)
    assert weights[1] < weights[2]  # HOLD should weigh less than SELL (30%)
