from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.schema import FEATURE_COLUMNS
from src.serving.inference import InferenceEngine


class _MockModel:
    def predict_proba(self, X):
        return np.array([[0.1, 0.2, 0.7]])  # BUY wins (index 2)


class _MockModelNoProba:
    pass  # no predict_proba


def _make_row():
    return pd.Series({col: 0.0 for col in FEATURE_COLUMNS})


def test_predict_returns_signal_and_confidence():
    engine = InferenceEngine(
        model=_MockModel(),
        imputation_params={},
        ticker_map={"AAPL": 0},
    )
    signal, confidence = engine.predict("AAPL", _make_row())
    assert signal == "BUY"
    assert abs(confidence - 0.7) < 1e-6


def test_predict_unknown_ticker_uses_minus_one():
    engine = InferenceEngine(
        model=_MockModel(),
        imputation_params={},
        ticker_map={"AAPL": 0},
    )
    signal, confidence = engine.predict("UNKNOWN", _make_row())
    # Should still return a valid signal (ticker_id = -1 but model still runs)
    assert signal in ("BUY", "HOLD", "SELL")


def test_predict_fallback_uniform_distribution():
    engine = InferenceEngine(
        model=_MockModelNoProba(),
        imputation_params={},
        ticker_map={"AAPL": 0},
    )
    signal, confidence = engine.predict("AAPL", _make_row())
    assert signal in ("BUY", "HOLD", "SELL")
    assert abs(confidence - 1 / 3) < 1e-6
