from __future__ import annotations

import numpy as np
import pandas as pd

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


# ----- confidence threshold -----

class _LowConfidenceModel:
    """Top class is BUY but only at 0.45 — below typical τ values."""

    def predict_proba(self, X):
        return np.array([[0.30, 0.25, 0.45]])


def test_predict_demotes_to_hold_when_below_threshold():
    engine = InferenceEngine(
        model=_LowConfidenceModel(),
        imputation_params={},
        ticker_map={"AAPL": 0},
        confidence_threshold=0.50,
    )
    signal, confidence = engine.predict("AAPL", _make_row())
    assert signal == "HOLD"
    # Reported confidence is the *raw* top-class probability — keeps logging
    # honest even after demotion.
    assert abs(confidence - 0.45) < 1e-6


def test_predict_keeps_buy_when_above_threshold():
    engine = InferenceEngine(
        model=_MockModel(),  # 0.7 BUY
        imputation_params={},
        ticker_map={"AAPL": 0},
        confidence_threshold=0.50,
    )
    signal, confidence = engine.predict("AAPL", _make_row())
    assert signal == "BUY"
    assert abs(confidence - 0.7) < 1e-6


def test_predict_no_threshold_falls_back_to_argmax():
    """Legacy artifacts without τ should not gate anything."""
    engine = InferenceEngine(
        model=_LowConfidenceModel(),
        imputation_params={},
        ticker_map={"AAPL": 0},
        confidence_threshold=None,
    )
    signal, _ = engine.predict("AAPL", _make_row())
    assert signal == "BUY"
