from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.classification import compute_classification_metrics


def _perfect(n=30):
    labels = (["BUY"] * 10 + ["HOLD"] * 10 + ["SELL"] * 10)[:n]
    return pd.Series(labels), pd.Series(labels)


def test_f1_macro_computation():
    y_true, y_pred = _perfect()
    m = compute_classification_metrics(y_true, y_pred)
    assert m["f1_macro"] == pytest.approx(1.0)


def test_mcc_computation():
    y_true, y_pred = _perfect()
    m = compute_classification_metrics(y_true, y_pred)
    assert m["mcc"] == pytest.approx(1.0)


def test_confusion_matrix_shape():
    y_true = pd.Series(["BUY", "HOLD", "SELL", "BUY"])
    y_pred = pd.Series(["BUY", "HOLD", "SELL", "HOLD"])
    m = compute_classification_metrics(y_true, y_pred)
    cm = m["confusion_matrix"]
    assert len(cm) == 3
    assert all(len(row) == 3 for row in cm)


def test_per_class_keys_present():
    y_true, y_pred = _perfect()
    m = compute_classification_metrics(y_true, y_pred)
    for label in ["BUY", "HOLD", "SELL"]:
        assert label in m["per_class"]
