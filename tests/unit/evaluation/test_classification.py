from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.classification import compute_classification_metrics


def _perfect(n: int = 30) -> tuple[pd.Series, pd.Series]:
    labels = (["BUY"] * 10 + ["HOLD"] * 10 + ["SELL"] * 10)[:n]
    return pd.Series(labels), pd.Series(labels)


# ----- legacy assertions (still hold) -----

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


# ----- new ROC-AUC behaviour -----

def test_roc_auc_is_none_without_proba():
    y_true, y_pred = _perfect()
    m = compute_classification_metrics(y_true, y_pred)
    assert m["roc_auc"] is None


def test_roc_auc_finite_with_perfect_proba():
    """Perfect probabilities → AUC = 1.0."""
    labels = ["SELL", "HOLD", "BUY"] * 10
    y = pd.Series(labels)
    # One-hot probabilities aligned with each true class (column order SELL/HOLD/BUY).
    proba = np.zeros((len(labels), 3))
    for i, lab in enumerate(labels):
        proba[i, {"SELL": 0, "HOLD": 1, "BUY": 2}[lab]] = 1.0
    m = compute_classification_metrics(y, y, y_proba=proba)
    assert m["roc_auc"] == pytest.approx(1.0)


def test_roc_auc_handles_alt_proba_class_order():
    """If wrapper.classes_ is (BUY=2, HOLD=1, SELL=0), columns should be reordered."""
    labels = ["SELL", "HOLD", "BUY"] * 10
    y = pd.Series(labels)
    canonical = np.zeros((len(labels), 3))
    for i, lab in enumerate(labels):
        canonical[i, {"SELL": 0, "HOLD": 1, "BUY": 2}[lab]] = 1.0
    # Reverse to BUY/HOLD/SELL column order.
    reversed_proba = canonical[:, [2, 1, 0]]
    m = compute_classification_metrics(
        y, y,
        y_proba=reversed_proba,
        proba_classes=(2, 1, 0),
    )
    assert m["roc_auc"] == pytest.approx(1.0)


def test_roc_auc_none_when_only_one_true_class():
    """ROC-AUC is undefined if all true labels are the same class."""
    y = pd.Series(["HOLD"] * 10)
    proba = np.tile([0.0, 1.0, 0.0], (10, 1))
    m = compute_classification_metrics(y, y, y_proba=proba)
    assert m["roc_auc"] is None


def test_roc_auc_skipped_for_wrong_shape():
    y_true, y_pred = _perfect()
    proba = np.zeros((len(y_true), 2))  # wrong number of columns
    m = compute_classification_metrics(y_true, y_pred, y_proba=proba)
    assert m["roc_auc"] is None
