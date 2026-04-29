from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

_LABELS = ["SELL", "HOLD", "BUY"]
_LABEL_TO_INT = {"SELL": 0, "HOLD": 1, "BUY": 2}
# Column order in y_proba arrays — matches src.models.architectures.base.TARGET_ENCODING.
_PROBA_COLUMN_ORDER = (0, 1, 2)


def compute_classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_proba: np.ndarray | None = None,
    proba_classes: Sequence[int] | None = None,
) -> dict:
    """F1-macro, per-class F1/precision/recall, MCC, ROC-AUC, confusion matrix.

    ROC-AUC requires class probabilities, not hard predictions. If ``y_proba``
    is omitted the AUC field is set to ``None`` rather than computed against
    one-hot hard labels (which collapses AUC to a function of accuracy and is
    misleading).

    Args:
        y_true: ground-truth string labels (BUY / HOLD / SELL).
        y_pred: predicted string labels.
        y_proba: optional ``(n, 3)`` probability matrix. Column order is
            specified by ``proba_classes``; if omitted, the default
            ``(SELL=0, HOLD=1, BUY=2)`` order is assumed.
        proba_classes: optional sequence of integer class IDs in column order.
            Use this when the wrapper's ``classes_`` attribute differs from
            the default order (e.g. an estimator trained on a fold missing one
            label).
    """
    y_true_enc = y_true.map(_LABEL_TO_INT)
    y_pred_enc = y_pred.map(_LABEL_TO_INT)

    f1_macro = f1_score(y_true_enc, y_pred_enc, average="macro", labels=[0, 1, 2], zero_division=0)
    mcc = matthews_corrcoef(y_true_enc, y_pred_enc)

    report = classification_report(
        y_true_enc, y_pred_enc,
        labels=[0, 1, 2], target_names=_LABELS,
        output_dict=True, zero_division=0,
    )

    roc_auc = _roc_auc_from_proba(y_true_enc, y_proba, proba_classes)

    cm = confusion_matrix(y_true_enc, y_pred_enc, labels=[0, 1, 2]).tolist()

    return {
        "f1_macro": f1_macro,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "per_class": {label: report.get(label, {}) for label in _LABELS},
    }


def _roc_auc_from_proba(
    y_true_enc: pd.Series,
    y_proba: np.ndarray | None,
    proba_classes: Sequence[int] | None,
) -> float | None:
    """Multiclass ROC-AUC (OvR macro). Returns None if probabilities absent or
    the validation fold contains fewer than two classes.
    """
    if y_proba is None:
        return None
    if y_proba.ndim != 2 or y_proba.shape[1] != 3:
        logger.warning("y_proba shape %s; expected (n, 3); skipping ROC-AUC", y_proba.shape)
        return None
    if y_true_enc.nunique() < 2:
        return None

    # Reorder columns so they match (SELL=0, HOLD=1, BUY=2) regardless of how
    # the upstream estimator ordered them.
    if proba_classes is not None:
        try:
            idx = [list(proba_classes).index(c) for c in _PROBA_COLUMN_ORDER]
            y_proba = y_proba[:, idx]
        except ValueError:
            logger.warning(
                "proba_classes=%s does not contain all of (0, 1, 2); skipping ROC-AUC",
                list(proba_classes),
            )
            return None

    try:
        return float(
            roc_auc_score(
                y_true_enc, y_proba,
                multi_class="ovr", average="macro", labels=[0, 1, 2],
            )
        )
    except ValueError as exc:
        # Raised when one class is absent from y_true and ROC-AUC degenerates.
        logger.warning("ROC-AUC computation failed: %s", exc)
        return None
