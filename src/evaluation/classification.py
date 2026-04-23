from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

_LABELS = ["SELL", "HOLD", "BUY"]
_LABEL_TO_INT = {"SELL": 0, "HOLD": 1, "BUY": 2}


def compute_classification_metrics(
    y_true: pd.Series, y_pred: pd.Series
) -> dict:
    """Compute F1-macro, per-class F1/precision/recall, MCC, ROC-AUC, confusion matrix."""
    y_true_enc = y_true.map(_LABEL_TO_INT)
    y_pred_enc = y_pred.map(_LABEL_TO_INT)

    f1_macro = f1_score(y_true_enc, y_pred_enc, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true_enc, y_pred_enc)

    report = classification_report(
        y_true_enc, y_pred_enc,
        labels=[0, 1, 2], target_names=_LABELS,
        output_dict=True, zero_division=0,
    )

    try:
        y_true_bin = pd.get_dummies(y_true).reindex(columns=_LABELS, fill_value=0)
        y_pred_bin = pd.get_dummies(y_pred).reindex(columns=_LABELS, fill_value=0)
        roc_auc = roc_auc_score(y_true_bin, y_pred_bin, multi_class="ovr", average="macro")
    except Exception:
        roc_auc = None

    cm = confusion_matrix(y_true_enc, y_pred_enc, labels=[0, 1, 2]).tolist()

    return {
        "f1_macro": f1_macro,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "per_class": {label: report.get(label, {}) for label in _LABELS},
    }
