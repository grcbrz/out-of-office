from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.classification import compute_classification_metrics


def aggregate_across_folds(fold_metrics: list[dict]) -> dict:
    """Compute mean and std of each metric across folds."""
    if not fold_metrics:
        return {}
    keys = [k for k in fold_metrics[0] if isinstance(fold_metrics[0][k], (int, float))]
    result: dict = {}
    for k in keys:
        values = [m[k] for m in fold_metrics if m.get(k) is not None]
        result[f"{k}_mean"] = float(np.mean(values))
        result[f"{k}_std"] = float(np.std(values))
    return result


def per_ticker_breakdown(
    y_true: pd.Series,
    y_pred: pd.Series,
    tickers: pd.Series,
    forward_returns: pd.Series,
) -> list[dict]:
    """Compute per-ticker F1-macro, hit rate, and signal distribution."""
    from src.evaluation.financial import compute_financial_metrics

    results = []
    for ticker in tickers.unique():
        mask = tickers == ticker
        if mask.sum() == 0:
            continue
        metrics = compute_classification_metrics(y_true[mask], y_pred[mask])
        fin = compute_financial_metrics(y_pred[mask], forward_returns[mask])
        results.append({
            "ticker": ticker,
            "f1_macro": metrics["f1_macro"],
            "hit_rate": fin["hit_rate"],
            "signal_distribution": fin["signal_distribution"],
        })
    return results
