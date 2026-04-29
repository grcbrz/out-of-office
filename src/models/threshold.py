"""Confidence thresholding for BUY / SELL signals.

Why this exists
---------------
A 3-class classifier emits a signal on every row, including rows where the
top class probability is barely above 1/3. Trading on those low-confidence
rows is the dominant source of bad Sharpe in this kind of model: most of the
"alpha" is concentrated in the high-confidence tail; the low-confidence
middle is essentially a coin flip with transaction-cost-free assumptions
that won't survive contact with a broker.

Two functions live here:

* ``apply_confidence_threshold(preds, proba, tau)`` — replaces any
  ``preds[i]`` whose ``max(proba[i]) <= tau`` with the HOLD class. Pure,
  cheap, used both in training (post-fit thresholding for financial metrics)
  and in serving (inference-time gating). No SciPy / scikit-learn deps so
  the serving path stays light.

* ``calibrate_threshold(...)`` — grid-searches τ on a validation set to
  maximise validation Sharpe. The returned τ is what production trades on.

Honesty caveat: τ is calibrated on the same validation window we report
metrics on, which makes the in-fold Sharpe slightly optimistic. With 21-day
val windows splitting further would leave too few rows. The walk-forward
harness mitigates this — across N folds the optimism averages out, and
production trades on next-fold data with the previous fold's τ.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from src.evaluation.financial import compute_financial_metrics

logger = logging.getLogger(__name__)

# Class index for HOLD — must match TARGET_ENCODING in
# src.models.architectures.base. Hardcoded to keep this module dependency-free
# for the serving path.
HOLD_CLASS_INDEX: int = 1
_INT_TO_LABEL = {0: "SELL", 1: "HOLD", 2: "BUY"}

# Default candidate grid. Spans from "no thresholding" (1/3 = uniform proba)
# to "very strict". 1/3 is the lower bound — below that, max(proba) is always
# above τ for a 3-class softmax so the threshold is a no-op.
DEFAULT_CANDIDATE_TAUS: tuple[float, ...] = (
    0.34, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
)
# A τ that demotes more than this fraction of all directional signals to
# HOLD is treated as degenerate (the strategy effectively stops trading).
_MIN_DIRECTIONAL_FRACTION: float = 0.05


def apply_confidence_threshold(
    predictions: np.ndarray,
    predictions_proba: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Demote low-confidence signals to HOLD.

    Args:
        predictions: integer class predictions, shape ``(n,)``.
        predictions_proba: per-class probabilities, shape ``(n, 3)`` in the
            canonical SELL/HOLD/BUY column order.
        tau: confidence threshold. Predictions with ``max(proba) <= tau``
            become HOLD; the rest pass through unchanged. ``tau <= 1/3`` is a
            no-op for 3-class softmax.

    Returns:
        New integer-prediction array. Original ``predictions`` is not modified.
    """
    if predictions_proba.ndim != 2 or predictions_proba.shape[1] != 3:
        raise ValueError(
            f"predictions_proba must have shape (n, 3); got {predictions_proba.shape}"
        )
    if len(predictions) != len(predictions_proba):
        raise ValueError(
            "predictions and predictions_proba must have the same length"
        )
    out = np.asarray(predictions, dtype=int).copy()
    max_proba = predictions_proba.max(axis=1)
    out[max_proba <= tau] = HOLD_CLASS_INDEX
    return out


def calibrate_threshold(
    predictions: np.ndarray,
    predictions_proba: np.ndarray,
    forward_log_returns: pd.Series,
    candidate_taus: Iterable[float] = DEFAULT_CANDIDATE_TAUS,
) -> tuple[float, dict]:
    """Pick τ that maximises validation Sharpe.

    Iterates ``candidate_taus``, applies the threshold, computes financial
    metrics on the resulting strategy, and returns the τ with the best Sharpe.
    Ties broken by preferring **lower** τ (more trades — makes the strategy
    less degenerate when several thresholds are statistically equivalent).

    Args:
        predictions: shape ``(n,)``, integer classes.
        predictions_proba: shape ``(n, 3)``.
        forward_log_returns: forward log returns aligned to predictions.
        candidate_taus: thresholds to search over.

    Returns:
        ``(best_tau, financial_metrics_at_best_tau)``. If every τ is degenerate
        (fewer than ``_MIN_DIRECTIONAL_FRACTION`` of rows trade), the lowest
        candidate τ is returned with whatever metrics it produced — this is a
        flag that the model is not generating meaningful confidence signal,
        and the quality gate should fail downstream.
    """
    candidate_taus = list(candidate_taus)
    if not candidate_taus:
        raise ValueError("candidate_taus must contain at least one value")

    n = len(predictions)
    min_directional = max(1, int(round(n * _MIN_DIRECTIONAL_FRACTION)))

    best_tau: float | None = None
    best_sharpe = -np.inf
    best_metrics: dict | None = None
    fallback_tau = min(candidate_taus)
    fallback_metrics: dict | None = None

    for tau in sorted(candidate_taus):
        thresholded = apply_confidence_threshold(predictions, predictions_proba, tau)
        directional_count = int((thresholded != HOLD_CLASS_INDEX).sum())
        signals_str = pd.Series(thresholded).map(_INT_TO_LABEL)
        metrics = compute_financial_metrics(signals_str, forward_log_returns)

        if tau == fallback_tau:
            fallback_metrics = metrics

        if directional_count < min_directional:
            logger.debug(
                "tau=%.2f produced only %d directional signals (<%d) — skipping",
                tau, directional_count, min_directional,
            )
            continue

        sharpe = metrics["sharpe_ratio"]
        if sharpe is None or np.isnan(sharpe):
            continue

        # Strictly greater so ties favour the *lowest* τ already searched.
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_tau = tau
            best_metrics = metrics

    if best_tau is None:
        logger.warning(
            "no τ in %s produced a non-degenerate strategy; "
            "falling back to τ=%.2f", candidate_taus, fallback_tau,
        )
        return fallback_tau, fallback_metrics or {}

    logger.info(
        "calibrated confidence threshold τ=%.2f (Sharpe=%.3f)",
        best_tau, best_sharpe,
    )
    return best_tau, best_metrics or {}
