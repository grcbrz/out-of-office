from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.threshold import (
    DEFAULT_CANDIDATE_TAUS,
    HOLD_CLASS_INDEX,
    apply_confidence_threshold,
    calibrate_threshold,
)


# ----- apply_confidence_threshold -----

def test_apply_threshold_demotes_low_confidence():
    """Predictions whose top probability ≤ τ become HOLD; others pass through."""
    preds = np.array([0, 2, 1, 0, 2])
    proba = np.array([
        [0.7, 0.2, 0.1],   # high confidence SELL → keep
        [0.2, 0.2, 0.6],   # high confidence BUY  → keep
        [0.4, 0.5, 0.1],   # already HOLD         → unchanged
        [0.4, 0.4, 0.2],   # tied → demoted
        [0.34, 0.33, 0.33],  # barely above 1/3 → demoted at τ=0.50
    ])
    out = apply_confidence_threshold(preds, proba, tau=0.50)
    assert out.tolist() == [0, 2, 1, HOLD_CLASS_INDEX, HOLD_CLASS_INDEX]


def test_apply_threshold_below_uniform_is_noop():
    """τ ≤ 1/3 cannot demote any prediction in a 3-class softmax."""
    preds = np.array([0, 1, 2])
    proba = np.array([[0.4, 0.3, 0.3], [0.34, 0.34, 0.32], [0.33, 0.33, 0.34]])
    np.testing.assert_array_equal(
        apply_confidence_threshold(preds, proba, tau=0.33),
        preds,
    )


def test_apply_threshold_does_not_mutate_input():
    preds = np.array([0, 0])  # both SELL; row 1 has high confidence → kept
    proba = np.array([[0.4, 0.3, 0.3], [0.7, 0.2, 0.1]])
    out = apply_confidence_threshold(preds, proba, tau=0.5)
    assert preds.tolist() == [0, 0]  # original untouched
    assert out.tolist() == [HOLD_CLASS_INDEX, 0]  # low-conf demoted, high-conf kept


def test_apply_threshold_validates_proba_shape():
    with pytest.raises(ValueError, match="shape"):
        apply_confidence_threshold(
            np.array([0]), np.array([[0.5, 0.5]]), tau=0.5,
        )


def test_apply_threshold_validates_lengths_match():
    with pytest.raises(ValueError, match="same length"):
        apply_confidence_threshold(
            np.array([0]),
            np.array([[0.4, 0.3, 0.3], [0.4, 0.3, 0.3]]),
            tau=0.5,
        )


# ----- calibrate_threshold -----

def _strategic_dataset() -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Construct val data where high-confidence calls win, low-confidence lose.

    Three correct high-confidence BUYs (max_proba=0.80) followed by three
    losing low-confidence BUYs (max_proba=0.45). At τ=0.50 the strategy
    keeps only the winners, so Sharpe is much higher than at τ=0.34 where
    the losers are still traded.
    """
    proba = np.tile(np.array([0.05, 0.15, 0.80]), (3, 1)).tolist()
    proba += np.tile(np.array([0.30, 0.25, 0.45]), (3, 1)).tolist()
    proba_arr = np.array(proba)
    preds = np.array([2, 2, 2, 2, 2, 2])  # all BUY
    log_returns = pd.Series([0.02, 0.025, 0.018, -0.05, -0.04, -0.03])
    return preds, proba_arr, log_returns


def test_calibrate_picks_threshold_that_beats_no_threshold():
    preds, proba, log_returns = _strategic_dataset()
    tau, metrics = calibrate_threshold(preds, proba, log_returns,
                                       candidate_taus=(0.34, 0.50))
    assert tau == 0.50
    # At τ=0.50 only the three winning trades survive — Sharpe must be positive.
    assert metrics["sharpe_ratio"] > 0.0


def test_calibrate_lowest_tau_on_tie():
    """Two τ values that produce identical strategy series → the lower one wins."""
    # Build val data where τ=0.40 and τ=0.50 keep exactly the same trades —
    # all four high-confidence BUYs have max proba = 0.55 (above both thresholds);
    # there are no other predictions, so both thresholds behave identically.
    proba = np.tile(np.array([0.20, 0.25, 0.55]), (4, 1))
    preds = np.array([2, 2, 2, 2])
    log_returns = pd.Series([0.01, 0.02, -0.005, 0.015])
    tau, _ = calibrate_threshold(preds, proba, log_returns,
                                 candidate_taus=(0.40, 0.50))
    assert tau == 0.40


def test_calibrate_falls_back_when_every_tau_degenerates():
    """If every τ leaves <5% directional trades, the lowest candidate is returned."""
    # Two BUYs with low confidence; at any τ ≥ 0.40 fewer than 5% remain (we
    # have only 2 rows, so 5% is 1 — but min_directional is max(1, ...), so we
    # need a bigger n to actually trip the gate).
    n = 100
    proba = np.tile(np.array([0.34, 0.33, 0.33]), (n, 1))
    preds = np.zeros(n, dtype=int)
    preds[:2] = 2  # two BUYs only
    log_returns = pd.Series(np.zeros(n))
    tau, _ = calibrate_threshold(preds, proba, log_returns,
                                 candidate_taus=(0.40, 0.50, 0.60))
    # All taus demote the two BUYs → 0 trades → fallback to lowest.
    assert tau == 0.40


def test_calibrate_default_grid_is_sane():
    """Sanity check on the shipped default grid."""
    assert min(DEFAULT_CANDIDATE_TAUS) > 1 / 3  # nothing below uniform
    assert max(DEFAULT_CANDIDATE_TAUS) < 1.0   # nothing above 100% confidence
    assert sorted(DEFAULT_CANDIDATE_TAUS) == list(DEFAULT_CANDIDATE_TAUS)


def test_calibrate_empty_grid_raises():
    preds = np.array([0])
    proba = np.array([[0.4, 0.3, 0.3]])
    with pytest.raises(ValueError, match="at least one"):
        calibrate_threshold(preds, proba, pd.Series([0.0]), candidate_taus=())
