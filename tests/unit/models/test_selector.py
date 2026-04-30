from __future__ import annotations

import pytest

from src.models.selector import ModelResult, select_winner


def test_model_selection_f1_macro():
    """Highest F1 wins regardless of preference order."""
    results = [
        ModelResult("lightgbm", 0.40),
        ModelResult("future_transformer", 0.45),
    ]
    winner = select_winner(results)
    assert winner.model_name == "future_transformer"


def test_model_selection_tiebreak_prefers_lightgbm():
    """Equal F1 → preference order respected (lightgbm beats unknown name)."""
    results = [
        ModelResult("future_transformer", 0.40),
        ModelResult("lightgbm", 0.40),
    ]
    winner = select_winner(results)
    assert winner.model_name == "lightgbm"


def test_tiebreak_prefers_lightgbm_over_randomforest():
    """Equal F1 between the two production candidates → lightgbm wins.

    Locks the contract documented in Spec 04 §6 and `_PREFERENCE_ORDER`:
    boosting preferred over bagging when neither has a measurable F1 edge,
    because LightGBM trains faster and tends to be more sample-efficient on
    tabular daily-equity data.
    """
    results = [
        ModelResult("randomforest", 0.40),
        ModelResult("lightgbm", 0.40),
    ]
    winner = select_winner(results)
    assert winner.model_name == "lightgbm"


def test_randomforest_wins_when_strictly_higher_f1():
    """The preference is a tie-break only — strict F1 advantage always wins."""
    results = [
        ModelResult("lightgbm", 0.40),
        ModelResult("randomforest", 0.42),
    ]
    winner = select_winner(results)
    assert winner.model_name == "randomforest"


def test_select_winner_single():
    results = [ModelResult("lightgbm", 0.35)]
    winner = select_winner(results)
    assert winner.model_name == "lightgbm"


def test_select_winner_empty_raises():
    with pytest.raises(ValueError):
        select_winner([])


def test_select_winner_excludes_baseline_by_flag():
    """A baseline with the highest F1 must not be picked as the production winner."""
    results = [
        ModelResult("lightgbm", 0.40),
        ModelResult("baseline_last_direction", 0.55, is_baseline=True),
    ]
    winner = select_winner(results)
    assert winner.model_name == "lightgbm"


def test_select_winner_excludes_baseline_by_name():
    """Even without is_baseline=True, the baseline name list filters it out."""
    results = [
        ModelResult("lightgbm", 0.40),
        ModelResult("baseline_last_direction", 0.50),  # name match
    ]
    winner = select_winner(results)
    assert winner.model_name == "lightgbm"


def test_select_winner_only_baselines_raises():
    with pytest.raises(ValueError, match="baselines excluded"):
        select_winner([ModelResult("baseline_last_direction", 0.50, is_baseline=True)])
