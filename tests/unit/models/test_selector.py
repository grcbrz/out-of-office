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
    """Equal F1 → preference order respected."""
    results = [
        ModelResult("future_transformer", 0.40),
        ModelResult("lightgbm", 0.40),
    ]
    winner = select_winner(results)
    assert winner.model_name == "lightgbm"


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
