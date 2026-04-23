from __future__ import annotations

import pytest

from src.models.selector import ModelResult, select_winner


def test_model_selection_f1_macro():
    results = [
        ModelResult("nhits", 0.40),
        ModelResult("patchtst", 0.45),
        ModelResult("autoformer", 0.38),
    ]
    winner = select_winner(results)
    assert winner.model_name == "patchtst"


def test_model_selection_tiebreak():
    """Equal F1 → prefer nhits > patchtst > autoformer."""
    results = [
        ModelResult("autoformer", 0.40),
        ModelResult("patchtst", 0.40),
        ModelResult("nhits", 0.40),
    ]
    winner = select_winner(results)
    assert winner.model_name == "nhits"


def test_select_winner_single():
    results = [ModelResult("nhits", 0.35)]
    winner = select_winner(results)
    assert winner.model_name == "nhits"


def test_select_winner_empty_raises():
    with pytest.raises(ValueError):
        select_winner([])
