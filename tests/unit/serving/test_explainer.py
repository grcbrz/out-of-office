from __future__ import annotations

import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from src.serving.explainer import ServingExplainer


def test_explain_returns_top_features_structure():
    """Explainer returns dict with top_features list and explainer_used."""
    model = MagicMock()
    model.predict = MagicMock(return_value=np.array([1]))

    features = ["f1", "f2", "f3"]
    explainer = ServingExplainer(model=model, trained_features=features)

    row = pd.Series({"f1": 0.1, "f2": 0.2, "f3": 0.3})
    result = explainer.explain(row)

    assert "top_features" in result
    assert "explainer_used" in result
    assert isinstance(result["top_features"], list)


def test_explain_returns_empty_on_failure():
    """Explainer returns empty list and 'none' when it fails."""
    model = MagicMock()
    model.predict = MagicMock(side_effect=Exception("broken"))

    explainer = ServingExplainer(model=model, trained_features=["f1"])
    row = pd.Series({"f1": 0.5})
    result = explainer.explain(row)

    assert result["top_features"] == []
    assert result["explainer_used"] == "none"


def test_explain_handles_missing_features():
    """Explainer gracefully handles feature rows missing some columns."""
    model = MagicMock()
    model.predict = MagicMock(return_value=np.array([1]))

    features = ["f1", "f2", "f3"]
    explainer = ServingExplainer(model=model, trained_features=features)

    # Row missing f2
    row = pd.Series({"f1": 0.1, "f3": 0.3})
    result = explainer.explain(row)

    # Should not crash
    assert isinstance(result, dict)
    assert "top_features" in result
