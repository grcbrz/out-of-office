"""Selection-logic tests for the SHAP explainer.

Spies on ``shap.TreeExplainer`` and ``shap.KernelExplainer`` so we don't
incur the cost of actual SHAP computation — the goal is to verify the
explainer-type selection contract documented in
``src/evaluation/explainability/shap_explainer.py``.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def background():
    return pd.DataFrame(np.random.default_rng(0).standard_normal((40, 5)))


def _spy_shap(monkeypatch):
    import shap

    tree_spy = MagicMock(name="TreeExplainer", return_value=MagicMock(spec=shap.TreeExplainer))
    kernel_spy = MagicMock(name="KernelExplainer", return_value=MagicMock(spec=shap.KernelExplainer))
    monkeypatch.setattr(shap, "TreeExplainer", tree_spy)
    monkeypatch.setattr(shap, "KernelExplainer", kernel_spy)
    return tree_spy, kernel_spy


def test_tree_explainer_selected_for_lightgbm(background, monkeypatch):
    """A trained LightGBM booster must be detected as a tree model."""
    from lightgbm import LGBMClassifier
    from src.evaluation.explainability.shap_explainer import SHAPExplainer

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((60, 5)))
    y = rng.integers(0, 3, size=60)
    model = LGBMClassifier(n_estimators=5, num_leaves=4, verbose=-1)
    model.fit(X, y)

    tree_spy, kernel_spy = _spy_shap(monkeypatch)
    explainer = SHAPExplainer(model, background)
    selected = explainer.fit()
    assert selected == "TreeExplainer"
    tree_spy.assert_called_once()
    kernel_spy.assert_not_called()


def test_kernel_explainer_fallback_for_non_tree(background, monkeypatch):
    """A model with no tree structure routes to KernelExplainer."""
    from src.evaluation.explainability.shap_explainer import SHAPExplainer

    class PlainClassifier:
        def predict_proba(self, X):
            return np.tile([1 / 3, 1 / 3, 1 / 3], (len(X), 1))

    tree_spy, kernel_spy = _spy_shap(monkeypatch)
    explainer = SHAPExplainer(PlainClassifier(), background)
    selected = explainer.fit()
    assert selected == "KernelExplainer"
    tree_spy.assert_not_called()
    kernel_spy.assert_called_once()


def test_top_features_orders_by_absolute_shap():
    """top_features is independent of the explainer choice."""
    from src.evaluation.explainability.shap_explainer import SHAPExplainer

    explainer = SHAPExplainer(model=MagicMock(), background=pd.DataFrame())
    # 3 classes × 4 samples × 5 features. Feature 2 has the largest |SHAP|.
    shap_values = np.zeros((3, 4, 5))
    shap_values[:, :, 2] = 0.5
    shap_values[:, :, 0] = 0.1
    feature_names = ["f0", "f1", "f2", "f3", "f4"]
    top = explainer.top_features(shap_values, feature_names, n=2)
    assert top[0]["feature"] == "f2"
    assert top[1]["feature"] == "f0"
