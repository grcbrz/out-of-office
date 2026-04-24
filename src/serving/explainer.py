from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ServingExplainer:
    """Per-prediction SHAP explainer for the serving layer.

    Loads SHAP explainer from evaluation artifacts if available,
    otherwise returns empty explanations.
    """

    def __init__(self, model: Any, trained_features: list[str]) -> None:
        self._model = model
        self._trained_features = trained_features
        self._shap_explainer: Any = None

    def explain(self, feature_row: pd.Series) -> dict:
        """Return SHAP top-5 features for a single prediction row."""
        try:
            if self._shap_explainer is None:
                self._init_explainer()
            if self._shap_explainer is None:
                return {"top_features": [], "explainer_used": "none"}

            # Prepare features in correct order
            cols = [c for c in self._trained_features if c in feature_row.index]
            X = feature_row[cols].values.reshape(1, -1).astype(float)

            shap_values = self._shap_explainer.shap_values(X)

            # TreeExplainer returns (n_samples, n_features, n_classes) for multiclass
            # KernelExplainer returns (n_samples, n_features) or list of arrays
            if isinstance(shap_values, np.ndarray):
                if shap_values.ndim == 3:
                    # Multiclass from TreeExplainer: shape (1, n_features, n_classes)
                    # Use class 2 (BUY) by convention
                    shap_vals = np.abs(shap_values[0, :, 2] if shap_values.shape[2] > 2 else shap_values[0, :, 0])
                elif shap_values.ndim == 2:
                    # Binary or single sample: shape (n_features,) or (1, n_features)
                    shap_vals = np.abs(shap_values[0]) if shap_values.shape[0] == 1 else np.abs(shap_values)
                else:
                    shap_vals = np.abs(shap_values)
            elif isinstance(shap_values, list):
                # KernelExplainer multiclass output: list per class
                shap_vals = np.abs(shap_values[2][0] if len(shap_values) > 2 else shap_values[0][0])
            else:
                shap_vals = np.abs(shap_values)
            top_indices = np.argsort(shap_vals)[::-1][:5]
            top_features = [
                {"feature": cols[i], "shap_value": float(shap_vals[i])}
                for i in top_indices if i < len(cols)
            ]

            return {"top_features": top_features, "explainer_used": "TreeExplainer"}

        except Exception as e:
            logger.warning("SHAP explanation failed: %s", e)
            return {"top_features": [], "explainer_used": "none"}

    def _init_explainer(self) -> None:
        """Initialize SHAP explainer if possible."""
        try:
            import shap
            # For sklearn tree/ensemble models, use TreeExplainer (much faster than KernelExplainer)
            if hasattr(self._model, "estimators_"):  # ExtraTreesClassifier, RandomForest, etc.
                self._shap_explainer = shap.TreeExplainer(self._model)
            elif hasattr(self._model, "predict"):
                # Fallback to KernelExplainer with minimal background
                self._shap_explainer = shap.KernelExplainer(
                    self._model.predict,
                    np.zeros((10, len(self._trained_features))),
                )
        except Exception as e:
            logger.warning("Could not initialize SHAP explainer: %s", e)
            self._shap_explainer = None
