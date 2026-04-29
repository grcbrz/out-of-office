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
        self._explainer_type: str = "none"

    def explain(self, feature_row: pd.Series) -> dict:
        """Return SHAP top-5 features for a single prediction row."""
        try:
            if self._shap_explainer is None:
                self._init_explainer()
            if self._shap_explainer is None:
                return {"top_features": [], "explainer_used": "none"}

            cols = [c for c in self._trained_features if c in feature_row.index]
            X = feature_row[cols].fillna(0.0).values.reshape(1, -1).astype(float)

            raw = self._shap_explainer.shap_values(X)

            # Normalise to (n_features,) signed values for the predicted-class slice.
            # TreeExplainer multiclass → ndarray (1, n_features, n_classes) in SHAP ≥0.41
            # KernelExplainer multiclass → list of n_classes arrays each (1, n_features)
            if isinstance(raw, np.ndarray) and raw.ndim == 3:
                # shape (1, n_features, n_classes) — use mean abs across classes for ranking,
                # keep signed BUY slice for the reported value
                abs_vals = np.abs(raw[0]).mean(axis=1)       # (n_features,)
                signed_vals = raw[0, :, 2] if raw.shape[2] > 2 else raw[0, :, 0]
            elif isinstance(raw, list):
                # list[n_classes] of (1, n_features)
                stacked = np.array([c[0] for c in raw])      # (n_classes, n_features)
                abs_vals = np.abs(stacked).mean(axis=0)
                signed_vals = stacked[2] if len(raw) > 2 else stacked[0]
            elif isinstance(raw, np.ndarray) and raw.ndim == 2:
                abs_vals = np.abs(raw[0])
                signed_vals = raw[0]
            else:
                abs_vals = np.abs(raw)
                signed_vals = raw

            top_idx = np.argsort(abs_vals)[::-1][:5]
            top_features = [
                {"feature": cols[i], "shap_value": float(signed_vals[i])}
                for i in top_idx if i < len(cols)
            ]
            return {"top_features": top_features, "explainer_used": self._explainer_type}

        except Exception as e:
            logger.warning("SHAP explanation failed: %s", e)
            return {"top_features": [], "explainer_used": "none"}

    def _init_explainer(self) -> None:
        """Initialise SHAP explainer. TreeExplainer for LightGBM/tree models; KernelExplainer fallback."""
        import shap

        self._explainer_type = "none"
        # booster_ → LightGBM/XGBoost; estimators_ → sklearn forests
        is_tree = hasattr(self._model, "booster_") or hasattr(self._model, "estimators_")
        if is_tree:
            try:
                self._shap_explainer = shap.TreeExplainer(self._model)
                self._explainer_type = "TreeExplainer"
                logger.info("SHAP TreeExplainer initialised for %s", type(self._model).__name__)
                return
            except Exception as exc:
                logger.warning("TreeExplainer failed (%s); trying KernelExplainer", exc)

        if hasattr(self._model, "predict_proba"):
            try:
                background = np.zeros((10, len(self._trained_features)))
                self._shap_explainer = shap.KernelExplainer(
                    self._model.predict_proba, background
                )
                self._explainer_type = "KernelExplainer"
                logger.info("SHAP KernelExplainer initialised for %s", type(self._model).__name__)
            except Exception as exc:
                logger.warning("KernelExplainer also failed: %s", exc)
