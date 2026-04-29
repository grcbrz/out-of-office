"""SHAP value computation.

v1 production candidate is LightGBM, so the explainer prefers
``shap.TreeExplainer`` (exact, fast, no sampling). For arbitrary estimators —
including the naive baseline and any future neural model — falls back to
``shap.KernelExplainer`` on a 100-sample background subset.

Multi-class output: ``shap_values`` has shape ``(n_classes, n_samples, n_features)``
for TreeExplainer / KernelExplainer alike. Aggregation methods below collapse
across classes when needed.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_KERNEL_BACKGROUND_SIZE = 100


class SHAPExplainer:
    """Compute SHAP values for a fitted classifier."""

    def __init__(self, model: Any, background: pd.DataFrame) -> None:
        self._model = model
        self._background = background
        self._explainer: Any = None
        self._explainer_type: str = "none"

    def fit(self) -> str:
        """Initialise the explainer. Returns the type that was selected."""
        import shap

        if self._is_tree_model():
            try:
                self._explainer = shap.TreeExplainer(self._model)
                self._explainer_type = "TreeExplainer"
                logger.info("using shap.TreeExplainer for %s", type(self._model).__name__)
                return self._explainer_type
            except Exception as exc:
                logger.warning(
                    "TreeExplainer failed (%s); falling back to KernelExplainer", exc,
                )

        background_sample = self._background.sample(
            n=min(_KERNEL_BACKGROUND_SIZE, len(self._background)), random_state=42,
        )
        predict_fn = (
            self._model.predict_proba
            if hasattr(self._model, "predict_proba") else self._model.predict
        )
        self._explainer = shap.KernelExplainer(predict_fn, background_sample.values)
        self._explainer_type = "KernelExplainer"
        logger.info("using shap.KernelExplainer for %s", type(self._model).__name__)
        return self._explainer_type

    def explain(self, X: pd.DataFrame) -> tuple[np.ndarray, str]:
        """Compute SHAP values for ``X``. Returns (shap_values, explainer_type).

        ``shap_values`` shape is ``(n_classes, n_samples, n_features)`` for
        multiclass problems, or ``(n_samples, n_features)`` for the degenerate
        single-class fallback.
        """
        if self._explainer is None:
            self.fit()
        assert self._explainer is not None
        shap_values = self._explainer.shap_values(X.values)
        return np.array(shap_values), self._explainer_type

    def top_features(
        self,
        shap_values: np.ndarray,
        feature_names: list[str],
        n: int = 5,
        target_class: int | None = None,
    ) -> list[dict]:
        """Top-n features by absolute SHAP value.

        Args:
            shap_values: array as returned by ``explain``.
            feature_names: column names aligned to the last axis.
            n: number of features to return.
            target_class: when SHAP returns a per-class array, restrict to a
                single class (e.g. the predicted class). When ``None``, the
                absolute SHAP is averaged across classes.
        """
        per_feature_abs = self._mean_abs_shap_per_feature(shap_values, target_class)
        signed = self._mean_signed_shap_per_feature(shap_values, target_class)
        order = np.argsort(per_feature_abs)[::-1][:n]
        return [
            {"feature": feature_names[i], "shap_value": float(signed[i])}
            for i in order
        ]

    # ------------------------------------------------------------------

    def _is_tree_model(self) -> bool:
        """Heuristic check for a SHAP-supported tree booster.

        Covers LightGBM, XGBoost, CatBoost, and sklearn ensemble trees. The
        ``BaselineLastDirectionWrapper`` is a thin Python class with no booster
        and is correctly excluded.
        """
        cls = type(self._model).__name__.lower()
        tree_markers = ("lgbm", "xgb", "catboost", "forest", "tree", "boosting", "booster")
        if any(m in cls for m in tree_markers):
            return True
        # LightGBM exposes booster_; sklearn trees expose estimators_.
        return hasattr(self._model, "booster_") or hasattr(self._model, "estimators_")

    @staticmethod
    def _mean_abs_shap_per_feature(
        shap_values: np.ndarray, target_class: int | None,
    ) -> np.ndarray:
        if shap_values.ndim == 3:
            if target_class is not None:
                return np.abs(shap_values[target_class]).mean(axis=0)
            return np.abs(shap_values).mean(axis=(0, 1))
        if shap_values.ndim == 2:
            return np.abs(shap_values).mean(axis=0)
        return np.abs(shap_values)

    @staticmethod
    def _mean_signed_shap_per_feature(
        shap_values: np.ndarray, target_class: int | None,
    ) -> np.ndarray:
        if shap_values.ndim == 3:
            if target_class is not None:
                return shap_values[target_class].mean(axis=0)
            return shap_values.mean(axis=(0, 1))
        if shap_values.ndim == 2:
            return shap_values.mean(axis=0)
        return shap_values
