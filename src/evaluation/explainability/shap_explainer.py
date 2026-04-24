from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """Computes SHAP values for model predictions.

    Attempts DeepExplainer first; falls back to KernelExplainer on failure.
    N-HiTS, PatchTST, and Autoformer are all supported (or their mocks in tests).
    """

    def __init__(self, model: Any, background: pd.DataFrame) -> None:
        self._model = model
        self._background = background
        self._explainer = None
        self._explainer_type: str = "none"

    def fit(self) -> str:
        """Initialize the explainer. Returns which type was selected."""
        try:
            import shap
            self._explainer = shap.DeepExplainer(self._model, self._background.values)
            self._explainer_type = "DeepExplainer"
        except Exception as e:
            logger.warning("DeepExplainer failed (%s); falling back to KernelExplainer", e)
            import shap
            background_sample = self._background.sample(
                min(100, len(self._background)), random_state=42
            )
            self._explainer = shap.KernelExplainer(
                self._model.predict if hasattr(self._model, "predict") else self._model,
                background_sample.values,
            )
            self._explainer_type = "KernelExplainer"
        return self._explainer_type

    def explain(self, X: pd.DataFrame) -> tuple[np.ndarray, str]:
        """Return (shap_values, explainer_type) for given input."""
        if self._explainer is None:
            self.fit()
        assert self._explainer is not None
        shap_values = self._explainer.shap_values(X.values)
        return np.array(shap_values), self._explainer_type

    def top_features(
        self, shap_values: np.ndarray, feature_names: list[str], n: int = 5
    ) -> list[dict]:
        """Return top-n features by absolute SHAP value."""
        if shap_values.ndim > 1:
            abs_shap = np.abs(shap_values).mean(axis=0)
        else:
            abs_shap = np.abs(shap_values)
        indices = np.argsort(abs_shap)[::-1][:n]
        return [
            {"feature": feature_names[i], "shap_value": float(shap_values.mean(axis=0)[i] if shap_values.ndim > 1 else shap_values[i])}
            for i in indices
        ]
