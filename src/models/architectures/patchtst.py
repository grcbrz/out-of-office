from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src.models.architectures.base import BaseModelWrapper

MODEL_NAME = "patchtst"


class PatchTSTWrapper(BaseModelWrapper):
    """PatchTST-shaped wrapper backed by sklearn GradientBoostingClassifier.

    The Spec-04 architecture is neuralforecast.PatchTST; deferred for the same
    reason as N-HiTS.
    """

    name = MODEL_NAME

    def _build_model(self, class_weights: dict) -> Any:
        return GradientBoostingClassifier(
            n_estimators=self._config.get("n_estimators", 100),
            max_depth=self._config.get("max_depth", 3),
            learning_rate=self._config.get("learning_rate", 0.1),
            random_state=self._config.get("random_seed", 42),
        )

    def _fit(self, model: Any, X: pd.DataFrame, y: pd.Series, class_weights: dict) -> None:
        sample_weight = compute_sample_weight(class_weight="balanced", y=y) if class_weights else None
        model.fit(X, y, sample_weight=sample_weight)
