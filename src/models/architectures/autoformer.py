from __future__ import annotations

from typing import Any

from sklearn.ensemble import ExtraTreesClassifier

from src.models.architectures.base import BaseModelWrapper

MODEL_NAME = "autoformer"


class AutoformerWrapper(BaseModelWrapper):
    """Autoformer-shaped wrapper backed by sklearn ExtraTreesClassifier.

    The Spec-04 architecture is neuralforecast.Autoformer; deferred for the same
    reason as N-HiTS.
    """

    name = MODEL_NAME

    def _build_model(self, class_weights: dict) -> Any:
        return ExtraTreesClassifier(
            n_estimators=self._config.get("n_estimators", 200),
            max_depth=self._config.get("max_depth", None),
            class_weight="balanced" if class_weights else None,
            n_jobs=self._config.get("n_jobs", 1),
            random_state=self._config.get("random_seed", 42),
        )
