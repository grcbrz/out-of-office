from __future__ import annotations

from typing import Any

from sklearn.neural_network import MLPClassifier

from src.models.architectures.base import BaseModelWrapper

MODEL_NAME = "nhits"


class NHiTSWrapper(BaseModelWrapper):
    """N-HiTS-shaped wrapper backed by sklearn MLPClassifier.

    The Spec-04 architecture is neuralforecast.NHITS; that dependency is deferred
    until torch + neuralforecast are available. The wrapper interface
    (train / predict / predict_proba / save / load) is stable.
    """

    name = MODEL_NAME

    def _build_model(self, class_weights: dict) -> Any:
        return MLPClassifier(
            hidden_layer_sizes=tuple(self._config.get("hidden_layer_sizes", (128, 64))),
            max_iter=self._config.get("max_iter", 200),
            learning_rate_init=self._config.get("learning_rate", 1e-3),
            random_state=self._config.get("random_seed", 42),
        )
