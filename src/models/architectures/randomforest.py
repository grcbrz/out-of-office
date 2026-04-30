"""RandomForest multi-class classifier — diversity candidate alongside LightGBM.

Bagged-tree counterpart to the gradient-boosted LightGBM. RandomForest wins
on folds where boosting overfits sequential noise; loses on folds where the
global signal is steep and residual fitting helps. Including both candidates
buys uncorrelated failure modes, which is the real diversification benefit.

Class weights
-------------
RandomForest takes ``class_weight={class_id: weight}`` at *construction* time.
The internal bootstrap then reweights draws correctly. Passing the weights
as ``sample_weight`` at fit time would interact badly with the bootstrap and
double-up the weighting on minority classes — a subtle, well-known footgun.
This is the primary contract difference from ``LightGBMWrapper``, which uses
``sample_weight``.
"""
from __future__ import annotations

import logging
from typing import Any

from src.models.architectures.base import BaseModelWrapper

logger = logging.getLogger(__name__)

MODEL_NAME = "randomforest"

# Defaults tuned for ~12k training rows × ~30 stationary features.
# Overridable via configs/models/randomforest.yaml; pipeline-invariant
# settings (n_jobs=1, random_state=42) are always honoured by the YAML.
_DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": None,
    "min_samples_leaf": 5,
    "min_samples_split": 10,
    "max_features": "sqrt",
    "bootstrap": True,
    "n_jobs": 1,         # determinism — matches the LightGBM thread pinning
    "random_state": 42,
}


class RandomForestWrapper(BaseModelWrapper):
    """sklearn RandomForestClassifier with the project's BaseModelWrapper contract."""

    name = MODEL_NAME

    def _build_model(self, class_weights: dict) -> Any:
        from sklearn.ensemble import RandomForestClassifier

        params: dict[str, Any] = {**_DEFAULT_PARAMS}
        # Apply YAML overrides; ``random_seed`` is the project-wide config key,
        # mapped to sklearn's ``random_state``.
        params.update({k: v for k, v in self._config.items() if k != "random_seed"})
        if "random_seed" in self._config:
            params["random_state"] = self._config["random_seed"]

        # Coerce keys → int (JSON round-trips can stringify them) and pass at
        # construction. Empty dict → don't set the kwarg (sklearn default = None).
        if class_weights:
            params["class_weight"] = {
                int(k): float(v) for k, v in class_weights.items()
            }

        return RandomForestClassifier(**params)

    # No _fit override needed — the base class default ``model.fit(X, y)`` is
    # correct because class_weight is applied at construction, not at fit time.
