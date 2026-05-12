"""LightGBM multi-class classifier — v1 production candidate.

Replaces the placeholder N-HiTS / Autoformer wrappers (sklearn ExtraTrees /
MLPClassifier under transformer-flavoured names). Picked over real
neuralforecast Autoformer/N-HiTS for two reasons:

  1. Daily 3-class equity classification with ~12k training rows is too small
     for transformers to outperform gradient-boosted trees.
  2. Trees handle mixed-scale features and integer-encoded calendars without
     tedious pre-processing, and SHAP values via TreeExplainer are exact and
     fast (no sampling).

The harness still supports adding a real neuralforecast model later by
implementing another ``BaseModelWrapper`` subclass and adding it to
``training_pipeline._CANDIDATE_NAMES`` / ``_instantiate_wrapper``.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.models.architectures.base import BaseModelWrapper

logger = logging.getLogger(__name__)

MODEL_NAME = "lightgbm"
_DEFAULT_PARAMS: dict[str, Any] = {
    # multi-class classification with 3 classes (SELL/HOLD/BUY)
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    # capacity / regularisation — modest defaults; tune via configs/models/lightgbm.yaml
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.0,
    "reg_lambda": 0.1,
    # noise control
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}


class LightGBMWrapper(BaseModelWrapper):
    """Gradient-boosted multi-class classifier."""

    name = MODEL_NAME

    def _build_model(self, class_weights: dict) -> Any:
        from lightgbm import LGBMClassifier

        params: dict[str, Any] = {**_DEFAULT_PARAMS}
        params.update({k: v for k, v in self._config.items() if k != "random_seed"})
        # Standardise on `random_state`; allow legacy YAML key `random_seed`.
        if "random_seed" in self._config:
            params["random_state"] = self._config["random_seed"]
        # Don't let the YAML override num_class — the pipeline always emits 3 classes.
        params["num_class"] = 3
        return LGBMClassifier(**params)

    def _fit(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        class_weights: dict,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        class_sw = self._sample_weights(y, class_weights)
        if class_sw is not None and sample_weight is not None:
            combined = class_sw * sample_weight
        else:
            combined = class_sw if class_sw is not None else sample_weight
        model.fit(X, y, sample_weight=combined)

    @staticmethod
    def _sample_weights(y: pd.Series, class_weights: dict) -> np.ndarray | None:
        """Map per-class weights → per-sample weights aligned to y.

        ``class_weights`` keys are integer class IDs (0=SELL, 1=HOLD, 2=BUY) as
        produced by ``DataPreparer.compute_class_weights``. Returns ``None`` if
        the dict is empty so LightGBM uses unweighted training.
        """
        if not class_weights:
            return None
        # Coerce keys to int — they may arrive as strings from JSON round-trips.
        weights = {int(k): float(v) for k, v in class_weights.items()}
        return y.map(weights).fillna(1.0).to_numpy()
