from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src.features.schema import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

MODEL_NAME = "patchtst"
_TARGET_ENCODING = {"SELL": 0, "HOLD": 1, "BUY": 2}


class PatchTSTWrapper:
    """PatchTST-shaped wrapper backed by sklearn GradientBoostingClassifier.

    The Spec-04 architecture is neuralforecast.PatchTST; deferred for the same
    reason as N-HiTS.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._model: GradientBoostingClassifier | None = None
        self._features: list[str] = []

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, class_weights: dict) -> None:
        features = [c for c in FEATURE_COLUMNS if c in X_train.columns]
        mask = y_train.notna()
        X = X_train.loc[mask, features].astype(float).fillna(0.0)
        y = y_train.loc[mask].map(_TARGET_ENCODING).astype(int)
        if X.empty:
            raise ValueError("patchtst: no training rows after dropping null targets")
        sample_weight = compute_sample_weight(class_weight="balanced", y=y) if class_weights else None
        self._model = GradientBoostingClassifier(
            n_estimators=self._config.get("n_estimators", 100),
            max_depth=self._config.get("max_depth", 3),
            learning_rate=self._config.get("learning_rate", 0.1),
            random_state=self._config.get("random_seed", 42),
        )
        self._model.fit(X, y, sample_weight=sample_weight)
        self._features = features
        logger.info("patchtst training complete (%d rows, %d features)", len(X), len(features))

    def predict(self, X_val: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("patchtst: model not trained")
        X = X_val[self._features].astype(float).fillna(0.0)
        return self._model.predict(X).astype(int)

    def predict_proba(self, X_val: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("patchtst: model not trained")
        X = X_val[self._features].astype(float).fillna(0.0)
        return self._model.predict_proba(X)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with (path / "model.pkl").open("wb") as f:
            pickle.dump({"model": self._model, "features": self._features, "config": self._config}, f)

    @classmethod
    def load(cls, path: Path, config: dict) -> "PatchTSTWrapper":
        wrapper = cls(config)
        with (path / "model.pkl").open("rb") as f:
            data = pickle.load(f)
        wrapper._model = data["model"]
        wrapper._features = data["features"]
        return wrapper
