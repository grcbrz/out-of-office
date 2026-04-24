from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from src.features.schema import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

MODEL_NAME = "autoformer"
_TARGET_ENCODING = {"SELL": 0, "HOLD": 1, "BUY": 2}


class AutoformerWrapper:
    """Autoformer-shaped wrapper backed by sklearn ExtraTreesClassifier.

    The Spec-04 architecture is neuralforecast.Autoformer; deferred for the same
    reason as N-HiTS.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._model: ExtraTreesClassifier | None = None
        self._features: list[str] = []

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, class_weights: dict) -> None:
        features = [c for c in FEATURE_COLUMNS if c in X_train.columns]
        mask = y_train.notna()
        X = X_train.loc[mask, features].astype(float).fillna(0.0)
        y = y_train.loc[mask].map(_TARGET_ENCODING).astype(int)
        if X.empty:
            raise ValueError("autoformer: no training rows after dropping null targets")
        self._model = ExtraTreesClassifier(
            n_estimators=self._config.get("n_estimators", 200),
            max_depth=self._config.get("max_depth", None),
            class_weight="balanced" if class_weights else None,
            n_jobs=self._config.get("n_jobs", 1),
            random_state=self._config.get("random_seed", 42),
        )
        self._model.fit(X, y)
        self._features = features
        logger.info("autoformer training complete (%d rows, %d features)", len(X), len(features))

    def predict(self, X_val: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("autoformer: model not trained")
        X = X_val[self._features].astype(float).fillna(0.0)
        return self._model.predict(X).astype(int)

    def predict_proba(self, X_val: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("autoformer: model not trained")
        X = X_val[self._features].astype(float).fillna(0.0)
        return self._model.predict_proba(X)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with (path / "model.pkl").open("wb") as f:
            pickle.dump({"model": self._model, "features": self._features, "config": self._config}, f)

    @classmethod
    def load(cls, path: Path, config: dict) -> "AutoformerWrapper":
        wrapper = cls(config)
        with (path / "model.pkl").open("rb") as f:
            data = pickle.load(f)
        wrapper._model = data["model"]
        wrapper._features = data["features"]
        return wrapper
