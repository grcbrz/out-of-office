from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

from src.features.schema import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

MODEL_NAME = "nhits"
_TARGET_ENCODING = {"SELL": 0, "HOLD": 1, "BUY": 2}


class NHiTSWrapper:
    """N-HiTS-shaped wrapper backed by sklearn MLPClassifier.

    The Spec-04 architecture is neuralforecast.NHITS; that dependency is deferred
    until torch + neuralforecast are available. The wrapper interface
    (train / predict / save / load) is preserved so the substitution is contained.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._model: MLPClassifier | None = None
        self._features: list[str] = []

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, class_weights: dict) -> None:
        features = [c for c in FEATURE_COLUMNS if c in X_train.columns]
        mask = y_train.notna()
        X = X_train.loc[mask, features].astype(float).fillna(0.0)
        y = y_train.loc[mask].map(_TARGET_ENCODING).astype(int)
        if X.empty:
            raise ValueError("nhits: no training rows after dropping null targets")
        self._model = MLPClassifier(
            hidden_layer_sizes=tuple(self._config.get("hidden_layer_sizes", (128, 64))),
            max_iter=self._config.get("max_iter", 200),
            learning_rate_init=self._config.get("learning_rate", 1e-3),
            random_state=self._config.get("random_seed", 42),
        )
        self._model.fit(X, y)
        self._features = features
        logger.info("nhits training complete (%d rows, %d features)", len(X), len(features))

    def predict(self, X_val: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("nhits: model not trained")
        X = X_val[self._features].astype(float).fillna(0.0)
        return self._model.predict(X).astype(int)

    def predict_proba(self, X_val: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("nhits: model not trained")
        X = X_val[self._features].astype(float).fillna(0.0)
        return self._model.predict_proba(X)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with (path / "model.pkl").open("wb") as f:
            pickle.dump({"model": self._model, "features": self._features, "config": self._config}, f)

    @classmethod
    def load(cls, path: Path, config: dict) -> "NHiTSWrapper":
        wrapper = cls(config)
        with (path / "model.pkl").open("rb") as f:
            data = pickle.load(f)
        wrapper._model = data["model"]
        wrapper._features = data["features"]
        return wrapper
