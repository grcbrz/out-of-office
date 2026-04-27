from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.schema import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

TARGET_ENCODING: dict[str, int] = {"SELL": 0, "HOLD": 1, "BUY": 2}


class BaseModelWrapper:
    """Shared scaffold for sklearn-backed model wrappers.

    Subclasses implement _build_model() to return a configured sklearn estimator.
    Optionally override _fit() when the estimator needs non-standard fit arguments
    (e.g. sample_weight).
    """

    name: str = ""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._model: Any = None
        self._features: list[str] = []

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    def _build_model(self, class_weights: dict) -> Any:
        raise NotImplementedError

    def _fit(self, model: Any, X: pd.DataFrame, y: pd.Series, class_weights: dict) -> None:
        model.fit(X, y)

    # ------------------------------------------------------------------
    # Shared implementation
    # ------------------------------------------------------------------

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, class_weights: dict) -> None:
        features = [c for c in FEATURE_COLUMNS if c in X_train.columns]
        mask = y_train.notna()
        X = X_train.loc[mask, features].astype(float).fillna(0.0)
        y = y_train.loc[mask].map(TARGET_ENCODING).astype(int)
        if X.empty:
            raise ValueError(f"{self.name}: no training rows after dropping null targets")
        model = self._build_model(class_weights)
        self._fit(model, X, y, class_weights)
        self._model = model
        self._features = features
        logger.info("%s training complete (%d rows, %d features)", self.name, len(X), len(features))

    def predict(self, X_val: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError(f"{self.name}: model not trained")
        X = X_val[self._features].astype(float).fillna(0.0)
        return self._model.predict(X).astype(int)

    def predict_proba(self, X_val: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError(f"{self.name}: model not trained")
        X = X_val[self._features].astype(float).fillna(0.0)
        return self._model.predict_proba(X)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with (path / "model.pkl").open("wb") as f:
            pickle.dump({"model": self._model, "features": self._features, "config": self._config}, f)

    @classmethod
    def load(cls, path: Path, config: dict) -> "BaseModelWrapper":
        wrapper = cls(config)
        with (path / "model.pkl").open("rb") as f:
            data = pickle.load(f)
        wrapper._model = data["model"]
        wrapper._features = data["features"]
        return wrapper
