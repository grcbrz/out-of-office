"""Naive last-direction baseline.

Spec 04 §6 mandates benchmarking against at least one naive baseline. This
wrapper is **not a candidate for production** — it exists so every fold has a
floor metric the real models must beat.

Rule (mirrors target labelling, but lagged):

    if log_return_lag1 > p70_train  →  BUY
    if log_return_lag1 < p30_train  →  SELL
    else                            →  HOLD

``p30_train`` and ``p70_train`` are the 30th / 70th percentiles of
``log_return_lag1`` measured on the training fold only. The baseline holds no
sklearn estimator — fit is a no-op storing the two thresholds.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.architectures.base import BaseModelWrapper, TARGET_ENCODING

logger = logging.getLogger(__name__)

MODEL_NAME = "baseline_last_direction"
_REQUIRED_COLUMN = "log_return_lag1"
_BUY_PERCENTILE = 70
_SELL_PERCENTILE = 30


class BaselineLastDirectionWrapper(BaseModelWrapper):
    """Last-direction baseline. Fits two scalar thresholds on the training fold."""

    name = MODEL_NAME

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config or {})
        self._p30: float | None = None
        self._p70: float | None = None

    # ------------------------------------------------------------------
    # Override BaseModelWrapper because we hold no sklearn estimator.
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        class_weights: dict,
    ) -> None:
        if _REQUIRED_COLUMN not in X_train.columns:
            raise ValueError(
                f"{self.name}: required column '{_REQUIRED_COLUMN}' missing"
            )
        series = X_train[_REQUIRED_COLUMN].dropna()
        if series.empty:
            raise ValueError(
                f"{self.name}: '{_REQUIRED_COLUMN}' contains only nulls in training fold"
            )
        self._p30 = float(np.percentile(series, _SELL_PERCENTILE))
        self._p70 = float(np.percentile(series, _BUY_PERCENTILE))
        # Persist a feature list of length 1 so save/load and SHAP wiring are uniform.
        self._features = [_REQUIRED_COLUMN]
        logger.info(
            "%s training complete: p30=%.5f p70=%.5f (n=%d)",
            self.name, self._p30, self._p70, len(series),
        )

    def predict(self, X_val: pd.DataFrame) -> np.ndarray:
        self._require_fitted()
        lag = X_val[_REQUIRED_COLUMN].astype(float).fillna(0.0)
        out = np.full(len(lag), TARGET_ENCODING["HOLD"], dtype=int)
        out[lag.values > self._p70] = TARGET_ENCODING["BUY"]
        out[lag.values < self._p30] = TARGET_ENCODING["SELL"]
        return out

    def predict_proba(self, X_val: pd.DataFrame) -> np.ndarray:
        """Hard-decision baseline → degenerate one-hot probabilities.

        Used for ROC-AUC and downstream uncertainty plumbing. Not meaningful
        as a confidence estimate.
        """
        preds = self.predict(X_val)
        proba = np.zeros((len(preds), 3), dtype=float)
        for row_idx, class_idx in enumerate(preds):
            proba[row_idx, class_idx] = 1.0
        return proba

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with (path / "model.pkl").open("wb") as f:
            pickle.dump(
                {
                    "p30": self._p30,
                    "p70": self._p70,
                    "features": self._features,
                    "config": self._config,
                },
                f,
            )

    @classmethod
    def load(
        cls,
        path: Path,
        config: dict | None = None,
    ) -> "BaselineLastDirectionWrapper":
        wrapper = cls(config or {})
        with (path / "model.pkl").open("rb") as f:
            data = pickle.load(f)
        wrapper._p30 = data["p30"]
        wrapper._p70 = data["p70"]
        wrapper._features = data["features"]
        return wrapper

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_fitted(self) -> None:
        if self._p30 is None or self._p70 is None:
            raise RuntimeError(f"{self.name}: model not trained")
