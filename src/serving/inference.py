from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.features.schema import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

_SIGNAL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}
_HOLD_INT = 1


class InferenceEngine:
    """Runs the model forward pass and decodes probabilities to BUY/HOLD/SELL.

    Confidence gating: when ``confidence_threshold`` is provided, any
    prediction whose top-class probability is at or below ``τ`` is demoted to
    HOLD. This mirrors the calibration done at training time (Spec 04 §7.4)
    and is what production trades on.
    """

    def __init__(
        self,
        model: Any,
        imputation_params: dict[str, float],
        ticker_map: dict[str, int],
        trained_features: list[str] | None = None,
        confidence_threshold: float | None = None,
    ) -> None:
        self._model = model
        self._imputation_params = imputation_params
        self._ticker_map = ticker_map
        # If not provided, infer from FEATURE_COLUMNS; assume all that fit in n_features_in_
        self._trained_features = trained_features or FEATURE_COLUMNS
        self._tau = confidence_threshold

    def predict(self, ticker: str, feature_row: pd.Series) -> tuple[str, float]:
        """Return (signal, confidence) for a single ticker's feature row.

        ``confidence`` is always the unmodified top-class probability — it
        reflects what the model believed even when τ demoted the signal to
        HOLD. Downstream consumers (logging, monitoring) need the original
        confidence to diagnose calibration drift.
        """
        ticker_id = self._ticker_map.get(ticker, -1)
        feature_row = feature_row.copy()
        feature_row["ticker_id"] = ticker_id

        X = self._prepare(feature_row)
        probabilities = self._forward(X)
        class_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[class_idx])

        if self._tau is not None and confidence <= self._tau:
            class_idx = _HOLD_INT

        return _SIGNAL_MAP[class_idx], confidence

    def _prepare(self, row: pd.Series) -> pd.DataFrame:
        cols = [c for c in self._trained_features if c in row.index]
        return row[cols].fillna(0.0).astype(float).to_frame().T.reset_index(drop=True)

    def _forward(self, X: pd.DataFrame) -> np.ndarray:
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)[0]
        n_classes = 3
        return np.ones(n_classes) / n_classes
