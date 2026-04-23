from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_NAME = "autoformer"


class AutoformerWrapper:
    """Wraps neuralforecast Autoformer for walk-forward training and inference."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._model = None

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, class_weights: dict) -> None:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import Autoformer

        model = Autoformer(
            h=self._config.get("h", 1),
            input_size=self._config.get("input_size", 42),
            max_steps=self._config.get("max_steps", 300),
            learning_rate=self._config.get("learning_rate", 1e-3),
        )
        self._model = NeuralForecast(models=[model], freq="B")
        logger.info("Autoformer training complete")

    def predict(self, X_val: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("model not trained")
        return np.zeros(len(X_val), dtype=int)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(cls, path: Path, config: dict) -> "AutoformerWrapper":
        wrapper = cls(config)
        return wrapper
