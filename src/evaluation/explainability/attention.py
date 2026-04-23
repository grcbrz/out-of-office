from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_TRANSFORMER_MODELS = {"patchtst", "autoformer"}


class AttentionExtractor:
    """Extracts attention weights from transformer encoder final layer.

    Returns None for N-HiTS (no attention mechanism).
    """

    def __init__(self, model: Any, model_name: str) -> None:
        self._model = model
        self._model_name = model_name

    def extract(self, X: Any) -> list[float] | None:
        """Return averaged attention weights or None for non-transformer models."""
        if self._model_name not in _TRANSFORMER_MODELS:
            logger.debug("attention weights not applicable for %s", self._model_name)
            return None

        try:
            weights = self._model.get_attention_weights(X)
            # Average across heads: shape (n_heads, seq, seq) → (seq, seq) → last query → (seq,)
            avg = np.mean(weights, axis=0)[-1]
            return avg.tolist()
        except Exception as exc:
            logger.warning("attention extraction failed for %s: %s", self._model_name, exc)
            return None
