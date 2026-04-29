"""Attention-weight extractor — currently a no-op.

v1 production candidate is LightGBM (no attention mechanism). This module is
kept so callers don't break, and so wiring is in place for a future
neuralforecast (Autoformer / PatchTST) candidate. While no transformer model
is in production, ``extract`` always returns ``None``.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# When a real transformer wrapper ships, add its model_name string here.
_TRANSFORMER_MODELS: frozenset[str] = frozenset()


class AttentionExtractor:
    """No-op until a transformer architecture re-enters the codebase."""

    def __init__(self, model: Any, model_name: str) -> None:
        self._model = model
        self._model_name = model_name

    def extract(self, X: Any) -> list[float] | None:
        if self._model_name not in _TRANSFORMER_MODELS:
            logger.debug(
                "attention extraction skipped: %s has no attention mechanism",
                self._model_name,
            )
            return None

        # Reachable only if a transformer wrapper is registered above.
        try:
            import numpy as np
            weights = self._model.get_attention_weights(X)
            avg = np.mean(weights, axis=0)[-1]
            return avg.tolist()
        except Exception as exc:  # pragma: no cover — guarded by the gate above
            logger.warning(
                "attention extraction failed for %s: %s", self._model_name, exc,
            )
            return None
