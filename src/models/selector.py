from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Tie-breaking preference: simpler / cheaper architectures first. v1 ships with
# only LightGBM but the slot is preserved so adding a real neuralforecast
# candidate later doesn't mean rewiring the selector.
_PREFERENCE_ORDER = ["lightgbm"]
# Names that must never be selected as production winners — they exist only as
# benchmarks for the quality gate.
_BASELINE_NAMES = frozenset({"baseline_last_direction"})


@dataclass
class ModelResult:
    model_name: str
    f1_macro: float
    artifact_path: str | None = None
    predictions: Any = None  # int array aligned with labelled rows of val_df
    predictions_proba: np.ndarray | None = None  # (n, 3) probabilities, columns ordered SELL/HOLD/BUY
    proba_classes: tuple[int, ...] | None = None  # underlying estimator's class order
    is_baseline: bool = False
    confidence_threshold: float | None = None  # τ chosen on this fold's val set; None until calibrated


def select_winner(results: list[ModelResult]) -> ModelResult:
    """Highest F1-macro among non-baseline candidates. Tie-break by preference."""
    candidates = [r for r in results if not r.is_baseline and r.model_name not in _BASELINE_NAMES]
    if not candidates:
        raise ValueError("no candidate model results to select from (baselines excluded)")

    def _rank(r: ModelResult) -> tuple[float, int]:
        preference = (
            _PREFERENCE_ORDER.index(r.model_name)
            if r.model_name in _PREFERENCE_ORDER
            else 99
        )
        return (r.f1_macro, -preference)

    return max(candidates, key=_rank)
