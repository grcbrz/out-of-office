from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Tie-breaking preference: prefer simpler architecture
_PREFERENCE_ORDER = ["nhits", "autoformer"]


@dataclass
class ModelResult:
    model_name: str
    f1_macro: float
    artifact_path: str | None = None
    predictions: Any = None  # numpy int array aligned with the labelled rows of val_df


def select_winner(results: list[ModelResult]) -> ModelResult:
    """Return the ModelResult with the highest F1-macro.

    Ties broken by preference: N-HiTS > Autoformer.
    """
    if not results:
        raise ValueError("no model results to select from")

    def _rank(r: ModelResult) -> tuple[float, int]:
        preference = _PREFERENCE_ORDER.index(r.model_name) if r.model_name in _PREFERENCE_ORDER else 99
        return (r.f1_macro, -preference)

    return max(results, key=_rank)
