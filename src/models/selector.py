from __future__ import annotations

from dataclasses import dataclass

# Tie-breaking preference: prefer simpler architecture
_PREFERENCE_ORDER = ["nhits", "patchtst", "autoformer"]


@dataclass
class ModelResult:
    model_name: str
    f1_macro: float
    artifact_path: str | None = None


def select_winner(results: list[ModelResult]) -> ModelResult:
    """Return the ModelResult with the highest F1-macro.

    Ties broken by preference: N-HiTS > PatchTST > Autoformer.
    """
    if not results:
        raise ValueError("no model results to select from")

    def _rank(r: ModelResult) -> tuple[float, int]:
        preference = _PREFERENCE_ORDER.index(r.model_name) if r.model_name in _PREFERENCE_ORDER else 99
        return (r.f1_macro, -preference)

    return max(results, key=_rank)
