from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class PredictionDriftResult:
    triggered: bool
    chi2_pvalue: float
    current_distribution: dict[str, float]
    degenerate_signal: bool
    dominant_class: str | None


def detect_prediction_drift(
    reference_counts: dict[str, int],
    current_counts: dict[str, int],
    chi2_pvalue_threshold: float,
    max_signal_concentration: float,
) -> PredictionDriftResult:
    labels = ["BUY", "HOLD", "SELL"]
    ref_arr = np.array([reference_counts.get(l, 0) for l in labels], dtype=float)
    cur_arr = np.array([current_counts.get(l, 0) for l in labels], dtype=float)

    total_current = cur_arr.sum()
    if total_current == 0:
        return PredictionDriftResult(
            triggered=False,
            chi2_pvalue=1.0,
            current_distribution={l: 0.0 for l in labels},
            degenerate_signal=False,
            dominant_class=None,
        )

    current_distribution = {l: float(cur_arr[i] / total_current) for i, l in enumerate(labels)}

    # Normalise reference to expected counts matching current total
    total_ref = ref_arr.sum()
    if total_ref == 0:
        expected = np.ones(3) * (total_current / 3)
    else:
        expected = ref_arr / total_ref * total_current

    # Avoid chi-squared with zero expected cells
    mask = expected > 0
    if mask.sum() < 2:
        chi2_pvalue = 1.0
    else:
        _, chi2_pvalue = stats.chisquare(f_obs=cur_arr[mask], f_exp=expected[mask])

    triggered = chi2_pvalue < chi2_pvalue_threshold

    degenerate_signal = any(v >= max_signal_concentration for v in current_distribution.values())
    dominant_class = max(current_distribution, key=current_distribution.__getitem__) if degenerate_signal else None

    return PredictionDriftResult(
        triggered=triggered,
        chi2_pvalue=float(chi2_pvalue),
        current_distribution=current_distribution,
        degenerate_signal=degenerate_signal,
        dominant_class=dominant_class,
    )


class PredictionDriftDetector:
    def __init__(
        self,
        chi2_pvalue_threshold: float = 0.05,
        max_signal_concentration: float = 0.80,
    ) -> None:
        self._chi2_threshold = chi2_pvalue_threshold
        self._concentration = max_signal_concentration

    def run(
        self,
        reference_counts: dict[str, int],
        current_counts: dict[str, int],
    ) -> PredictionDriftResult:
        return detect_prediction_drift(
            reference_counts=reference_counts,
            current_counts=current_counts,
            chi2_pvalue_threshold=self._chi2_threshold,
            max_signal_concentration=self._concentration,
        )
