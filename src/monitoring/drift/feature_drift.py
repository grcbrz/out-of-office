from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class FeatureDriftResult:
    feature: str
    ks_pvalue: float
    psi: float
    severity: str  # "none" | "warning" | "significant"
    triggered: bool


def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two continuous distributions."""
    combined = np.concatenate([reference, current])
    bin_edges = np.histogram_bin_edges(combined, bins=bins)

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_pct = (ref_counts + 1e-6) / (len(reference) + 1e-6 * bins)
    cur_pct = (cur_counts + 1e-6) / (len(current) + 1e-6 * bins)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def detect_feature_drift(
    feature: str,
    reference: np.ndarray,
    current: np.ndarray,
    ks_pvalue_threshold: float,
    psi_warning_threshold: float,
    psi_alert_threshold: float,
) -> FeatureDriftResult:
    ks_stat, ks_pvalue = stats.ks_2samp(reference, current)
    psi = compute_psi(reference, current)

    if psi >= psi_alert_threshold:
        severity = "significant"
    elif psi >= psi_warning_threshold:
        severity = "warning"
    else:
        severity = "none"

    triggered = ks_pvalue < ks_pvalue_threshold and psi >= psi_alert_threshold

    return FeatureDriftResult(
        feature=feature,
        ks_pvalue=float(ks_pvalue),
        psi=float(psi),
        severity=severity,
        triggered=bool(triggered),
    )


class FeatureDriftDetector:
    def __init__(
        self,
        ks_pvalue_threshold: float = 0.05,
        psi_warning_threshold: float = 0.10,
        psi_alert_threshold: float = 0.20,
    ) -> None:
        self._ks_threshold = ks_pvalue_threshold
        self._psi_warning = psi_warning_threshold
        self._psi_alert = psi_alert_threshold

    def run(
        self,
        reference_stats: dict[str, np.ndarray],
        current_stats: dict[str, np.ndarray],
    ) -> list[FeatureDriftResult]:
        results = []
        for feature, ref_values in reference_stats.items():
            if feature not in current_stats:
                continue
            cur_values = current_stats[feature]
            result = detect_feature_drift(
                feature=feature,
                reference=ref_values,
                current=cur_values,
                ks_pvalue_threshold=self._ks_threshold,
                psi_warning_threshold=self._psi_warning,
                psi_alert_threshold=self._psi_alert,
            )
            results.append(result)
        return results
