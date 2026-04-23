from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class EvaluationQualityGateError(Exception):
    """Raised when production fold metrics fail to meet minimum thresholds."""


class QualityGate:
    """Checks production fold metrics against configurable thresholds.

    All thresholds are loaded from configs/evaluation.yaml — never hardcoded.
    """

    def __init__(
        self,
        f1_macro_min: float = 0.35,
        mcc_min: float = 0.05,
        hit_rate_min: float = 0.50,
        max_signal_concentration: float = 0.80,
    ) -> None:
        self._f1_macro_min = f1_macro_min
        self._mcc_min = mcc_min
        self._hit_rate_min = hit_rate_min
        self._max_signal_concentration = max_signal_concentration

    def check(self, metrics: dict) -> None:
        """Raise EvaluationQualityGateError if any gate fails."""
        failures: list[str] = []

        f1 = metrics.get("f1_macro", 0.0)
        if f1 < self._f1_macro_min:
            failures.append(f"f1_macro={f1:.3f} < {self._f1_macro_min}")

        mcc = metrics.get("mcc", 0.0)
        if mcc < self._mcc_min:
            failures.append(f"mcc={mcc:.3f} < {self._mcc_min}")

        hit_rate = metrics.get("hit_rate")
        if hit_rate is not None and hit_rate < self._hit_rate_min:
            failures.append(f"hit_rate={hit_rate:.3f} < {self._hit_rate_min}")

        signal_dist = metrics.get("signal_distribution", {})
        for label, rate in signal_dist.items():
            if rate > self._max_signal_concentration:
                failures.append(
                    f"signal concentration {label}={rate:.2f} > {self._max_signal_concentration}"
                )

        if failures:
            msg = "Quality gate failed: " + "; ".join(failures)
            logger.error(msg)
            raise EvaluationQualityGateError(msg)

        logger.info("Quality gate passed: f1=%.3f, mcc=%.3f, hit_rate=%s", f1, mcc, hit_rate)
