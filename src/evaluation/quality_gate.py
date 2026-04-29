from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class EvaluationQualityGateError(Exception):
    """Raised when production fold metrics fail to meet minimum thresholds."""


class QualityGate:
    """Gate production fold metrics against absolute floors AND a baseline delta.

    Two classes of check:
      * **Absolute floors** — f1, MCC, hit rate, signal concentration. These guard
        against regressions that fall below sane minimums regardless of how the
        baseline performed.
      * **Baseline delta** — the production model must beat the naive baseline by
        at least ``min_delta_over_baseline`` F1 points. This is the gate that
        actually answers "did the model add anything over the trivial benchmark".
        If ``baseline_f1_macro`` is missing from the metrics dict the delta check
        is skipped (with a warning) — this should only happen for legacy artifacts
        produced before the baseline was wired in.

    All thresholds are loaded from configs/evaluation.yaml — never hardcoded.
    """

    def __init__(
        self,
        f1_macro_min: float = 0.35,
        mcc_min: float = 0.05,
        hit_rate_min: float = 0.50,
        max_signal_concentration: float = 0.80,
        min_delta_over_baseline: float = 0.02,
    ) -> None:
        self._f1_macro_min = f1_macro_min
        self._mcc_min = mcc_min
        self._hit_rate_min = hit_rate_min
        self._max_signal_concentration = max_signal_concentration
        self._min_delta_over_baseline = min_delta_over_baseline

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
                    f"signal concentration {label}={rate:.2f} > "
                    f"{self._max_signal_concentration}"
                )

        baseline_f1 = metrics.get("baseline_f1_macro")
        if baseline_f1 is None:
            logger.warning(
                "baseline_f1_macro absent from metrics — skipping baseline delta gate"
            )
        else:
            delta = f1 - baseline_f1
            if delta < self._min_delta_over_baseline:
                failures.append(
                    f"f1_macro - baseline_f1_macro = {delta:+.3f} < "
                    f"{self._min_delta_over_baseline} "
                    f"(model={f1:.3f}, baseline={baseline_f1:.3f})"
                )

        if failures:
            msg = "Quality gate failed: " + "; ".join(failures)
            logger.error(msg)
            raise EvaluationQualityGateError(msg)

        logger.info(
            "Quality gate passed: f1=%.3f, mcc=%.3f, hit_rate=%s, baseline_f1=%s",
            f1, mcc, hit_rate,
            f"{baseline_f1:.3f}" if baseline_f1 is not None else "n/a",
        )
