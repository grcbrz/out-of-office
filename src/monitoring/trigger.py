from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path

from src.monitoring.alerts import AlertWriter, MonitoringAlert
from src.monitoring.degradation import DegradationResult
from src.monitoring.drift.feature_drift import FeatureDriftResult
from src.monitoring.drift.prediction_drift import PredictionDriftResult
from src.monitoring.persistence import update_status

logger = logging.getLogger(__name__)


@dataclass
class TriggerDecision:
    retraining_required: bool
    trigger_reason: str | None


class RetrainingTrigger:
    def __init__(
        self,
        status_path: str | Path,
        alert_writer: AlertWriter,
    ) -> None:
        self._status_path = Path(status_path)
        self._alert_writer = alert_writer

    def evaluate(
        self,
        run_date: dt.date,
        feature_drift_results: list[FeatureDriftResult],
        prediction_drift_result: PredictionDriftResult,
        degradation_result: DegradationResult,
    ) -> TriggerDecision:
        feature_drift_triggered = any(r.triggered for r in feature_drift_results)
        pred_drift_triggered = prediction_drift_result.triggered or prediction_drift_result.degenerate_signal
        degradation_triggered = degradation_result.triggered

        retraining_required = feature_drift_triggered or pred_drift_triggered or degradation_triggered

        reasons = []
        if feature_drift_triggered:
            reasons.append("feature_drift")
        if pred_drift_triggered:
            reasons.append("prediction_drift")
        if degradation_triggered:
            reasons.append("performance_degradation")

        trigger_reason = ",".join(reasons) if reasons else None
        date_str = str(run_date)

        if retraining_required:
            alert = MonitoringAlert(
                alert_date=date_str,
                retraining_triggered=True,
                feature_drift={
                    "triggered": feature_drift_triggered,
                    "features": [
                        {
                            "feature": r.feature,
                            "ks_pvalue": r.ks_pvalue,
                            "psi": r.psi,
                            "severity": r.severity,
                        }
                        for r in feature_drift_results
                        if r.triggered
                    ],
                },
                prediction_drift={
                    "triggered": prediction_drift_result.triggered,
                    "chi2_pvalue": prediction_drift_result.chi2_pvalue,
                    "current_distribution": prediction_drift_result.current_distribution,
                },
                performance_degradation={
                    "triggered": degradation_triggered,
                    "hit_rate_current_window": degradation_result.hit_rate,
                    "consecutive_windows_below_threshold": degradation_result.consecutive_windows_below,
                },
            )
            self._alert_writer.write(alert)

        update_status(
            self._status_path,
            last_monitored=date_str,
            retraining_required=retraining_required,
            consecutive_degradation_windows=degradation_result.consecutive_windows_below,
            last_retraining_trigger=date_str if retraining_required else None,
            last_retraining_trigger_reason=trigger_reason,
        )

        return TriggerDecision(
            retraining_required=retraining_required,
            trigger_reason=trigger_reason,
        )

    def reset_after_successful_retraining(self, run_date: dt.date) -> None:
        update_status(
            self._status_path,
            retraining_required=False,
            last_retraining_trigger_reason=None,
        )
        logger.info("retraining_required reset to false after successful retrain on %s", run_date)
