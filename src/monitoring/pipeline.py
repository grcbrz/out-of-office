from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.monitoring.alerts import AlertWriter
from src.monitoring.degradation import DegradationDetector
from src.monitoring.drift.evidently_report import EvidentlyReporter
from src.monitoring.drift.feature_drift import FeatureDriftDetector
from src.monitoring.drift.prediction_drift import PredictionDriftDetector
from src.monitoring.persistence import append_monitoring_csv, read_status
from src.monitoring.trigger import RetrainingTrigger

logger = logging.getLogger(__name__)


class MonitoringPipeline:
    def __init__(self, config_path: str | Path = "configs/monitoring.yaml") -> None:
        with open(config_path) as fh:
            self._cfg = yaml.safe_load(fh)

        alerts_dir = self._cfg["alerts"]["output_dir"]
        status_path = "data/monitoring/status.json"
        history_path = self._cfg["monitoring_history"]["output_path"]

        self._status_path = Path(status_path)
        self._history_path = Path(history_path)

        fd_cfg = self._cfg["feature_drift"]
        self._feature_detector = FeatureDriftDetector(
            ks_pvalue_threshold=fd_cfg["ks_pvalue_threshold"],
            psi_warning_threshold=fd_cfg["psi_warning_threshold"],
            psi_alert_threshold=fd_cfg["psi_alert_threshold"],
        )

        pd_cfg = self._cfg["prediction_drift"]
        self._prediction_detector = PredictionDriftDetector(
            chi2_pvalue_threshold=pd_cfg["chi2_pvalue_threshold"],
            max_signal_concentration=pd_cfg["max_signal_concentration"],
        )

        deg_cfg = self._cfg["performance_degradation"]
        self._degradation_detector = DegradationDetector(
            hit_rate_threshold=deg_cfg["hit_rate_threshold"],
            consecutive_windows_required=deg_cfg["consecutive_windows_required"],
        )

        ev_cfg = self._cfg["evidently"]
        self._evidently_reporter = EvidentlyReporter(
            output_dir=ev_cfg["output_dir"],
            report_frequency_days=ev_cfg["report_frequency_days"],
        )

        alert_writer = AlertWriter(output_dir=alerts_dir)
        self._trigger = RetrainingTrigger(
            status_path=status_path,
            alert_writer=alert_writer,
        )

    def run(
        self,
        run_date: dt.date,
        reference_feature_stats: dict[str, np.ndarray],
        current_feature_stats: dict[str, np.ndarray],
        reference_signal_counts: dict[str, int],
        current_signal_counts: dict[str, int],
        predictions_df: pd.DataFrame,
        ohlcv_df: pd.DataFrame,
        run_number: int = 1,
        reference_df: pd.DataFrame | None = None,
        current_df: pd.DataFrame | None = None,
    ) -> dict:
        status = read_status(self._status_path)
        prev_consecutive = status.get("consecutive_degradation_windows", 0)

        feature_results = self._feature_detector.run(reference_feature_stats, current_feature_stats)
        prediction_result = self._prediction_detector.run(reference_signal_counts, current_signal_counts)
        degradation_result = self._degradation_detector.run(predictions_df, ohlcv_df, prev_consecutive)

        decision = self._trigger.evaluate(
            run_date=run_date,
            feature_drift_results=feature_results,
            prediction_drift_result=prediction_result,
            degradation_result=degradation_result,
        )

        n_features_drifted = sum(1 for r in feature_results if r.triggered)
        max_psi = max((r.psi for r in feature_results), default=0.0)

        row = {
            "date": str(run_date),
            "n_features_drifted": n_features_drifted,
            "max_psi": max_psi,
            "prediction_drift_pvalue": prediction_result.chi2_pvalue,
            "hit_rate_21d": degradation_result.hit_rate,
            "retraining_triggered": decision.retraining_required,
            "trigger_reason": decision.trigger_reason,
        }
        append_monitoring_csv(row, self._history_path)

        if self._evidently_reporter.should_generate(run_number) and reference_df is not None and current_df is not None:
            self._evidently_reporter.generate(
                reference_df=reference_df,
                current_df=current_df,
                run_date=str(run_date),
            )

        logger.info(
            "monitoring complete for %s — retraining_required=%s reason=%s",
            run_date, decision.retraining_required, decision.trigger_reason,
        )
        return {"decision": decision, "feature_drift": feature_results, "prediction_drift": prediction_result, "degradation": degradation_result}
