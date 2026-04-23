from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.monitoring.alerts import AlertWriter
from src.monitoring.degradation import DegradationResult
from src.monitoring.drift.feature_drift import FeatureDriftResult
from src.monitoring.drift.prediction_drift import PredictionDriftResult
from src.monitoring.trigger import RetrainingTrigger


def _make_trigger(tmp_path: Path) -> tuple[RetrainingTrigger, Path, Path]:
    alerts_dir = tmp_path / "alerts"
    status_path = tmp_path / "status.json"
    alert_writer = AlertWriter(output_dir=alerts_dir)
    trigger = RetrainingTrigger(status_path=status_path, alert_writer=alert_writer)
    return trigger, status_path, alerts_dir


def _clean_feature_result() -> FeatureDriftResult:
    return FeatureDriftResult(feature="f", ks_pvalue=0.5, psi=0.05, severity="none", triggered=False)


def _clean_pred_result() -> PredictionDriftResult:
    return PredictionDriftResult(
        triggered=False, chi2_pvalue=0.5,
        current_distribution={"BUY": 0.33, "HOLD": 0.34, "SELL": 0.33},
        degenerate_signal=False, dominant_class=None,
    )


def _clean_degradation() -> DegradationResult:
    return DegradationResult(hit_rate=0.6, triggered=False, consecutive_windows_below=0)


def test_status_json_updated(tmp_path):
    trigger, status_path, _ = _make_trigger(tmp_path)
    trigger.evaluate(
        run_date=dt.date(2026, 4, 23),
        feature_drift_results=[_clean_feature_result()],
        prediction_drift_result=_clean_pred_result(),
        degradation_result=_clean_degradation(),
    )
    status = json.loads(status_path.read_text())
    assert status["last_monitored"] == "2026-04-23"
    assert status["retraining_required"] is False


def test_alert_written_on_breach(tmp_path):
    trigger, status_path, alerts_dir = _make_trigger(tmp_path)
    drifted = FeatureDriftResult(feature="x", ks_pvalue=0.01, psi=0.25, severity="significant", triggered=True)
    trigger.evaluate(
        run_date=dt.date(2026, 4, 23),
        feature_drift_results=[drifted],
        prediction_drift_result=_clean_pred_result(),
        degradation_result=_clean_degradation(),
    )
    alert_files = list(alerts_dir.glob("*.json"))
    assert len(alert_files) == 1
    alert = json.loads(alert_files[0].read_text())
    assert alert["retraining_triggered"] is True


def test_no_alert_on_clean_run(tmp_path):
    trigger, _, alerts_dir = _make_trigger(tmp_path)
    trigger.evaluate(
        run_date=dt.date(2026, 4, 23),
        feature_drift_results=[_clean_feature_result()],
        prediction_drift_result=_clean_pred_result(),
        degradation_result=_clean_degradation(),
    )
    assert not list(alerts_dir.glob("*.json"))


def test_status_reset_after_retraining(tmp_path):
    trigger, status_path, _ = _make_trigger(tmp_path)
    drifted = FeatureDriftResult(feature="x", ks_pvalue=0.01, psi=0.25, severity="significant", triggered=True)
    trigger.evaluate(
        run_date=dt.date(2026, 4, 23),
        feature_drift_results=[drifted],
        prediction_drift_result=_clean_pred_result(),
        degradation_result=_clean_degradation(),
    )
    assert json.loads(status_path.read_text())["retraining_required"] is True

    trigger.reset_after_successful_retraining(dt.date(2026, 4, 24))
    assert json.loads(status_path.read_text())["retraining_required"] is False


def test_status_persists_on_failed_retrain(tmp_path):
    # Simulate: trigger set → quality gate failed → flag stays True
    trigger, status_path, _ = _make_trigger(tmp_path)
    drifted = FeatureDriftResult(feature="x", ks_pvalue=0.01, psi=0.25, severity="significant", triggered=True)
    trigger.evaluate(
        run_date=dt.date(2026, 4, 23),
        feature_drift_results=[drifted],
        prediction_drift_result=_clean_pred_result(),
        degradation_result=_clean_degradation(),
    )
    # Quality gate failure — do NOT call reset; evaluate again next night
    trigger.evaluate(
        run_date=dt.date(2026, 4, 24),
        feature_drift_results=[drifted],
        prediction_drift_result=_clean_pred_result(),
        degradation_result=_clean_degradation(),
    )
    assert json.loads(status_path.read_text())["retraining_required"] is True
