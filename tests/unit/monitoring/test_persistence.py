from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.monitoring.persistence import append_monitoring_csv, read_status, update_status


def test_monitoring_history_appended(tmp_path):
    path = tmp_path / "monitoring_history.csv"
    row1 = {
        "date": "2026-04-22", "n_features_drifted": 0, "max_psi": 0.05,
        "prediction_drift_pvalue": 0.4, "hit_rate_21d": 0.55,
        "retraining_triggered": False, "trigger_reason": None,
    }
    row2 = {
        "date": "2026-04-23", "n_features_drifted": 1, "max_psi": 0.22,
        "prediction_drift_pvalue": 0.03, "hit_rate_21d": 0.42,
        "retraining_triggered": True, "trigger_reason": "feature_drift",
    }
    append_monitoring_csv(row1, path)
    append_monitoring_csv(row2, path)

    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert rows[0]["date"] == "2026-04-22"
    assert rows[1]["trigger_reason"] == "feature_drift"


def test_status_file_created_if_missing(tmp_path):
    path = tmp_path / "sub" / "status.json"
    status = update_status(path, last_monitored="2026-04-23", retraining_required=True)
    assert path.exists()
    assert status["retraining_required"] is True


def test_read_status_defaults_when_missing(tmp_path):
    path = tmp_path / "does_not_exist.json"
    status = read_status(path)
    assert status["retraining_required"] is False
    assert status["consecutive_degradation_windows"] == 0


def test_update_status_merges(tmp_path):
    path = tmp_path / "status.json"
    update_status(path, retraining_required=True, consecutive_degradation_windows=1)
    update_status(path, retraining_required=False)
    status = json.loads(path.read_text())
    assert status["retraining_required"] is False
    assert status["consecutive_degradation_windows"] == 1  # preserved from first update
