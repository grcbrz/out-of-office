from __future__ import annotations

import csv
import json
from datetime import date

import pandas as pd
import pytest

from src.monitoring.persistence import (
    append_monitoring_csv,
    load_feature_window,
    load_monitoring_reference,
    read_status,
    save_monitoring_reference,
    update_status,
)


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


def test_save_and_load_monitoring_reference_roundtrip(tmp_path):
    save_monitoring_reference(
        artifact_dir=tmp_path,
        training_start=date(2024, 6, 1),
        training_end=date(2025, 6, 1),
        signal_counts={"BUY": 100, "HOLD": 200, "SELL": 80},
    )
    ref = load_monitoring_reference(tmp_path)
    assert ref["training_start_date"] == "2024-06-01"
    assert ref["training_end_date"] == "2025-06-01"
    assert ref["signal_counts"] == {"BUY": 100, "HOLD": 200, "SELL": 80}


def test_load_monitoring_reference_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_monitoring_reference(tmp_path)


def test_load_feature_window_filters_by_date(tmp_path):
    ticker_dir = tmp_path / "AAPL"
    ticker_dir.mkdir()
    pd.DataFrame({
        "ticker": ["AAPL"] * 5,
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        "log_return": [0.01, 0.02, 0.03, 0.04, 0.05],
    }).to_csv(ticker_dir / "2024-01-05.csv", index=False)

    out = load_feature_window(tmp_path, date(2024, 1, 2), date(2024, 1, 4), ["log_return"])
    assert sorted(out["log_return"].tolist()) == [0.02, 0.03, 0.04]


def test_load_feature_window_missing_dir_returns_empty(tmp_path):
    out = load_feature_window(tmp_path / "nope", date(2024, 1, 1), date(2024, 1, 5), ["log_return"])
    assert out["log_return"].size == 0


def test_load_feature_window_picks_latest_csv_per_ticker(tmp_path):
    ticker_dir = tmp_path / "AAPL"
    ticker_dir.mkdir()
    pd.DataFrame({
        "ticker": ["AAPL"], "date": ["2024-01-01"], "log_return": [0.99],
    }).to_csv(ticker_dir / "2024-01-01.csv", index=False)
    pd.DataFrame({
        "ticker": ["AAPL"] * 2, "date": ["2024-01-01", "2024-01-02"],
        "log_return": [0.10, 0.20],
    }).to_csv(ticker_dir / "2024-01-02.csv", index=False)

    out = load_feature_window(tmp_path, date(2024, 1, 1), date(2024, 1, 2), ["log_return"])
    assert sorted(out["log_return"].tolist()) == [0.10, 0.20]
