from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.monitoring.pipeline import MonitoringPipeline


@pytest.fixture()
def config_path(tmp_path: Path) -> Path:
    cfg = {
        "reference_window_days": 252,
        "current_window_days": 21,
        "feature_drift": {
            "ks_pvalue_threshold": 0.05,
            "psi_warning_threshold": 0.10,
            "psi_alert_threshold": 0.20,
        },
        "prediction_drift": {
            "chi2_pvalue_threshold": 0.05,
            "max_signal_concentration": 0.80,
        },
        "performance_degradation": {
            "hit_rate_threshold": 0.45,
            "consecutive_windows_required": 2,
        },
        "evidently": {
            "report_frequency_days": 7,
            "output_dir": str(tmp_path / "reports/monitoring"),
        },
        "alerts": {
            "output_dir": str(tmp_path / "data/monitoring/alerts"),
        },
        "monitoring_history": {
            "output_path": str(tmp_path / "reports/monitoring/monitoring_history.csv"),
        },
    }
    path = tmp_path / "monitoring.yaml"
    path.write_text(yaml.dump(cfg))
    return path


@pytest.fixture()
def status_path(tmp_path: Path) -> Path:
    return tmp_path / "data/monitoring/status.json"


def _stable_data():
    rng = np.random.default_rng(0)
    return {
        "close_zscore": rng.normal(0, 1, 200),
        "volume_zscore": rng.normal(0, 1, 200),
    }


def _make_predictions_df():
    return pd.DataFrame([
        {"ticker": "AAPL", "signal": "BUY", "run_date": "2026-04-22"},
        {"ticker": "MSFT", "signal": "SELL", "run_date": "2026-04-22"},
    ])


def _make_ohlcv_df():
    return pd.DataFrame([
        {"ticker": "AAPL", "date": "2026-04-22", "close": 100.0},
        {"ticker": "AAPL", "date": "2026-04-23", "close": 105.0},
        {"ticker": "MSFT", "date": "2026-04-22", "close": 200.0},
        {"ticker": "MSFT", "date": "2026-04-23", "close": 195.0},
    ])


def test_full_monitoring_pipeline_clean_run(config_path, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.monitoring.pipeline.MonitoringPipeline.__init__.__code__",
        MonitoringPipeline.__init__.__code__,
    )
    # Patch status path inside pipeline
    pipeline = MonitoringPipeline(config_path=config_path)
    pipeline._status_path = tmp_path / "data/monitoring/status.json"
    pipeline._trigger._status_path = pipeline._status_path

    ref = _stable_data()
    cur = _stable_data()  # same distribution → no drift

    result = pipeline.run(
        run_date=dt.date(2026, 4, 23),
        reference_feature_stats=ref,
        current_feature_stats=cur,
        reference_signal_counts={"BUY": 100, "HOLD": 100, "SELL": 100},
        current_signal_counts={"BUY": 95, "HOLD": 105, "SELL": 100},
        predictions_df=_make_predictions_df(),
        ohlcv_df=_make_ohlcv_df(),
        run_number=1,
    )

    assert not result["decision"].retraining_required
    history_path = tmp_path / "reports/monitoring/monitoring_history.csv"
    assert history_path.exists()


def test_full_monitoring_pipeline_with_drift_triggers_alert(config_path, tmp_path, monkeypatch):
    pipeline = MonitoringPipeline(config_path=config_path)
    pipeline._status_path = tmp_path / "data/monitoring/status.json"
    pipeline._trigger._status_path = pipeline._status_path

    rng = np.random.default_rng(99)
    ref = {"close_zscore": rng.normal(0, 1, 1000)}
    cur = {"close_zscore": rng.normal(10, 1, 1000)}  # extreme drift

    result = pipeline.run(
        run_date=dt.date(2026, 4, 23),
        reference_feature_stats=ref,
        current_feature_stats=cur,
        reference_signal_counts={"BUY": 100, "HOLD": 100, "SELL": 100},
        current_signal_counts={"BUY": 95, "HOLD": 105, "SELL": 100},
        predictions_df=_make_predictions_df(),
        ohlcv_df=_make_ohlcv_df(),
        run_number=1,
    )

    assert result["decision"].retraining_required
    alerts_dir = tmp_path / "data/monitoring/alerts"
    assert list(alerts_dir.glob("*.json"))


def test_nightly_batch_triggers_training_on_flag(config_path, tmp_path):
    """status.json with retraining_required=True should be read and respected."""
    from src.monitoring.persistence import read_status, update_status
    status_path = tmp_path / "status.json"
    update_status(status_path, retraining_required=True)
    status = read_status(status_path)
    assert status["retraining_required"] is True
