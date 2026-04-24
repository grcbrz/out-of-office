from __future__ import annotations

import csv
import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_HISTORY_COLUMNS = [
    "date",
    "n_features_drifted",
    "max_psi",
    "prediction_drift_pvalue",
    "hit_rate_21d",
    "retraining_triggered",
    "trigger_reason",
]


def append_monitoring_csv(row: dict, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_HISTORY_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    logger.debug("monitoring history appended: %s", path)


def update_status(status_path: str | Path, **updates) -> dict:
    path = Path(status_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        current = json.loads(path.read_text())
    else:
        current = {
            "last_monitored": None,
            "retraining_required": False,
            "consecutive_degradation_windows": 0,
            "last_retraining_trigger": None,
            "last_retraining_trigger_reason": None,
        }

    current.update(updates)
    path.write_text(json.dumps(current, indent=2, default=str))
    return current


def read_status(status_path: str | Path) -> dict:
    path = Path(status_path)
    if not path.exists():
        return {
            "last_monitored": None,
            "retraining_required": False,
            "consecutive_degradation_windows": 0,
            "last_retraining_trigger": None,
            "last_retraining_trigger_reason": None,
        }
    return json.loads(path.read_text())


def save_monitoring_reference(
    artifact_dir: str | Path,
    training_start: date,
    training_end: date,
    signal_counts: dict[str, int],
) -> None:
    """Persist the reference window pointer used by MonitoringPipeline.

    Stores only dates + signal counts — feature distributions are recomputed
    from data/features/ at monitoring time to avoid duplicating raw data.
    """
    path = Path(artifact_dir) / "monitoring_reference.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "training_start_date": str(training_start),
        "training_end_date": str(training_end),
        "signal_counts": {k: int(v) for k, v in signal_counts.items()},
    }
    path.write_text(json.dumps(payload, indent=2))


def load_monitoring_reference(artifact_dir: str | Path) -> dict:
    path = Path(artifact_dir) / "monitoring_reference.json"
    if not path.exists():
        raise FileNotFoundError(f"monitoring_reference.json missing at {path}")
    return json.loads(path.read_text())


def load_feature_window(
    features_dir: str | Path,
    start_date: date,
    end_date: date,
    columns: list[str],
) -> dict[str, np.ndarray]:
    """Load per-feature value arrays from data/features/ within [start_date, end_date].

    Picks the most recent run-date CSV per ticker (each file contains the full
    history up to that run-date) and slices to the requested window.
    """
    features_dir = Path(features_dir)
    if not features_dir.exists():
        return {col: np.array([]) for col in columns}

    frames: list[pd.DataFrame] = []
    for ticker_dir in sorted(features_dir.iterdir()):
        if not ticker_dir.is_dir():
            continue
        csv_files = sorted(ticker_dir.glob("*.csv"))
        if not csv_files:
            continue
        df = pd.read_csv(csv_files[-1])
        df["date"] = pd.to_datetime(df["date"]).dt.date
        mask = (df["date"] >= start_date) & (df["date"] <= end_date)
        frames.append(df.loc[mask])

    if not frames:
        return {col: np.array([]) for col in columns}

    combined = pd.concat(frames, ignore_index=True)
    return {
        col: combined[col].dropna().to_numpy()
        for col in columns
        if col in combined.columns
    }
