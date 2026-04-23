from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict
from pathlib import Path

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
