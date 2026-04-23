from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MonitoringAlert:
    alert_date: str
    retraining_triggered: bool
    feature_drift: dict
    prediction_drift: dict
    performance_degradation: dict


class AlertWriter:
    def __init__(self, output_dir: str | Path) -> None:
        self._output_dir = Path(output_dir)

    def write(self, alert: MonitoringAlert) -> Path:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / f"{alert.alert_date}.json"
        path.write_text(json.dumps(asdict(alert), indent=2, default=str))
        logger.info("monitoring alert written: %s", path)
        return path
