from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

from src.ingestion.persistence import write_json

logger = logging.getLogger(__name__)

_ALERTS_DIR = Path("data/raw/alerts")


class AlertWriter:
    """Writes a structured alert JSON to data/raw/alerts/{date}.json.

    Only written when at least one failure occurred. Calling write() with
    an empty failure set is a no-op.
    """

    def __init__(self, alerts_dir: Path = _ALERTS_DIR) -> None:
        self._alerts_dir = alerts_dir

    def write(self, run_date: date, payload: dict[str, Any]) -> None:
        has_failures = (
            payload.get("ohlcv_failed")
            or payload.get("sentiment_failed")
        )
        if not has_failures:
            return
        path = self._alerts_dir / f"{run_date}.json"
        write_json(path, payload)
        logger.warning("alert written to %s", path)
