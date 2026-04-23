from __future__ import annotations

import csv
import logging
from pathlib import Path

from src.serving.schemas import PredictionRecord

logger = logging.getLogger(__name__)

_PREDICTIONS_DIR = Path("data/predictions")


def append_prediction_csv(record: PredictionRecord, predictions_dir: Path = _PREDICTIONS_DIR) -> None:
    """Append a prediction record to data/predictions/{date}.csv. Never overwrites."""
    dest = predictions_dir / f"{record.run_date}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    row = record.model_dump()
    write_header = not dest.exists()
    with dest.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    logger.debug("appended prediction for %s to %s", record.ticker, dest)
