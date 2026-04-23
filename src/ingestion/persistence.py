from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records to a CSV file. Parent directories are created if needed.

    Does not overwrite existing files — idempotency is the caller's responsibility.
    """
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    logger.debug("wrote %d rows to %s", len(records), path)


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write data as JSON. Parent directories are created if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.debug("wrote json to %s", path)
