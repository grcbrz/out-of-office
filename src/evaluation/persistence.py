from __future__ import annotations

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def write_csv_reports(report_dir: Path, reports: dict[str, list[dict]]) -> None:
    """Write multiple CSV reports to report_dir in append mode."""
    report_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in reports.items():
        if not rows:
            continue
        path = report_dir / filename
        write_header = not path.exists()
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
        logger.debug("appended %d rows to %s", len(rows), path)


def log_to_mlflow(run_params: dict, metrics: dict, tags: dict) -> None:
    """Log params, metrics, and tags to MLflow if available."""
    try:
        import mlflow
        for k, v in run_params.items():
            mlflow.log_param(k, v)
        for k, v in metrics.items():
            if v is not None:
                mlflow.log_metric(k, float(v))
        for k, v in tags.items():
            mlflow.set_tag(k, str(v))
    except ImportError:
        logger.debug("mlflow not installed; skipping tracking")
    except Exception as exc:
        logger.warning("mlflow logging failed: %s", exc)
