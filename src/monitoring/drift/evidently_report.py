from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class EvidentlyReporter:
    def __init__(self, output_dir: str | Path, report_frequency_days: int = 7) -> None:
        self._output_dir = Path(output_dir)
        self._frequency = report_frequency_days

    def should_generate(self, run_number: int) -> bool:
        return run_number % self._frequency == 0

    def generate(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        run_date: str,
    ) -> None:
        try:
            from evidently import ColumnMapping
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset
        except ImportError:
            logger.warning("evidently not installed — skipping drift report")
            return

        report_dir = self._output_dir / run_date
        report_dir.mkdir(parents=True, exist_ok=True)

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_df, current_data=current_df, column_mapping=ColumnMapping())

        html_path = report_dir / "data_drift.html"
        json_path = report_dir / "data_drift.json"

        report.save_html(str(html_path))

        report_dict = report.as_dict()
        json_path.write_text(json.dumps(report_dict, default=str))

        logger.info("evidently drift report written to %s", report_dir)
