from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import yaml

from src.evaluation.persistence import log_to_mlflow, write_csv_reports
from src.evaluation.quality_gate import EvaluationQualityGateError, QualityGate

logger = logging.getLogger(__name__)

_PRODUCTION_DIR = Path("models/production")
_REPORT_DIR = Path("reports/evaluation")
_CONFIG_PATH = Path("configs/evaluation.yaml")


class EvaluationDataError(Exception):
    """Raised when the production artifact is missing the metrics needed for evaluation."""


class EvaluationPipeline:
    """Loads production-fold metrics, enforces quality gates, persists evaluation reports.

    Per Spec 05 §6: writes per-fold and aggregated CSVs (append mode) plus a
    quality_gate_result.json under reports/evaluation/{run_date}/. Re-raises
    EvaluationQualityGateError after persisting reports so the caller can
    decide whether to keep the production artifact.
    """

    def __init__(
        self,
        production_dir: Path = _PRODUCTION_DIR,
        report_dir: Path = _REPORT_DIR,
        config_path: Path = _CONFIG_PATH,
    ) -> None:
        self._production_dir = production_dir
        self._report_dir = report_dir
        self._gate = self._load_quality_gate(config_path)

    def run(self, run_date: date) -> None:
        metadata = self._load_production_metadata()
        production_metrics = metadata.get("production_fold")
        if not production_metrics:
            raise EvaluationDataError(
                "production_fold metrics missing from artifact metadata.json"
            )

        date_dir = self._report_dir / str(run_date)
        write_csv_reports(date_dir, {
            "metrics_per_fold.csv": metadata.get("fold_metrics", []),
            "metrics_aggregated.csv": [metadata.get("aggregated", {})] if metadata.get("aggregated") else [],
        })

        passed, failure_msg = self._evaluate_gate(production_metrics)
        self._write_gate_result(date_dir, production_metrics, passed, failure_msg)

        log_to_mlflow(
            run_params={
                "model_name": metadata.get("model_name", ""),
                "run_date": str(run_date),
            },
            metrics={k: v for k, v in production_metrics.items() if isinstance(v, (int, float))},
            tags={"production_fold": "true", "quality_gate_passed": str(passed).lower()},
        )

        if not passed:
            raise EvaluationQualityGateError(failure_msg or "quality gate failed")

        logger.info("evaluation complete for %s: gate passed", run_date)

    def _load_production_metadata(self) -> dict:
        if not self._production_dir.exists():
            raise EvaluationDataError(f"no production directory at {self._production_dir}")
        model_dirs = [p for p in self._production_dir.iterdir() if p.is_dir()]
        if not model_dirs:
            raise EvaluationDataError(f"no production model under {self._production_dir}")
        metadata_path = model_dirs[0] / "metadata.json"
        if not metadata_path.exists():
            raise EvaluationDataError(f"missing metadata.json at {metadata_path}")
        with metadata_path.open() as f:
            return json.load(f)

    def _evaluate_gate(self, production_metrics: dict) -> tuple[bool, str | None]:
        try:
            self._gate.check(production_metrics)
            return True, None
        except EvaluationQualityGateError as exc:
            return False, str(exc)

    @staticmethod
    def _load_quality_gate(config_path: Path) -> QualityGate:
        if not config_path.exists():
            return QualityGate()
        with config_path.open() as f:
            cfg = yaml.safe_load(f) or {}
        gates = cfg.get("quality_gates", {})
        return QualityGate(
            f1_macro_min=gates.get("f1_macro_min", 0.35),
            mcc_min=gates.get("mcc_min", 0.05),
            hit_rate_min=gates.get("hit_rate_min", 0.50),
            max_signal_concentration=gates.get("max_signal_concentration", 0.80),
        )

    @staticmethod
    def _write_gate_result(
        date_dir: Path, metrics: dict, passed: bool, failure_msg: str | None,
    ) -> None:
        date_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "passed": passed,
            "failure_message": failure_msg,
            "metrics": {k: v for k, v in metrics.items() if not isinstance(v, dict)},
        }
        (date_dir / "quality_gate_result.json").write_text(json.dumps(payload, indent=2, default=str))
