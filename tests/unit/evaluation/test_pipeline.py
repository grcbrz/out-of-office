from __future__ import annotations

import csv
import json
from datetime import date

import pytest

from src.evaluation.pipeline import EvaluationDataError, EvaluationPipeline
from src.evaluation.quality_gate import EvaluationQualityGateError


def _write_metadata(prod_dir, model_name, metadata):
    model_dir = prod_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "metadata.json").write_text(json.dumps(metadata))


def _passing_metrics():
    return {
        "f1_macro": 0.42, "mcc": 0.10, "hit_rate": 0.55,
        "signal_distribution": {"BUY": 0.3, "HOLD": 0.4, "SELL": 0.3},
    }


def _passing_metadata():
    return {
        "model_name": "nhits",
        "fold_metrics": [
            {"fold": 0, "f1_macro": 0.40, "mcc": 0.08, "hit_rate": 0.52},
            {"fold": 1, "f1_macro": 0.42, "mcc": 0.10, "hit_rate": 0.55},
        ],
        "aggregated": {"f1_macro_mean": 0.41, "f1_macro_std": 0.01},
        "production_fold": _passing_metrics(),
    }


def test_run_writes_reports_and_passes_gate(tmp_path):
    prod_dir = tmp_path / "production"
    report_dir = tmp_path / "reports"
    _write_metadata(prod_dir, "nhits", _passing_metadata())

    EvaluationPipeline(production_dir=prod_dir, report_dir=report_dir).run(date(2026, 4, 24))

    date_dir = report_dir / "2026-04-24"
    assert (date_dir / "metrics_per_fold.csv").exists()
    assert (date_dir / "metrics_aggregated.csv").exists()
    gate = json.loads((date_dir / "quality_gate_result.json").read_text())
    assert gate["passed"] is True
    assert gate["failure_message"] is None


def test_run_raises_when_gate_fails_but_still_writes_reports(tmp_path):
    prod_dir = tmp_path / "production"
    report_dir = tmp_path / "reports"
    metadata = _passing_metadata()
    metadata["production_fold"]["f1_macro"] = 0.10  # below 0.35 threshold
    _write_metadata(prod_dir, "nhits", metadata)

    pipeline = EvaluationPipeline(production_dir=prod_dir, report_dir=report_dir)
    with pytest.raises(EvaluationQualityGateError):
        pipeline.run(date(2026, 4, 24))

    gate = json.loads((report_dir / "2026-04-24" / "quality_gate_result.json").read_text())
    assert gate["passed"] is False
    assert "f1_macro" in gate["failure_message"]


def test_run_raises_when_no_production_directory(tmp_path):
    pipeline = EvaluationPipeline(production_dir=tmp_path / "missing", report_dir=tmp_path / "r")
    with pytest.raises(EvaluationDataError):
        pipeline.run(date(2026, 4, 24))


def test_run_raises_when_metadata_missing_production_fold(tmp_path):
    prod_dir = tmp_path / "production"
    _write_metadata(prod_dir, "nhits", {"model_name": "nhits", "fold_metrics": []})
    pipeline = EvaluationPipeline(production_dir=prod_dir, report_dir=tmp_path / "r")
    with pytest.raises(EvaluationDataError, match="production_fold"):
        pipeline.run(date(2026, 4, 24))


def test_run_appends_to_existing_csv(tmp_path):
    """Spec 05 §6.2: CSVs append across runs, never overwrite."""
    prod_dir = tmp_path / "production"
    report_dir = tmp_path / "reports"
    _write_metadata(prod_dir, "nhits", _passing_metadata())

    pipeline = EvaluationPipeline(production_dir=prod_dir, report_dir=report_dir)
    pipeline.run(date(2026, 4, 24))
    pipeline.run(date(2026, 4, 24))

    csv_path = report_dir / "2026-04-24" / "metrics_per_fold.csv"
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4  # 2 folds * 2 runs
