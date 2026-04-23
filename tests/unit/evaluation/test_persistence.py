from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch

import pytest

from src.evaluation.persistence import log_to_mlflow, write_csv_reports


def test_write_csv_reports_creates_files(tmp_path):
    reports = {
        "metrics.csv": [{"fold": 0, "f1": 0.4}, {"fold": 1, "f1": 0.5}],
        "per_ticker.csv": [{"ticker": "AAPL", "f1": 0.45}],
    }
    write_csv_reports(tmp_path, reports)
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "per_ticker.csv").exists()

    with (tmp_path / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2


def test_write_csv_reports_skips_empty():
    from io import StringIO
    # Should not raise for empty row lists
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        write_csv_reports(pathlib.Path(td), {"empty.csv": []})
        assert not (pathlib.Path(td) / "empty.csv").exists()


def test_write_csv_reports_appends(tmp_path):
    reports = {"out.csv": [{"x": 1}]}
    write_csv_reports(tmp_path, reports)
    write_csv_reports(tmp_path, reports)
    with (tmp_path / "out.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2


def test_log_to_mlflow_handles_import_error():
    with patch.dict("sys.modules", {"mlflow": None}):
        # Should not raise even if mlflow is missing
        log_to_mlflow({"lr": 0.01}, {"f1": 0.4}, {"model": "nhits"})


def test_log_to_mlflow_handles_generic_exception():
    import types
    fake_mlflow = types.ModuleType("mlflow")
    fake_mlflow.log_param = lambda k, v: (_ for _ in ()).throw(RuntimeError("oops"))
    with patch.dict("sys.modules", {"mlflow": fake_mlflow}):
        log_to_mlflow({"lr": 0.01}, {"f1": 0.4}, {"model": "nhits"})
