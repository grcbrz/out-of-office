from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.serving.loader import ArtifactLoader


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data))


def _make_valid_artifact(production_dir: Path, model_name: str = "nhits") -> Path:
    model_dir = production_dir / model_name
    model_dir.mkdir(parents=True)
    _write_json(model_dir / "imputation_params.json", {"close": 100.0})
    _write_json(model_dir / "ticker_map.json", {"AAPL": 0, "MSFT": 1})
    _write_json(model_dir / "metadata.json", {"f1_macro": 0.42, "model": model_name})
    return model_dir


def test_loads_valid_artifact(tmp_path):
    _make_valid_artifact(tmp_path)
    loader = ArtifactLoader(production_dir=tmp_path)
    loader.load()
    assert loader.is_loaded
    assert loader.model_name == "nhits"
    assert loader.ticker_map == {"AAPL": 0, "MSFT": 1}


def test_degraded_mode_when_no_artifact(tmp_path):
    loader = ArtifactLoader(production_dir=tmp_path)
    loader.load()
    assert not loader.is_loaded
    assert loader.model_name is None


def test_degraded_mode_when_incomplete_artifact(tmp_path):
    model_dir = tmp_path / "nhits"
    model_dir.mkdir()
    # Only write one of the required files
    _write_json(model_dir / "metadata.json", {"f1_macro": 0.42})
    loader = ArtifactLoader(production_dir=tmp_path)
    loader.load()
    assert not loader.is_loaded


def test_loads_metadata(tmp_path):
    _make_valid_artifact(tmp_path)
    loader = ArtifactLoader(production_dir=tmp_path)
    loader.load()
    assert loader.metadata["f1_macro"] == 0.42
