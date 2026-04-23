from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.models.persistence import load_artifact, save_artifact


def test_save_artifact_writes_files(tmp_path):
    model_dir = tmp_path / "src_model"
    model_dir.mkdir()
    prod_dir = tmp_path / "production"

    save_artifact(
        model_name="nhits",
        model_dir=model_dir,
        imputation_params={"close": 100.0},
        ticker_map={"AAPL": 0},
        class_weights={0: 1.0, 1: 1.0, 2: 1.0},
        metadata={"f1_macro": 0.42},
        production_dir=prod_dir,
    )

    dest = prod_dir / "nhits"
    assert (dest / "imputation_params.json").exists()
    assert (dest / "ticker_map.json").exists()
    assert (dest / "metadata.json").exists()
    assert (dest / "class_weights.json").exists()


def test_save_artifact_overwrites_existing(tmp_path):
    prod_dir = tmp_path / "production"
    model_dir = tmp_path / "src"
    model_dir.mkdir()

    for f1 in (0.40, 0.45):
        save_artifact(
            model_name="nhits",
            model_dir=model_dir,
            imputation_params={},
            ticker_map={},
            class_weights={},
            metadata={"f1_macro": f1},
            production_dir=prod_dir,
        )

    meta = json.loads((prod_dir / "nhits" / "metadata.json").read_text())
    assert meta["f1_macro"] == 0.45


def test_load_artifact_reads_files(tmp_path):
    model_dir = tmp_path / "src"
    model_dir.mkdir()
    prod_dir = tmp_path / "production"

    save_artifact(
        model_name="nhits",
        model_dir=model_dir,
        imputation_params={"close": 99.0},
        ticker_map={"AAPL": 0},
        class_weights={0: 1.0},
        metadata={"f1_macro": 0.41},
        production_dir=prod_dir,
    )

    result = load_artifact("nhits", prod_dir)
    assert result["ticker_map"] == {"AAPL": 0}
    assert result["metadata"]["f1_macro"] == 0.41


def test_save_artifact_copies_weights_if_present(tmp_path):
    model_dir = tmp_path / "src"
    model_dir.mkdir()
    (model_dir / "model.pt").write_bytes(b"fake weights")
    prod_dir = tmp_path / "production"

    save_artifact(
        model_name="nhits",
        model_dir=model_dir,
        imputation_params={},
        ticker_map={},
        class_weights={},
        metadata={},
        production_dir=prod_dir,
    )

    assert (prod_dir / "nhits" / "model.pt").exists()
