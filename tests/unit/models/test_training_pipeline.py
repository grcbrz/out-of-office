from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.features.schema import FEATURE_COLUMNS
from src.models.selector import ModelResult
from src.models.training_pipeline import (
    TrainingPipeline,
    _instantiate_wrapper,
    _load_model_config,
    _set_global_seeds,
    _train_one_model,
)


def _make_global_df(n_per_ticker: int = 90, n_tickers: int = 2, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for t_idx, ticker in enumerate([f"T{i}" for i in range(n_tickers)]):
        for i in range(n_per_ticker):
            row = {"ticker": ticker, "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)}
            for c in FEATURE_COLUMNS:
                row[c] = float(rng.standard_normal()) if c != "ticker_id" else t_idx
            row["forward_return"] = float(rng.standard_normal() * 0.01)
            row["target"] = ["SELL", "HOLD", "BUY"][i % 3]
            rows.append(row)
    return pd.DataFrame(rows)


def test_set_global_seeds_is_deterministic():
    import random
    _set_global_seeds(42)
    a = (random.random(), float(np.random.rand()))
    _set_global_seeds(42)
    b = (random.random(), float(np.random.rand()))
    assert a == b


def test_load_model_config_returns_empty_for_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr("src.models.training_pipeline._MODEL_CONFIG_DIR", tmp_path)
    assert _load_model_config("does_not_exist") == {}


def test_load_model_config_reads_yaml(tmp_path, monkeypatch):
    (tmp_path / "nhits.yaml").write_text("hidden_layer_sizes: [16]\nmax_iter: 5\n")
    monkeypatch.setattr("src.models.training_pipeline._MODEL_CONFIG_DIR", tmp_path)
    cfg = _load_model_config("nhits")
    assert cfg["hidden_layer_sizes"] == [16]
    assert cfg["max_iter"] == 5


@pytest.mark.parametrize("name", ["nhits", "patchtst", "autoformer"])
def test_instantiate_wrapper_returns_correct_type(name):
    wrapper = _instantiate_wrapper(name, {})
    assert wrapper.__class__.__name__.lower().startswith(name[:5])


def test_instantiate_wrapper_unknown_raises():
    with pytest.raises(ValueError, match="unknown model"):
        _instantiate_wrapper("transformer_xl", {})


def test_train_one_model_persists_and_returns_predictions(tmp_path):
    df = _make_global_df(n_per_ticker=30)
    df["ticker_id"] = pd.Categorical(df["ticker"]).codes
    train = df.iloc[:50].reset_index(drop=True)
    val = df.iloc[50:].reset_index(drop=True)
    result = _train_one_model("nhits", train, val, class_weights={}, fold_index=0, artifact_dir=tmp_path)
    assert result.model_name == "nhits"
    assert result.f1_macro >= 0.0
    assert result.predictions is not None
    assert (tmp_path / "fold0_nhits" / "model.pkl").exists()


def test_train_one_model_no_labelled_val_returns_zero(tmp_path):
    df = _make_global_df(n_per_ticker=30)
    df["ticker_id"] = pd.Categorical(df["ticker"]).codes
    train = df.iloc[:50].reset_index(drop=True)
    val = df.iloc[50:].copy().reset_index(drop=True)
    val["target"] = None
    result = _train_one_model("nhits", train, val, class_weights={}, fold_index=0, artifact_dir=tmp_path)
    assert result.f1_macro == 0.0
    assert result.predictions is None


def test_compute_fold_metrics_includes_signal_counts():
    pipeline = TrainingPipeline()
    val = pd.DataFrame({
        "target": ["BUY", "HOLD", "SELL", "BUY"],
        "forward_return": [0.01, -0.005, -0.02, 0.03],
    })
    winner = ModelResult(model_name="nhits", f1_macro=0.5, predictions=np.array([2, 1, 0, 2]))
    metrics = pipeline._compute_fold_metrics(fold_index=3, winner=winner, val_df=val)
    assert metrics["fold"] == 3
    assert metrics["model_name"] == "nhits"
    assert metrics["buy_count"] == 2
    assert metrics["hold_count"] == 1
    assert metrics["sell_count"] == 1


def test_compute_fold_metrics_handles_no_predictions():
    pipeline = TrainingPipeline()
    val = pd.DataFrame({"target": ["BUY"]})
    winner = ModelResult(model_name="nhits", f1_macro=0.0, predictions=None)
    metrics = pipeline._compute_fold_metrics(fold_index=0, winner=winner, val_df=val)
    assert metrics == {"fold": 0, "model_name": "nhits", "f1_macro": 0.0}


def test_run_raises_when_too_few_folds():
    df = _make_global_df(n_per_ticker=30)  # not enough for 3 folds
    pipeline = TrainingPipeline(train_window=252, step_size=21)
    with pytest.raises(ValueError, match="minimum is 3"):
        pipeline.run(df)


def test_run_writes_artifact_and_monitoring_reference(tmp_path):
    df = _make_global_df(n_per_ticker=200, n_tickers=2)
    pipeline = TrainingPipeline(
        train_window=120, step_size=20,
        production_dir=tmp_path / "prod",
        fold_artifact_dir=tmp_path / "folds",
        n_workers=1,
    )

    # Stub _train_all_models so the run completes fast (skip parallel sklearn training).
    def fake_train_all(self, train_df, val_df, class_weights, fold_index, artifact_dir):
        from src.models.architectures.nhits import NHiTSWrapper
        wrapper = NHiTSWrapper({"hidden_layer_sizes": [4], "max_iter": 5})
        wrapper.train(train_df, train_df["target"], class_weights)
        artifact = Path(artifact_dir) / f"fold{fold_index}_nhits"
        wrapper.save(artifact)
        val_subset = val_df.loc[val_df["target"].notna()].reset_index(drop=True)
        preds = wrapper.predict(val_subset)
        return [ModelResult(model_name="nhits", f1_macro=0.4, artifact_path=str(artifact), predictions=preds)]

    with patch.object(TrainingPipeline, "_train_all_models", fake_train_all):
        pipeline.run(df)

    assert (tmp_path / "prod" / "nhits" / "metadata.json").exists()
    assert (tmp_path / "prod" / "nhits" / "monitoring_reference.json").exists()
