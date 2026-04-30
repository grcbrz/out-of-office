from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.features.schema import FEATURE_COLUMNS
from src.models.selector import ModelResult
from src.models.training_pipeline import (
    _BASELINE_NAME,
    TrainingPipeline,
    _instantiate_wrapper,
    _load_model_config,
    _set_global_seeds,
    _train_one_model,
)


def _make_global_df(
    n_per_ticker: int = 90, n_tickers: int = 2, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for t_idx, ticker in enumerate([f"T{i}" for i in range(n_tickers)]):
        for i in range(n_per_ticker):
            row = {
                "ticker": ticker,
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
            }
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
    (tmp_path / "lightgbm.yaml").write_text("n_estimators: 16\nlearning_rate: 0.1\n")
    monkeypatch.setattr("src.models.training_pipeline._MODEL_CONFIG_DIR", tmp_path)
    cfg = _load_model_config("lightgbm")
    assert cfg["n_estimators"] == 16
    assert cfg["learning_rate"] == 0.1


@pytest.mark.parametrize(
    "name,prefix",
    [
        ("lightgbm", "lightgbm"),
        # RandomForestWrapper is dormant in PR 1: not in _CANDIDATE_NAMES so
        # it is never trained on production folds, but _instantiate_wrapper
        # already routes the name. The PR 2 activation flips the candidate
        # tuple; this test ensures the routing works the moment that flip
        # lands so PR 2 stays mechanical.
        ("randomforest", "randomforest"),
        (_BASELINE_NAME, "baseline"),
    ],
)
def test_instantiate_wrapper_returns_correct_type(name, prefix):
    wrapper = _instantiate_wrapper(name, {})
    assert wrapper.__class__.__name__.lower().startswith(prefix)


def test_instantiate_wrapper_unknown_raises():
    with pytest.raises(ValueError, match="unknown model"):
        _instantiate_wrapper("transformer_xl", {})


def test_train_one_model_persists_and_returns_predictions(tmp_path):
    df = _make_global_df(n_per_ticker=30)
    df["ticker_id"] = pd.Categorical(df["ticker"]).codes
    train = df.iloc[:50].reset_index(drop=True)
    val = df.iloc[50:].reset_index(drop=True)
    result = _train_one_model(
        "lightgbm", train, val, class_weights={0: 1.0, 1: 1.0, 2: 1.0},
        fold_index=0, artifact_dir=tmp_path,
    )
    assert result.model_name == "lightgbm"
    assert result.f1_macro >= 0.0
    assert result.predictions is not None
    assert result.predictions_proba is not None
    assert result.predictions_proba.shape == (len(val), 3)
    assert (tmp_path / "fold0_lightgbm" / "model.pkl").exists()


def test_train_one_model_baseline_marks_is_baseline(tmp_path):
    df = _make_global_df(n_per_ticker=30)
    df["ticker_id"] = pd.Categorical(df["ticker"]).codes
    train = df.iloc[:50].reset_index(drop=True)
    val = df.iloc[50:].reset_index(drop=True)
    result = _train_one_model(
        _BASELINE_NAME, train, val, class_weights={}, fold_index=0, artifact_dir=tmp_path,
    )
    assert result.is_baseline is True
    assert result.predictions is not None
    assert result.predictions_proba is not None


def test_train_one_model_no_labelled_val_returns_zero(tmp_path):
    df = _make_global_df(n_per_ticker=30)
    df["ticker_id"] = pd.Categorical(df["ticker"]).codes
    train = df.iloc[:50].reset_index(drop=True)
    val = df.iloc[50:].copy().reset_index(drop=True)
    val["target"] = None
    result = _train_one_model(
        "lightgbm", train, val, class_weights={0: 1.0, 1: 1.0, 2: 1.0},
        fold_index=0, artifact_dir=tmp_path,
    )
    assert result.f1_macro == 0.0
    assert result.predictions is None


def test_compute_fold_metrics_includes_signal_counts_and_baseline_field():
    pipeline = TrainingPipeline()
    val = pd.DataFrame({
        "target": ["BUY", "HOLD", "SELL", "BUY"],
        "forward_return": [0.01, -0.005, -0.02, 0.03],
    })
    winner = ModelResult(
        model_name="lightgbm", f1_macro=0.5, predictions=np.array([2, 1, 0, 2]),
    )
    metrics = pipeline._compute_fold_metrics(
        fold_index=3, winner=winner, val_df=val, baseline_f1=0.30,
    )
    assert metrics["fold"] == 3
    assert metrics["model_name"] == "lightgbm"
    assert metrics["buy_count"] == 2
    assert metrics["hold_count"] == 1
    assert metrics["sell_count"] == 1
    assert metrics["baseline_f1_macro"] == 0.30


def test_compute_fold_metrics_applies_threshold_to_financial_only():
    """F1 stays on raw preds; financial metrics use threshold-applied preds.

    All four predictions are BUY; two are high-confidence (max_proba=0.80) and
    two are low-confidence (max_proba=0.45). With τ=0.50 only the two
    high-confidence BUYs survive — buy_count must be 2 (not 4) and the
    rest become HOLD. F1 must reflect raw predictions (against true labels).
    """
    pipeline = TrainingPipeline()
    val = pd.DataFrame({
        "target": ["BUY", "BUY", "BUY", "BUY"],
        "forward_return": [0.02, 0.03, -0.04, -0.05],
    })
    proba = np.array([
        [0.10, 0.10, 0.80],   # high
        [0.10, 0.10, 0.80],   # high
        [0.30, 0.25, 0.45],   # low → demoted
        [0.30, 0.25, 0.45],   # low → demoted
    ])
    winner = ModelResult(
        model_name="lightgbm",
        f1_macro=0.5,
        predictions=np.array([2, 2, 2, 2]),  # all BUY
        predictions_proba=proba,
        confidence_threshold=0.50,
    )
    metrics = pipeline._compute_fold_metrics(
        fold_index=7, winner=winner, val_df=val, baseline_f1=0.30,
    )
    # Classification: against true labels of all-BUY, raw preds of all-BUY → F1 = 1
    # for the BUY class but 0 for SELL/HOLD; macro = 1/3.
    assert metrics["f1_macro"] == pytest.approx(1 / 3)
    # Financial: only 2 BUYs survived τ; the rest are HOLD.
    assert metrics["buy_count"] == 2
    assert metrics["hold_count"] == 2
    assert metrics["sell_count"] == 0
    assert metrics["confidence_threshold"] == 0.50


def test_compute_fold_metrics_handles_no_predictions():
    pipeline = TrainingPipeline()
    val = pd.DataFrame({"target": ["BUY"]})
    winner = ModelResult(model_name="lightgbm", f1_macro=0.0, predictions=None)
    metrics = pipeline._compute_fold_metrics(
        fold_index=0, winner=winner, val_df=val, baseline_f1=None,
    )
    assert metrics == {
        "fold": 0,
        "model_name": "lightgbm",
        "f1_macro": 0.0,
        "baseline_f1_macro": None,
        "confidence_threshold": None,
    }


def test_run_raises_when_too_few_folds():
    df = _make_global_df(n_per_ticker=30)
    pipeline = TrainingPipeline(train_window=252, step_size=21)
    with pytest.raises(ValueError, match="minimum is 3"):
        pipeline.run(df)


def test_run_writes_artifact_and_monitoring_reference(tmp_path):
    df = _make_global_df(n_per_ticker=200, n_tickers=2)
    pipeline = TrainingPipeline(
        train_window=120, step_size=20,
        production_dir=tmp_path / "prod",
        fold_artifact_dir=tmp_path / "folds",
    )

    # Stub _train_all_models so the run completes fast and returns a baseline too.
    def fake_train_all(self, train_df, val_df, class_weights, fold_index, artifact_dir):
        from src.models.architectures.baseline import BaselineLastDirectionWrapper
        from src.models.architectures.lightgbm import LightGBMWrapper

        artifact_dir = Path(artifact_dir)
        val_subset = val_df.loc[val_df["target"].notna()].reset_index(drop=True)

        lgbm = LightGBMWrapper({"n_estimators": 10, "num_leaves": 7})
        lgbm.train(train_df, train_df["target"], class_weights)
        lgbm_path = artifact_dir / f"fold{fold_index}_lightgbm"
        lgbm.save(lgbm_path)
        lgbm_preds = lgbm.predict(val_subset)

        baseline = BaselineLastDirectionWrapper({})
        baseline.train(train_df, train_df["target"], class_weights)
        baseline_path = artifact_dir / f"fold{fold_index}_{_BASELINE_NAME}"
        baseline.save(baseline_path)
        baseline_preds = baseline.predict(val_subset)

        return [
            ModelResult(
                model_name="lightgbm", f1_macro=0.4,
                artifact_path=str(lgbm_path), predictions=lgbm_preds,
            ),
            ModelResult(
                model_name=_BASELINE_NAME, f1_macro=0.30,
                artifact_path=str(baseline_path), predictions=baseline_preds,
                is_baseline=True,
            ),
        ]

    with patch.object(TrainingPipeline, "_train_all_models", fake_train_all):
        pipeline.run(df)

    assert (tmp_path / "prod" / "lightgbm" / "metadata.json").exists()
    assert (tmp_path / "prod" / "lightgbm" / "monitoring_reference.json").exists()
    import json
    metadata = json.loads(
        (tmp_path / "prod" / "lightgbm" / "metadata.json").read_text()
    )
    # Baseline metrics persisted alongside production-fold metrics so the gate can read them.
    assert metadata["baseline"]["name"] == _BASELINE_NAME
    assert metadata["production_fold"]["baseline_f1_macro"] == 0.30
