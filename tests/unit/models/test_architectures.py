from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.schema import FEATURE_COLUMNS
from src.models.architectures.autoformer import AutoformerWrapper
from src.models.architectures.nhits import NHiTSWrapper
from src.models.architectures.patchtst import PatchTSTWrapper

_WRAPPERS = [
    ("nhits", NHiTSWrapper),
    ("patchtst", PatchTSTWrapper),
    ("autoformer", AutoformerWrapper),
]


def _make_training_data(n: int = 60, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Build a small frame containing every FEATURE_COLUMNS entry plus a labelled target."""
    rng = np.random.default_rng(seed)
    columns = {col: rng.normal(size=n) for col in FEATURE_COLUMNS}
    columns["ticker_id"] = rng.integers(0, 5, size=n)
    df = pd.DataFrame(columns)
    # Roughly balanced labels so MLPClassifier doesn't degenerate.
    labels = np.tile(np.array(["SELL", "HOLD", "BUY"]), n // 3 + 1)[:n]
    rng.shuffle(labels)
    return df, pd.Series(labels)


@pytest.mark.parametrize("name,cls", _WRAPPERS)
def test_predict_raises_before_training(name, cls):
    wrapper = cls(config={})
    with pytest.raises(RuntimeError, match="not trained"):
        wrapper.predict(pd.DataFrame({c: [0.0] for c in FEATURE_COLUMNS}))


@pytest.mark.parametrize("name,cls", _WRAPPERS)
def test_train_predict_roundtrip(name, cls):
    X, y = _make_training_data()
    wrapper = cls(config={})
    wrapper.train(X, y, class_weights={0: 1.0, 1: 1.0, 2: 1.0})
    preds = wrapper.predict(X.head(10))
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (10,)
    assert set(np.unique(preds)).issubset({0, 1, 2})


@pytest.mark.parametrize("name,cls", _WRAPPERS)
def test_train_raises_when_no_labelled_rows(name, cls):
    X, _ = _make_training_data(n=10)
    y = pd.Series([None] * 10)
    wrapper = cls(config={})
    with pytest.raises(ValueError, match="no training rows"):
        wrapper.train(X, y, class_weights={})


@pytest.mark.parametrize("name,cls", _WRAPPERS)
def test_predict_proba_raises_before_training(name, cls):
    wrapper = cls(config={})
    with pytest.raises(RuntimeError, match="not trained"):
        wrapper.predict_proba(pd.DataFrame({c: [0.0] for c in FEATURE_COLUMNS}))


@pytest.mark.parametrize("name,cls", _WRAPPERS)
def test_predict_proba_returns_valid_distribution(name, cls):
    X, y = _make_training_data()
    wrapper = cls(config={})
    wrapper.train(X, y, class_weights={0: 1.0, 1: 1.0, 2: 1.0})
    proba = wrapper.predict_proba(X.head(10))
    assert proba.shape == (10, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert (proba >= 0).all() and (proba <= 1).all()


@pytest.mark.parametrize("name,cls", _WRAPPERS)
def test_predict_proba_consistent_with_predict(name, cls):
    """argmax of predict_proba must equal predict for each sample."""
    X, y = _make_training_data()
    wrapper = cls(config={})
    wrapper.train(X, y, class_weights={0: 1.0, 1: 1.0, 2: 1.0})
    preds = wrapper.predict(X.head(10))
    proba = wrapper.predict_proba(X.head(10))
    np.testing.assert_array_equal(preds, np.argmax(proba, axis=1))


@pytest.mark.parametrize("name,cls", _WRAPPERS)
def test_save_load_roundtrip(name, cls, tmp_path):
    X, y = _make_training_data()
    wrapper = cls(config={})
    wrapper.train(X, y, class_weights={0: 1.0, 1: 1.0, 2: 1.0})
    expected = wrapper.predict(X.head(5))

    artifact = tmp_path / "artifact"
    wrapper.save(artifact)
    assert (artifact / "model.pkl").exists()

    loaded = cls.load(artifact, config={})
    actual = loaded.predict(X.head(5))
    np.testing.assert_array_equal(expected, actual)
