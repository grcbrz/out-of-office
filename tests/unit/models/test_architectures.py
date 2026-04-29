"""Architecture-suite tests.

v1 ships with a single production candidate (LightGBM) plus the naive baseline.
The previous draft parametrised over `nhits`, `patchtst`, `autoformer`; those
files were placeholder sklearn estimators under transformer-flavoured names and
have been removed. New transformer candidates re-enter this matrix when their
wrappers and configs are merged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.schema import FEATURE_COLUMNS
from src.models.architectures.baseline import BaselineLastDirectionWrapper
from src.models.architectures.lightgbm import LightGBMWrapper

_WRAPPERS = [
    ("lightgbm", LightGBMWrapper),
    ("baseline_last_direction", BaselineLastDirectionWrapper),
]


def _make_training_data(n: int = 60, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic frame with every FEATURE_COLUMNS entry plus a balanced target."""
    rng = np.random.default_rng(seed)
    columns = {col: rng.normal(size=n) for col in FEATURE_COLUMNS}
    columns["ticker_id"] = rng.integers(0, 5, size=n)
    df = pd.DataFrame(columns)
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


# ----- LightGBM-specific behaviour -----

def test_lightgbm_class_weights_applied(monkeypatch):
    """sample_weight reaches the underlying LGBMClassifier.fit call.

    Spying on ``LGBMClassifier.fit`` keeps this regression-proof: if a future
    refactor stops passing sample_weight, this test fails immediately.
    """
    import lightgbm as lgb

    captured: dict = {}
    original_fit = lgb.LGBMClassifier.fit

    def spy_fit(self, X, y, *args, **kwargs):
        captured["sample_weight"] = kwargs.get("sample_weight")
        return original_fit(self, X, y, *args, **kwargs)

    monkeypatch.setattr(lgb.LGBMClassifier, "fit", spy_fit)

    X, y = _make_training_data(n=60)
    wrapper = LightGBMWrapper(config={"n_estimators": 5})
    wrapper.train(X, y, class_weights={0: 0.5, 1: 1.0, 2: 2.0})

    sw = captured["sample_weight"]
    assert sw is not None
    # Each sample's weight equals the weight of its class label.
    enc = y.map({"SELL": 0, "HOLD": 1, "BUY": 2}).to_numpy()
    expected = np.where(enc == 0, 0.5, np.where(enc == 1, 1.0, 2.0))
    np.testing.assert_allclose(sw, expected)


def test_lightgbm_no_class_weights_passes_none(monkeypatch):
    """Empty class_weights dict → sample_weight=None (LightGBM trains unweighted)."""
    import lightgbm as lgb

    captured: dict = {}
    original_fit = lgb.LGBMClassifier.fit

    def spy_fit(self, X, y, *args, **kwargs):
        captured["sample_weight"] = kwargs.get("sample_weight")
        return original_fit(self, X, y, *args, **kwargs)

    monkeypatch.setattr(lgb.LGBMClassifier, "fit", spy_fit)

    X, y = _make_training_data(n=60)
    wrapper = LightGBMWrapper(config={"n_estimators": 5})
    wrapper.train(X, y, class_weights={})
    assert captured["sample_weight"] is None


def test_lightgbm_num_class_cannot_be_overridden_by_config():
    """The pipeline always emits 3 classes; YAML must never break that contract."""
    wrapper = LightGBMWrapper(config={"num_class": 5, "n_estimators": 5})
    X, y = _make_training_data(n=30)
    wrapper.train(X, y, class_weights={0: 1.0, 1: 1.0, 2: 1.0})
    assert wrapper._model.get_params()["num_class"] == 3
