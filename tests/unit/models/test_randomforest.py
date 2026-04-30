"""RandomForest-specific behaviour.

The generic interface tests (predict-before-train raises, predict_proba sums
to 1, save/load roundtrip, argmax-of-proba == predict) live in
``test_architectures.py`` and run against every wrapper via parametrize.

This module covers the bits that are RF-specific:

  * ``class_weight`` is passed at *construction*, not at fit time
    (regression guard: passing it as ``sample_weight`` would interact badly
    with the RF bootstrap and double-up minority-class weights).
  * Stringified class-weight keys (from JSON round-trips) are coerced to int.
  * The project-wide ``random_seed`` config key maps to sklearn's
    ``random_state``.

Implementation note: assertions read the fitted model's ``get_params()``
rather than monkeypatching ``__init__``. sklearn's ``BaseEstimator`` rejects
constructor signatures containing ``*args`` (the standard spy pattern), so
introspecting the post-fit estimator is both simpler and a stronger
regression guard — it tests the *observable* state of the constructor
parameter, not the internal call shape.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.schema import FEATURE_COLUMNS
from src.models.architectures.randomforest import (
    MODEL_NAME,
    RandomForestWrapper,
)


def _make_training_data(n: int = 60, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    columns = {col: rng.normal(size=n) for col in FEATURE_COLUMNS}
    columns["ticker_id"] = rng.integers(0, 5, size=n)
    df = pd.DataFrame(columns)
    labels = np.tile(np.array(["SELL", "HOLD", "BUY"]), n // 3 + 1)[:n]
    rng.shuffle(labels)
    return df, pd.Series(labels)


# ---------- model name ----------

def test_model_name_constant():
    assert MODEL_NAME == "randomforest"


# ---------- class_weight wiring ----------

def test_class_weight_passed_to_constructor():
    """Regression guard: class_weight must be at __init__, not at fit time.

    If the wrapper ever regressed to passing ``sample_weight`` at fit instead
    of ``class_weight`` at construction, ``get_params()["class_weight"]``
    would be ``None`` and this assertion would fail. RandomForest's bootstrap
    interacts correctly with construction-time class_weight; passing weights
    at fit time would double-up minority-class influence.
    """
    X, y = _make_training_data(n=60)
    wrapper = RandomForestWrapper(config={"n_estimators": 5})
    wrapper.train(X, y, class_weights={0: 0.5, 1: 1.0, 2: 2.0})
    assert wrapper._model.get_params()["class_weight"] == {0: 0.5, 1: 1.0, 2: 2.0}


def test_no_class_weights_means_class_weight_is_none():
    """Empty dict → wrapper does not set the kwarg → sklearn default = None."""
    X, y = _make_training_data(n=60)
    wrapper = RandomForestWrapper(config={"n_estimators": 5})
    wrapper.train(X, y, class_weights={})
    assert wrapper._model.get_params()["class_weight"] is None


def test_string_class_weight_keys_coerced_to_int():
    """JSON round-trips can stringify dict keys. Wrapper must coerce."""
    X, y = _make_training_data(n=60)
    wrapper = RandomForestWrapper(config={"n_estimators": 5})
    wrapper.train(X, y, class_weights={"0": 0.5, "1": 1.0, "2": 2.0})
    assert wrapper._model.get_params()["class_weight"] == {0: 0.5, 1: 1.0, 2: 2.0}


# ---------- config key handling ----------

def test_random_seed_legacy_key_maps_to_random_state():
    """``random_seed`` is the project-wide config key (used by lightgbm.yaml,
    nhits.yaml historically, etc.). sklearn expects ``random_state`` — the
    wrapper must translate.
    """
    wrapper = RandomForestWrapper(config={"random_seed": 7, "n_estimators": 5})
    X, y = _make_training_data(n=30)
    wrapper.train(X, y, class_weights={0: 1.0, 1: 1.0, 2: 1.0})
    assert wrapper._model.get_params()["random_state"] == 7


def test_n_jobs_pinned_for_determinism():
    """Even without explicit override, the wrapper defaults n_jobs=1.

    The training pipeline forces n_jobs=1 again at instantiation; this test
    locks the default at the wrapper level so determinism is a property of
    the wrapper itself, not just the call site.
    """
    wrapper = RandomForestWrapper(config={"n_estimators": 5})
    X, y = _make_training_data(n=30)
    wrapper.train(X, y, class_weights={0: 1.0, 1: 1.0, 2: 1.0})
    assert wrapper._model.get_params()["n_jobs"] == 1
