from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.architectures.baseline import (
    BaselineLastDirectionWrapper,
    MODEL_NAME,
)


def _train_df(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"log_return_lag1": values})


def _val_df(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"log_return_lag1": values})


def test_train_requires_log_return_lag1():
    wrapper = BaselineLastDirectionWrapper()
    with pytest.raises(ValueError, match="log_return_lag1"):
        wrapper.train(
            pd.DataFrame({"other_column": [0.0, 0.1]}),
            pd.Series(["HOLD", "HOLD"]),
            class_weights={},
        )


def test_train_rejects_all_null():
    wrapper = BaselineLastDirectionWrapper()
    with pytest.raises(ValueError, match="only nulls"):
        wrapper.train(
            pd.DataFrame({"log_return_lag1": [np.nan, np.nan]}),
            pd.Series(["HOLD", "HOLD"]),
            class_weights={},
        )


def test_predict_emits_three_classes_using_train_thresholds():
    """p30/p70 are computed on train; predictions on val use those thresholds."""
    train_values = list(np.linspace(-0.05, 0.05, 100))  # p30 ≈ -0.020, p70 ≈ 0.020
    wrapper = BaselineLastDirectionWrapper()
    wrapper.train(
        _train_df(train_values),
        pd.Series(["HOLD"] * len(train_values)),
        class_weights={},
    )
    val = _val_df([-0.04, -0.001, 0.04])  # below p30, between, above p70
    preds = wrapper.predict(val)
    assert preds.tolist() == [0, 1, 2]  # SELL, HOLD, BUY


def test_predict_proba_is_one_hot():
    wrapper = BaselineLastDirectionWrapper()
    wrapper.train(
        _train_df([-0.02, 0.0, 0.02]),
        pd.Series(["HOLD", "HOLD", "HOLD"]),
        class_weights={},
    )
    proba = wrapper.predict_proba(_val_df([-0.05, 0.0, 0.05]))
    assert proba.shape == (3, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert ((proba == 0.0) | (proba == 1.0)).all()


def test_predict_raises_before_training():
    wrapper = BaselineLastDirectionWrapper()
    with pytest.raises(RuntimeError, match="not trained"):
        wrapper.predict(_val_df([0.0]))


def test_save_load_roundtrip(tmp_path):
    wrapper = BaselineLastDirectionWrapper()
    wrapper.train(
        _train_df([-0.02, 0.0, 0.02]),
        pd.Series(["HOLD", "HOLD", "HOLD"]),
        class_weights={},
    )
    expected = wrapper.predict(_val_df([-0.05, 0.0, 0.05]))

    artifact = tmp_path / "baseline"
    wrapper.save(artifact)
    assert (artifact / "model.pkl").exists()

    loaded = BaselineLastDirectionWrapper.load(artifact, config={})
    actual = loaded.predict(_val_df([-0.05, 0.0, 0.05]))
    np.testing.assert_array_equal(expected, actual)


def test_model_name_constant():
    assert MODEL_NAME == "baseline_last_direction"
