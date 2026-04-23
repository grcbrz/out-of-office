from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.models.architectures.autoformer import AutoformerWrapper
from src.models.architectures.nhits import NHiTSWrapper
from src.models.architectures.patchtst import PatchTSTWrapper


def _make_df(n=10) -> pd.DataFrame:
    return pd.DataFrame({"feature_a": np.random.randn(n), "ticker_id": range(n)})


def _make_labels(n=10) -> pd.Series:
    return pd.Series(["BUY"] * n)


# ── NHiTS ──


def test_nhits_predict_raises_before_training():
    wrapper = NHiTSWrapper(config={})
    with pytest.raises(RuntimeError, match="not trained"):
        wrapper.predict(_make_df())


def test_nhits_train_with_mock_neuralforecast():
    mock_nf_instance = MagicMock()
    mock_nf_class = MagicMock(return_value=mock_nf_instance)
    mock_nhits_class = MagicMock()

    with patch.dict("sys.modules", {
        "neuralforecast": MagicMock(NeuralForecast=mock_nf_class),
        "neuralforecast.models": MagicMock(NHITS=mock_nhits_class),
    }):
        wrapper = NHiTSWrapper(config={"h": 1, "max_steps": 10})
        wrapper.train(_make_df(), _make_labels(), class_weights={})
        assert wrapper._model is not None


def test_nhits_predict_returns_array():
    wrapper = NHiTSWrapper(config={})
    wrapper._model = MagicMock()  # set model directly
    result = wrapper.predict(_make_df(5))
    assert len(result) == 5


def test_nhits_save_creates_dir(tmp_path):
    wrapper = NHiTSWrapper(config={})
    wrapper._model = None  # no model, but save should still create dir
    wrapper.save(tmp_path / "artifact")
    assert (tmp_path / "artifact").exists()


# ── PatchTST ──


def test_patchtst_predict_raises_before_training():
    wrapper = PatchTSTWrapper(config={})
    with pytest.raises(RuntimeError, match="not trained"):
        wrapper.predict(_make_df())


def test_patchtst_train_with_mock_neuralforecast():
    mock_nf_class = MagicMock(return_value=MagicMock())
    mock_patchtst_class = MagicMock()

    with patch.dict("sys.modules", {
        "neuralforecast": MagicMock(NeuralForecast=mock_nf_class),
        "neuralforecast.models": MagicMock(PatchTST=mock_patchtst_class),
    }):
        wrapper = PatchTSTWrapper(config={"h": 1})
        wrapper.train(_make_df(), _make_labels(), class_weights={})
        assert wrapper._model is not None


def test_patchtst_load_returns_wrapper(tmp_path):
    wrapper = PatchTSTWrapper.load(tmp_path, config={})
    assert isinstance(wrapper, PatchTSTWrapper)


# ── Autoformer ──


def test_autoformer_predict_raises_before_training():
    wrapper = AutoformerWrapper(config={})
    with pytest.raises(RuntimeError, match="not trained"):
        wrapper.predict(_make_df())


def test_autoformer_train_with_mock_neuralforecast():
    mock_nf_class = MagicMock(return_value=MagicMock())
    mock_autoformer_class = MagicMock()

    with patch.dict("sys.modules", {
        "neuralforecast": MagicMock(NeuralForecast=mock_nf_class),
        "neuralforecast.models": MagicMock(Autoformer=mock_autoformer_class),
    }):
        wrapper = AutoformerWrapper(config={"h": 1})
        wrapper.train(_make_df(), _make_labels(), class_weights={})
        assert wrapper._model is not None


def test_autoformer_load_returns_wrapper(tmp_path):
    wrapper = AutoformerWrapper.load(tmp_path, config={})
    assert isinstance(wrapper, AutoformerWrapper)
