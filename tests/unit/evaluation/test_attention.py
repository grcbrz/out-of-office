from __future__ import annotations

from unittest.mock import MagicMock

from src.evaluation.explainability.attention import (
    AttentionExtractor,
    _TRANSFORMER_MODELS,
)


def test_attention_returns_null_for_lightgbm():
    """v1 production candidate has no attention mechanism."""
    extractor = AttentionExtractor(MagicMock(), "lightgbm")
    assert extractor.extract(None) is None


def test_attention_returns_null_for_baseline():
    extractor = AttentionExtractor(MagicMock(), "baseline_last_direction")
    assert extractor.extract(None) is None


def test_attention_does_not_invoke_model_for_unregistered_name():
    """The extractor must not call get_attention_weights when the model name is
    not registered as a transformer — protects against accidental crashes when
    callers pass a non-transformer estimator that has no such method.
    """
    model = MagicMock()
    extractor = AttentionExtractor(model, "lightgbm")
    extractor.extract(None)
    model.get_attention_weights.assert_not_called()


def test_transformer_models_set_is_currently_empty():
    """No transformer candidate is in production. When one is added,
    update this expectation alongside the wiring."""
    assert _TRANSFORMER_MODELS == frozenset()
