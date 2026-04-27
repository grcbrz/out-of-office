from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from src.evaluation.explainability.attention import AttentionExtractor


def _mock_transformer(n_heads=4, seq_len=10):
    model = MagicMock()
    weights = np.random.rand(n_heads, seq_len, seq_len)
    model.get_attention_weights.return_value = weights
    return model


def test_attention_null_for_nhits():
    model = MagicMock()
    extractor = AttentionExtractor(model, "nhits")
    result = extractor.extract(None)
    assert result is None
    model.get_attention_weights.assert_not_called()


def test_attention_extraction_transformer():
    model = _mock_transformer(n_heads=4, seq_len=10)
    extractor = AttentionExtractor(model, "autoformer")
    result = extractor.extract(None)
    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 10  # seq_len


def test_attention_averaging_across_heads():
    model = _mock_transformer(n_heads=2, seq_len=5)
    extractor = AttentionExtractor(model, "autoformer")
    result = extractor.extract(None)
    assert len(result) == 5
