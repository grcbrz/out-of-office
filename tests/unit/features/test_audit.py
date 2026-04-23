from __future__ import annotations

import pandas as pd
import pytest

from src.features.audit import LookaheadBiasError, lookahead_bias_guard, null_audit
from src.features.schema import FEATURE_COLUMNS


def test_lookahead_bias_guard_passes():
    """forward_return not in FEATURE_COLUMNS — guard must not raise."""
    df = pd.DataFrame({"forward_return": [0.01], "log_return": [0.01]})
    assert "forward_return" not in FEATURE_COLUMNS
    lookahead_bias_guard(df)  # must not raise


def test_lookahead_bias_guard_fails_if_in_feature_columns(monkeypatch):
    """Inject forward_return into FEATURE_COLUMNS → LookaheadBiasError raised."""
    import src.features.audit as audit_module
    monkeypatch.setattr(audit_module, "FEATURE_COLUMNS", FEATURE_COLUMNS + ["forward_return"])
    df = pd.DataFrame({"forward_return": [0.01]})
    with pytest.raises(LookaheadBiasError):
        audit_module.lookahead_bias_guard(df)


def test_null_audit_warns_on_core_feature_nulls(caplog):
    import logging
    df = pd.DataFrame({"log_return": [None, 0.01]})
    with caplog.at_level(logging.WARNING):
        null_audit(df, "AAPL")
    assert any("log_return" in r.message for r in caplog.records)


def test_null_audit_accepts_sentiment_nulls_silently(caplog):
    import logging
    df = pd.DataFrame({"bullish_percent": [None, None]})
    with caplog.at_level(logging.WARNING):
        null_audit(df, "AAPL")
    # No WARNING should be emitted for nullable sentiment columns
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert all("bullish_percent" not in r.message for r in warnings)
