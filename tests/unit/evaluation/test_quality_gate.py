from __future__ import annotations

import logging

import pytest

from src.evaluation.quality_gate import EvaluationQualityGateError, QualityGate


def _gate(min_delta: float = 0.02) -> QualityGate:
    return QualityGate(
        f1_macro_min=0.35,
        mcc_min=0.05,
        hit_rate_min=0.50,
        max_signal_concentration=0.80,
        min_delta_over_baseline=min_delta,
    )


def _passing_metrics(**overrides):
    base = {
        "f1_macro": 0.40, "mcc": 0.10,
        "hit_rate": 0.55,
        "signal_distribution": {"BUY": 0.30, "HOLD": 0.40, "SELL": 0.30},
        "baseline_f1_macro": 0.35,
    }
    base.update(overrides)
    return base


def test_quality_gate_passes_with_baseline():
    _gate().check(_passing_metrics())


def test_quality_gate_fails_f1():
    with pytest.raises(EvaluationQualityGateError, match="f1_macro"):
        _gate().check(_passing_metrics(f1_macro=0.30))


def test_quality_gate_fails_mcc():
    with pytest.raises(EvaluationQualityGateError, match="mcc"):
        _gate().check(_passing_metrics(mcc=0.02))


def test_quality_gate_fails_concentration():
    with pytest.raises(EvaluationQualityGateError, match="signal concentration"):
        _gate().check(_passing_metrics(
            signal_distribution={"BUY": 0.05, "HOLD": 0.90, "SELL": 0.05}
        ))


# ----- baseline delta gate -----

def test_quality_gate_fails_baseline_delta():
    """Model F1 above absolute floor but only barely above baseline → fails delta gate."""
    with pytest.raises(EvaluationQualityGateError, match="baseline_f1_macro"):
        _gate().check(_passing_metrics(f1_macro=0.40, baseline_f1_macro=0.39))


def test_quality_gate_fails_when_model_below_baseline():
    """Model worse than baseline → must fail."""
    with pytest.raises(EvaluationQualityGateError, match="baseline_f1_macro"):
        _gate().check(_passing_metrics(f1_macro=0.40, baseline_f1_macro=0.45))


def test_quality_gate_passes_when_baseline_missing(caplog):
    metrics = _passing_metrics()
    metrics.pop("baseline_f1_macro")
    with caplog.at_level(logging.WARNING):
        _gate().check(metrics)
    assert any("baseline_f1_macro absent" in r.message for r in caplog.records)


def test_quality_gate_min_delta_configurable():
    """Tightening the delta should turn a previously-passing run into a failure."""
    metrics = _passing_metrics(f1_macro=0.40, baseline_f1_macro=0.36)
    _gate(min_delta=0.02).check(metrics)  # passes
    with pytest.raises(EvaluationQualityGateError, match="baseline_f1_macro"):
        _gate(min_delta=0.05).check(metrics)
