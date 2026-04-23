from __future__ import annotations

import pytest

from src.evaluation.quality_gate import EvaluationQualityGateError, QualityGate


def _gate():
    return QualityGate(f1_macro_min=0.35, mcc_min=0.05, hit_rate_min=0.50, max_signal_concentration=0.80)


def test_quality_gate_passes():
    _gate().check({
        "f1_macro": 0.40, "mcc": 0.10,
        "hit_rate": 0.55,
        "signal_distribution": {"BUY": 0.30, "HOLD": 0.40, "SELL": 0.30},
    })


def test_quality_gate_fails_f1():
    with pytest.raises(EvaluationQualityGateError, match="f1_macro"):
        _gate().check({
            "f1_macro": 0.30, "mcc": 0.10,
            "hit_rate": 0.55,
            "signal_distribution": {},
        })


def test_quality_gate_fails_mcc():
    with pytest.raises(EvaluationQualityGateError, match="mcc"):
        _gate().check({
            "f1_macro": 0.40, "mcc": 0.02,
            "hit_rate": 0.55,
            "signal_distribution": {},
        })


def test_quality_gate_fails_concentration():
    with pytest.raises(EvaluationQualityGateError, match="signal concentration"):
        _gate().check({
            "f1_macro": 0.40, "mcc": 0.10,
            "hit_rate": 0.55,
            "signal_distribution": {"BUY": 0.05, "HOLD": 0.90, "SELL": 0.05},
        })
