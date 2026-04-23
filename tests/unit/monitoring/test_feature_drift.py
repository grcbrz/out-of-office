from __future__ import annotations

import numpy as np
import pytest

from src.monitoring.drift.feature_drift import (
    FeatureDriftDetector,
    compute_psi,
    detect_feature_drift,
)

_KS_THRESH = 0.05
_PSI_WARN = 0.10
_PSI_ALERT = 0.20


def _kwargs(**overrides):
    return {
        "ks_pvalue_threshold": _KS_THRESH,
        "psi_warning_threshold": _PSI_WARN,
        "psi_alert_threshold": _PSI_ALERT,
        **overrides,
    }


def test_ks_test_detects_drift():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 500)
    current = rng.normal(5, 1, 500)  # large shift
    result = detect_feature_drift("f", reference, current, **_kwargs())
    assert result.ks_pvalue < _KS_THRESH


def test_ks_test_no_drift():
    rng = np.random.default_rng(1)
    reference = rng.normal(0, 1, 500)
    current = rng.normal(0, 1, 500)
    result = detect_feature_drift("f", reference, current, **_kwargs())
    assert result.ks_pvalue > _KS_THRESH


def test_psi_no_drift():
    rng = np.random.default_rng(2)
    data = rng.normal(0, 1, 1000)
    psi = compute_psi(data, data)
    assert psi < _PSI_WARN


def test_psi_moderate_drift():
    rng = np.random.default_rng(3)
    reference = rng.normal(0, 1, 1000)
    current = rng.normal(0.2, 1, 1000)
    psi = compute_psi(reference, current)
    # PSI is in warning range — between no-drift and significant-drift thresholds
    assert _PSI_WARN <= psi < _PSI_ALERT


def test_psi_significant_drift():
    rng = np.random.default_rng(4)
    reference = rng.normal(0, 1, 1000)
    current = rng.normal(4, 1, 1000)
    psi = compute_psi(reference, current)
    assert psi >= _PSI_ALERT


def test_feature_drift_gate_requires_both():
    # KS detects drift but PSI is small → no trigger
    rng = np.random.default_rng(5)
    reference = rng.normal(0, 1, 1000)
    # slight shift: KS p < 0.05 on large sample but PSI < 0.20
    current = rng.normal(0.3, 1, 1000)
    result = detect_feature_drift("f", reference, current, **_kwargs())
    # Even if ks_pvalue < threshold, psi must also be >= alert threshold
    if result.ks_pvalue < _KS_THRESH:
        if result.psi < _PSI_ALERT:
            assert not result.triggered
    # If neither condition met, also not triggered
    if result.psi >= _PSI_ALERT and result.ks_pvalue < _KS_THRESH:
        assert result.triggered


def test_detector_runs_all_features():
    rng = np.random.default_rng(6)
    ref = {"a": rng.normal(0, 1, 200), "b": rng.normal(0, 1, 200)}
    cur = {"a": rng.normal(0, 1, 200), "b": rng.normal(10, 1, 200)}
    detector = FeatureDriftDetector(**_kwargs())
    results = detector.run(ref, cur)
    assert len(results) == 2
    features = {r.feature for r in results}
    assert features == {"a", "b"}


def test_detector_skips_missing_features():
    rng = np.random.default_rng(7)
    ref = {"a": rng.normal(0, 1, 200), "b": rng.normal(0, 1, 200)}
    cur = {"a": rng.normal(0, 1, 200)}  # "b" missing from current
    detector = FeatureDriftDetector(**_kwargs())
    results = detector.run(ref, cur)
    assert len(results) == 1
    assert results[0].feature == "a"
