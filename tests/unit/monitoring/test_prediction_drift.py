from __future__ import annotations


from src.monitoring.drift.prediction_drift import PredictionDriftDetector, detect_prediction_drift

_CHI2 = 0.05
_CONC = 0.80


def _detect(**overrides):
    defaults = dict(chi2_pvalue_threshold=_CHI2, max_signal_concentration=_CONC)
    return detect_prediction_drift(**{**defaults, **overrides})


def test_prediction_drift_chi2():
    reference = {"BUY": 100, "HOLD": 100, "SELL": 100}
    current = {"BUY": 200, "HOLD": 10, "SELL": 10}  # large shift
    result = _detect(reference_counts=reference, current_counts=current)
    assert result.triggered
    assert result.chi2_pvalue < _CHI2


def test_prediction_no_drift():
    reference = {"BUY": 100, "HOLD": 100, "SELL": 100}
    current = {"BUY": 98, "HOLD": 102, "SELL": 100}
    result = _detect(reference_counts=reference, current_counts=current)
    assert not result.triggered


def test_degenerate_signal_flag():
    reference = {"BUY": 100, "HOLD": 100, "SELL": 100}
    current = {"BUY": 5, "HOLD": 85, "SELL": 10}  # HOLD = 85%
    result = _detect(reference_counts=reference, current_counts=current)
    assert result.degenerate_signal
    assert result.dominant_class == "HOLD"


def test_no_degenerate_when_balanced():
    reference = {"BUY": 100, "HOLD": 100, "SELL": 100}
    current = {"BUY": 30, "HOLD": 40, "SELL": 30}
    result = _detect(reference_counts=reference, current_counts=current)
    assert not result.degenerate_signal
    assert result.dominant_class is None


def test_empty_current_returns_no_trigger():
    result = _detect(reference_counts={"BUY": 100, "HOLD": 100, "SELL": 100}, current_counts={})
    assert not result.triggered
    assert not result.degenerate_signal


def test_detector_class_delegates():
    detector = PredictionDriftDetector(chi2_pvalue_threshold=_CHI2, max_signal_concentration=_CONC)
    ref = {"BUY": 100, "HOLD": 100, "SELL": 100}
    cur = {"BUY": 200, "HOLD": 5, "SELL": 5}
    result = detector.run(ref, cur)
    assert result.triggered
