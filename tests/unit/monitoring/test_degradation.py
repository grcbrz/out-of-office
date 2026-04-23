from __future__ import annotations

import pandas as pd

from src.monitoring.degradation import DegradationDetector, compute_hit_rate


def _make_predictions(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["ticker", "signal", "run_date"])


def _make_ohlcv(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["ticker", "date", "close"])


def test_hit_rate_computation():
    preds = _make_predictions([
        ("AAPL", "BUY", "2026-01-02"),
        ("MSFT", "SELL", "2026-01-02"),
    ])
    ohlcv = _make_ohlcv([
        ("AAPL", "2026-01-02", 100.0),
        ("AAPL", "2026-01-03", 105.0),  # up → BUY correct
        ("MSFT", "2026-01-02", 200.0),
        ("MSFT", "2026-01-03", 195.0),  # down → SELL correct
    ])
    hit_rate = compute_hit_rate(preds, ohlcv)
    assert hit_rate == 1.0


def test_hit_rate_wrong_signals():
    preds = _make_predictions([
        ("AAPL", "BUY", "2026-01-02"),
        ("MSFT", "SELL", "2026-01-02"),
    ])
    ohlcv = _make_ohlcv([
        ("AAPL", "2026-01-02", 100.0),
        ("AAPL", "2026-01-03", 95.0),   # down → BUY wrong
        ("MSFT", "2026-01-02", 200.0),
        ("MSFT", "2026-01-03", 210.0),  # up → SELL wrong
    ])
    hit_rate = compute_hit_rate(preds, ohlcv)
    assert hit_rate == 0.0


def test_hit_rate_excludes_hold():
    preds = _make_predictions([
        ("AAPL", "HOLD", "2026-01-02"),
        ("MSFT", "BUY", "2026-01-02"),
    ])
    ohlcv = _make_ohlcv([
        ("AAPL", "2026-01-02", 100.0),
        ("AAPL", "2026-01-03", 110.0),
        ("MSFT", "2026-01-02", 200.0),
        ("MSFT", "2026-01-03", 205.0),  # BUY correct
    ])
    hit_rate = compute_hit_rate(preds, ohlcv)
    assert hit_rate == 1.0  # only MSFT counts


def test_hit_rate_none_when_only_hold():
    preds = _make_predictions([("AAPL", "HOLD", "2026-01-02")])
    ohlcv = _make_ohlcv([
        ("AAPL", "2026-01-02", 100.0),
        ("AAPL", "2026-01-03", 110.0),
    ])
    hit_rate = compute_hit_rate(preds, ohlcv)
    assert hit_rate is None


def test_degradation_single_window_no_trigger():
    preds = _make_predictions([("AAPL", "BUY", "2026-01-02")])
    ohlcv = _make_ohlcv([
        ("AAPL", "2026-01-02", 100.0),
        ("AAPL", "2026-01-03", 90.0),  # wrong → hit_rate = 0.0 < threshold
    ])
    detector = DegradationDetector(hit_rate_threshold=0.45, consecutive_windows_required=2)
    result = detector.run(preds, ohlcv, previous_consecutive_windows=0)
    assert not result.triggered
    assert result.consecutive_windows_below == 1


def test_degradation_two_windows_triggers():
    preds = _make_predictions([("AAPL", "BUY", "2026-01-02")])
    ohlcv = _make_ohlcv([
        ("AAPL", "2026-01-02", 100.0),
        ("AAPL", "2026-01-03", 90.0),  # wrong
    ])
    detector = DegradationDetector(hit_rate_threshold=0.45, consecutive_windows_required=2)
    result = detector.run(preds, ohlcv, previous_consecutive_windows=1)
    assert result.triggered
    assert result.consecutive_windows_below == 2


def test_degradation_resets_on_good_window():
    preds = _make_predictions([("AAPL", "BUY", "2026-01-02")])
    ohlcv = _make_ohlcv([
        ("AAPL", "2026-01-02", 100.0),
        ("AAPL", "2026-01-03", 110.0),  # correct → hit_rate = 1.0
    ])
    detector = DegradationDetector(hit_rate_threshold=0.45, consecutive_windows_required=2)
    result = detector.run(preds, ohlcv, previous_consecutive_windows=1)
    assert not result.triggered
    assert result.consecutive_windows_below == 0
