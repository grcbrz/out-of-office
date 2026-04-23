from __future__ import annotations

from src.serving.metrics_store import MetricsStore


def test_counters_increment():
    store = MetricsStore()
    store.record("AAPL", "BUY")
    store.record("MSFT", "SELL")
    snap = store.snapshot()
    assert snap["total_predictions"] == 2
    assert snap["tickers_served"] == 2


def test_signal_distribution():
    store = MetricsStore()
    for _ in range(3):
        store.record("AAPL", "BUY")
    for _ in range(7):
        store.record("MSFT", "HOLD")
    snap = store.snapshot()
    assert snap["signal_distribution"]["BUY"] == pytest.approx(0.3)
    assert snap["signal_distribution"]["HOLD"] == pytest.approx(0.7)


import pytest
