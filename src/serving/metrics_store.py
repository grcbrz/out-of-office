from __future__ import annotations

import threading
from datetime import datetime, timezone


class MetricsStore:
    """In-memory counters for /metrics endpoint. Reset on server restart."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total = 0
        self._by_signal: dict[str, int] = {"BUY": 0, "HOLD": 0, "SELL": 0}
        self._tickers: set[str] = set()
        self._last_at: datetime | None = None

    def record(self, ticker: str, signal: str) -> None:
        with self._lock:
            self._total += 1
            self._by_signal[signal] = self._by_signal.get(signal, 0) + 1
            self._tickers.add(ticker)
            self._last_at = datetime.now(timezone.utc)

    def snapshot(self) -> dict:
        with self._lock:
            total = self._total
            dist = {k: round(v / total, 3) if total > 0 else 0.0 for k, v in self._by_signal.items()}
            return {
                "total_predictions": total,
                "signal_distribution": dist,
                "tickers_served": len(self._tickers),
                "last_prediction_at": self._last_at.isoformat() if self._last_at else None,
            }
