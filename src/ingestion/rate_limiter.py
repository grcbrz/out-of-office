from __future__ import annotations

import threading
import time


class RateLimiter:
    """Token bucket that enforces a maximum calls-per-minute rate.

    Call acquire() before every outbound API request. It blocks until
    a token is available. All time.sleep() usage is confined here.
    """

    def __init__(self, calls_per_minute: int) -> None:
        self._interval = 60.0 / calls_per_minute
        self._lock = threading.Lock()
        self._last_call_time: float = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call_time
            wait = self._interval - elapsed
            if wait > 0:
                time.sleep(wait)
            self._last_call_time = time.monotonic()
