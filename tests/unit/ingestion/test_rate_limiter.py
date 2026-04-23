from __future__ import annotations

import time

from src.ingestion.rate_limiter import RateLimiter


def test_rate_limiter_enforces_spacing():
    """Two consecutive acquire() calls on a 60/min limiter should be ~1s apart."""
    limiter = RateLimiter(calls_per_minute=60)
    limiter.acquire()  # prime the bucket
    t0 = time.monotonic()
    limiter.acquire()
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.9, f"expected >=0.9s gap, got {elapsed:.3f}s"


def test_rate_limiter_high_rate_does_not_block_long():
    """A 3600/min limiter (1 call per 16ms) should not block more than 100ms."""
    limiter = RateLimiter(calls_per_minute=3600)
    limiter.acquire()
    t0 = time.monotonic()
    limiter.acquire()
    elapsed = time.monotonic() - t0
    assert elapsed < 0.1, f"expected <0.1s, got {elapsed:.3f}s"
