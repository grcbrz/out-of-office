from __future__ import annotations

"""Assert that no real network calls escape during ingestion tests."""



def test_httpx_not_imported_without_mock(monkeypatch):
    """Ensure PolygonClient and FinnhubClient cannot make real HTTP calls
    unless the underlying transport is patched. We verify by patching httpx.Client.get
    and asserting it raises if called unmocked.
    """
    import httpx
    import pytest

    calls: list = []

    def intercepting_get(self, url, **kwargs):
        calls.append(url)
        raise RuntimeError(f"Real HTTP call attempted: {url}")

    monkeypatch.setattr(httpx.Client, "get", intercepting_get)

    from src.ingestion.rate_limiter import RateLimiter
    from src.ingestion.clients.polygon import PolygonClient
    from datetime import date
    from unittest.mock import MagicMock

    limiter = MagicMock(spec=RateLimiter)
    client = PolygonClient("key", limiter)

    # tenacity wraps the intercepted error in RetryError after exhausting attempts
    import tenacity
    with pytest.raises((RuntimeError, tenacity.RetryError)):
        client._fetch_grouped_daily(date(2024, 1, 2))

    assert any("aggs/grouped" in url for url in calls), "No intercept recorded"
