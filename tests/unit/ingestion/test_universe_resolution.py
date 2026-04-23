from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.clients.polygon import PolygonClient
from src.ingestion.rate_limiter import RateLimiter


def _make_client() -> PolygonClient:
    limiter = MagicMock(spec=RateLimiter)
    return PolygonClient(api_key="test", rate_limiter=limiter)


def _make_results(n: int, base_volume: int = 1_000_000) -> list[dict]:
    return [
        {"T": f"T{i:03d}", "v": base_volume - i * 1000, "otc": None}
        for i in range(n)
    ]


def test_universe_top_50_by_volume():
    client = _make_client()
    results = _make_results(60)
    with patch.object(client, "_fetch_grouped_daily", return_value={"results": results}):
        universe = client.resolve_universe(date(2024, 1, 2))
    assert len(universe) == 50
    # First result has highest volume — must be first
    assert universe[0] == "T000"
    assert universe[49] == "T049"


def test_universe_less_than_50_warns(caplog):
    import logging
    client = _make_client()
    results = _make_results(30)
    with patch.object(client, "_fetch_grouped_daily", return_value={"results": results}):
        with caplog.at_level(logging.WARNING):
            universe = client.resolve_universe(date(2024, 1, 2))
    assert len(universe) == 30
    assert any("< 50" in r.message for r in caplog.records)


def test_universe_empty_results_returns_empty():
    client = _make_client()
    with patch.object(client, "_fetch_grouped_daily", return_value={"results": []}):
        universe = client.resolve_universe(date(2024, 1, 2))
    assert universe == []
