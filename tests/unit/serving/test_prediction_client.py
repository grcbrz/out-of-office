from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from scripts.prediction_client import PredictionClient


def _mock_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def test_run_sends_predict_date_key(monkeypatch):
    """Payload must use predict_date, not date — matches PredictRequest schema."""
    monkeypatch.setenv("API_TOKEN", "test-token-32-chars-xxxxxxxxxxxx")
    captured: list[dict] = []

    def fake_post(url, json, headers):
        captured.append(json)
        return _mock_response({"predictions": [], "run_date": "2026-04-24", "model": "autoformer", "warnings": []})

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__ = lambda s: s
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = fake_post
        mock_client_cls.return_value = mock_client

        client = PredictionClient()
        client.run(date(2026, 4, 24))

    assert captured, "POST was never called"
    assert "predict_date" in captured[0], "payload must use 'predict_date', not 'date'"
    assert "date" not in captured[0], "unexpected 'date' key — use 'predict_date'"
    assert captured[0]["predict_date"] == "2026-04-24"


def test_run_includes_bearer_token(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "my-secret-token-32-chars-xxxxxxx")
    captured_headers: list[dict] = []

    def fake_post(url, json, headers):
        captured_headers.append(headers)
        return _mock_response({"predictions": [], "run_date": "2026-04-24", "model": "autoformer", "warnings": []})

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__ = lambda s: s
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = fake_post
        mock_client_cls.return_value = mock_client

        PredictionClient().run(date(2026, 4, 24))

    assert captured_headers[0]["Authorization"] == "Bearer my-secret-token-32-chars-xxxxxxx"


def test_run_returns_parsed_response(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "test-token-32-chars-xxxxxxxxxxxx")
    expected = {
        "predictions": [{"ticker": "AAPL", "signal": "BUY", "confidence": 0.7}],
        "run_date": "2026-04-24",
        "model": "autoformer",
        "warnings": [],
    }

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__ = lambda s: s
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_response(expected)
        mock_client_cls.return_value = mock_client

        result = PredictionClient().run(date(2026, 4, 24))

    assert result == expected
