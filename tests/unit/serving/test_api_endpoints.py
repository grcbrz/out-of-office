from __future__ import annotations

import os
from datetime import date
from unittest.mock import patch

from fastapi.testclient import TestClient

_TOKEN = "a" * 32
_AUTH_HEADER = {"Authorization": f"Bearer {_TOKEN}"}


def _make_client():
    os.environ["API_TOKEN"] = _TOKEN
    from src.serving.app import app, _loader
    # Reset state
    _loader.is_loaded = False
    _loader.model_name = None
    _loader.ticker_map = {}
    return TestClient(app, raise_server_exceptions=False)


def test_predict_missing_token():
    client = _make_client()
    resp = client.post("/predict", json={"predict_date": str(date.today())})
    assert resp.status_code == 401  # no token at all — HTTPBearer returns 401


def test_predict_invalid_token():
    client = _make_client()
    resp = client.post("/predict", json={"predict_date": str(date.today())},
                       headers={"Authorization": "Bearer wrong_token"})
    assert resp.status_code == 401


def test_predict_no_artifact_returns_503():
    client = _make_client()
    resp = client.post("/predict", json={"predict_date": str(date.today())},
                       headers=_AUTH_HEADER)
    assert resp.status_code == 503


def test_health_degraded_when_no_artifact():
    client = _make_client()
    resp = client.get("/health", headers=_AUTH_HEADER)
    assert resp.status_code == 503


def test_predict_invalid_body_returns_422():
    client = _make_client()
    resp = client.post("/predict", json={"tickers": "not_a_list"},
                       headers=_AUTH_HEADER)
    assert resp.status_code == 422


def test_metrics_returns_json():
    client = _make_client()
    resp = client.get("/metrics", headers=_AUTH_HEADER)
    assert resp.status_code == 200
    data = resp.json()
    assert "total_predictions" in data


def test_predict_with_loaded_artifact(tmp_path):
    from src.serving.app import app, _loader
    os.environ["API_TOKEN"] = _TOKEN
    _loader.is_loaded = True
    _loader.model_name = "nhits"
    _loader.ticker_map = {"AAPL": 0, "MSFT": 1}
    _loader.metadata = {"f1_macro": 0.40}

    with patch("src.serving.app.append_prediction_csv"):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/predict",
            json={"tickers": ["AAPL"], "predict_date": str(date.today())},
            headers=_AUTH_HEADER,
        )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["predictions"]) == 1
    assert body["predictions"][0]["ticker"] == "AAPL"


def test_predict_unknown_ticker_400(tmp_path):
    from src.serving.app import app, _loader
    os.environ["API_TOKEN"] = _TOKEN
    _loader.is_loaded = True
    _loader.model_name = "nhits"
    _loader.ticker_map = {"AAPL": 0}
    _loader.metadata = {}

    with patch("src.serving.app.append_prediction_csv"):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/predict",
            json={"tickers": ["UNKNOWN_XYZ"], "predict_date": str(date.today())},
            headers=_AUTH_HEADER,
        )
    assert resp.status_code == 400
