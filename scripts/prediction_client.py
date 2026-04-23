"""Internal HTTP client that triggers the prediction endpoint in the nightly batch."""
from __future__ import annotations

import logging
import os
from datetime import date

import httpx

logger = logging.getLogger(__name__)


class PredictionClient:
    """Calls POST /predict on the running FastAPI server."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000") -> None:
        self._base_url = base_url
        self._token = os.environ.get("API_TOKEN", "")

    def run(self, run_date: date) -> dict:
        """Trigger prediction for the full universe on run_date."""
        headers = {"Authorization": f"Bearer {self._token}"}
        payload = {"date": str(run_date)}
        with httpx.Client(base_url=self._base_url, timeout=120) as client:
            resp = client.post("/predict", json=payload, headers=headers)
            resp.raise_for_status()
            result = resp.json()
        logger.info(
            "prediction complete: %d signals on %s",
            len(result.get("predictions", [])), run_date,
        )
        return result
