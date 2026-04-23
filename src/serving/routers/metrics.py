from __future__ import annotations

from fastapi import APIRouter, Depends

from src.serving.auth import require_auth
from src.serving.metrics_store import MetricsStore

router = APIRouter()
