from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.serving.auth import require_auth
from src.serving.schemas import PredictRequest, PredictResponse

router = APIRouter()
