from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.serving.loader import ArtifactLoader

router = APIRouter()


def get_health_response(loader: ArtifactLoader) -> dict:
    if not loader.is_loaded:
        raise HTTPException(status_code=503, detail={"status": "degraded", "reason": "no model loaded"})
    return {
        "status": "ok",
        "model": loader.model_name,
        "model_date": loader.metadata.get("fold_end_date"),
        "production_fold_f1_macro": loader.metadata.get("f1_macro"),
        "quality_gate_passed": loader.metadata.get("quality_gate_passed", True),
    }
