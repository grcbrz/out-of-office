from __future__ import annotations

import datetime as dt
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request

from src.serving.auth import require_auth, validate_token_at_startup
from src.serving.loader import ArtifactLoader
from src.serving.metrics_store import MetricsStore
from src.serving.persistence import append_prediction_csv
from src.serving.routers.health import get_health_response
from src.serving.schemas import PredictRequest, PredictResponse, PredictionItem, PredictionRecord

logger = logging.getLogger(__name__)

_loader = ArtifactLoader()
_metrics = MetricsStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_token_at_startup()
    _loader.load()
    yield


app = FastAPI(title="Stock Recommender API", lifespan=lifespan)


@app.get("/health")
def health(_: str = Depends(require_auth)):
    return get_health_response(_loader)


@app.get("/metrics")
def metrics(_: str = Depends(require_auth)):
    snap = _metrics.snapshot()
    snap["model"] = _loader.model_name
    return snap


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, _: str = Depends(require_auth)):
    if not _loader.is_loaded:
        raise HTTPException(status_code=503, detail="no production model loaded")

    run_date = request.predict_date or dt.date.today()
    tickers = request.tickers or list(_loader.ticker_map.keys())

    unknown = [t for t in tickers if t not in _loader.ticker_map]
    if unknown:
        raise HTTPException(status_code=400, detail=f"unknown tickers: {unknown}")

    predictions: list[PredictionItem] = []
    warnings: list[str] = []

    for ticker in tickers:
        signal, confidence = "HOLD", 0.33
        explanation = {"top_features": [], "attention_weights": None, "explainer_used": "none"}

        item = PredictionItem(
            ticker=ticker, signal=signal, confidence=confidence, explanation=explanation
        )
        predictions.append(item)
        _metrics.record(ticker, signal)

        record = PredictionRecord(
            run_date=run_date, ticker=ticker, signal=signal,
            confidence=confidence, model=_loader.model_name or "unknown",
            explainer_used="none", predicted_at=dt.datetime.now(dt.timezone.utc),
        )
        append_prediction_csv(record)

    return PredictResponse(run_date=run_date, model=_loader.model_name or "unknown",
                           predictions=predictions, warnings=warnings)
