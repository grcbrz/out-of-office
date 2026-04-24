from __future__ import annotations

import datetime as dt
import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException

from src.serving.auth import require_auth, validate_token_at_startup
from src.serving.explainer import ServingExplainer
from src.serving.inference import InferenceEngine
from src.serving.loader import ArtifactLoader
from src.serving.metrics_store import MetricsStore
from src.serving.persistence import append_prediction_csv
from src.serving.schemas import PredictRequest, PredictResponse, PredictionItem, PredictionRecord

logger = logging.getLogger(__name__)

_loader = ArtifactLoader()
_metrics = MetricsStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_token_at_startup()
    _loader.load()
    yield


app = FastAPI(title="OOO API", lifespan=lifespan)


@app.get("/health")
def health(_: str = Depends(require_auth)):
    if not _loader.is_loaded:
        raise HTTPException(status_code=503, detail={"status": "degraded", "reason": "no model loaded"})
    return {
        "status": "ok",
        "model": _loader.model_name,
        "model_date": _loader.metadata.get("fold_end_date"),
        "production_fold_f1_macro": _loader.metadata.get("f1_macro"),
        "quality_gate_passed": _loader.metadata.get("quality_gate_passed", True),
    }


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

    engine = InferenceEngine(
        model=_loader.model,
        imputation_params=_loader.imputation_params,
        ticker_map=_loader.ticker_map,
        trained_features=_loader.trained_features,
    )
    explainer = ServingExplainer(model=_loader.model, trained_features=_loader.trained_features)

    predictions: list[PredictionItem] = []
    warnings: list[str] = []

    from src.features.schema import FEATURE_COLUMNS

    for ticker in tickers:
        # Stub: in production, load actual feature row from data/features/{ticker}/{run_date}.csv
        feature_row = pd.Series({c: 0.0 for c in FEATURE_COLUMNS if c != "ticker_id"})

        try:
            signal, confidence = engine.predict(ticker, feature_row)
        except Exception as e:
            logger.warning("inference failed for %s: %s", ticker, e)
            signal, confidence = "HOLD", 0.33
            warnings.append(f"{ticker}: inference error — {e}")

        explanation = explainer.explain(feature_row)

        item = PredictionItem(
            ticker=ticker, signal=signal, confidence=confidence, explanation=explanation
        )
        predictions.append(item)
        _metrics.record(ticker, signal)

        record = PredictionRecord(
            run_date=run_date, ticker=ticker, signal=signal,
            confidence=confidence, model=_loader.model_name or "unknown",
            explainer_used=explanation.get("explainer_used", "none"), predicted_at=dt.datetime.now(dt.timezone.utc),
        )
        append_prediction_csv(record)

    return PredictResponse(run_date=run_date, model=_loader.model_name or "unknown",
                           predictions=predictions, warnings=warnings)
