from __future__ import annotations

import datetime as dt
import logging
from contextlib import asynccontextmanager
from pathlib import Path

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
_FEATURES_DIR = Path("data/features")


def _load_feature_row(ticker: str, run_date: dt.date) -> pd.Series | None:
    """Return the most-recent feature row for ticker on or before run_date."""
    ticker_dir = _FEATURES_DIR / ticker
    if not ticker_dir.exists():
        return None
    candidates = sorted(ticker_dir.glob("*.csv"))
    target = str(run_date)
    files = [f for f in candidates if f.stem <= target]
    if not files:
        return None
    df = pd.read_csv(files[-1])
    if df.empty:
        return None
    return df.iloc[-1]


@asynccontextmanager
async def lifespan(app: FastAPI):
    from dotenv import load_dotenv
    load_dotenv()
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


@app.post("/reload")
def reload(_: str = Depends(require_auth)):
    _loader.load()
    return {"status": "reloaded", "model": _loader.model_name, "is_loaded": _loader.is_loaded}


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

    for ticker in tickers:
        feature_row = _load_feature_row(ticker, run_date)
        if feature_row is None:
            warnings.append(f"{ticker}: no feature data for {run_date} — skipping")
            continue

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

        top = explanation.get("top_features", [])
        record = PredictionRecord(
            run_date=run_date, ticker=ticker, signal=signal,
            confidence=confidence, model=_loader.model_name or "unknown",
            explainer_used=explanation.get("explainer_used", "none"),
            predicted_at=dt.datetime.now(dt.timezone.utc),
            sentiment_available=bool(feature_row.get("sentiment_available", False)),
            top_feature_1=top[0]["feature"] if len(top) > 0 else None,
            top_feature_2=top[1]["feature"] if len(top) > 1 else None,
            top_feature_3=top[2]["feature"] if len(top) > 2 else None,
            top_feature_4=top[3]["feature"] if len(top) > 3 else None,
            top_feature_5=top[4]["feature"] if len(top) > 4 else None,
            shap_1=top[0]["shap_value"] if len(top) > 0 else None,
            shap_2=top[1]["shap_value"] if len(top) > 1 else None,
            shap_3=top[2]["shap_value"] if len(top) > 2 else None,
            shap_4=top[3]["shap_value"] if len(top) > 3 else None,
            shap_5=top[4]["shap_value"] if len(top) > 4 else None,
        )
        append_prediction_csv(record)

    return PredictResponse(run_date=run_date, model=_loader.model_name or "unknown",
                           predictions=predictions, warnings=warnings)
