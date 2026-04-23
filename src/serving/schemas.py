from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, field_validator


class PredictRequest(BaseModel):
    tickers: list[str] | None = None
    predict_date: dt.date | None = None  # None = today; named predict_date to avoid shadowing dt.date

    @field_validator("predict_date")
    @classmethod
    def date_not_future(cls, v: dt.date | None) -> dt.date | None:
        if v is not None and v > dt.date.today():
            raise ValueError(f"predict_date {v} is in the future")
        return v

    @field_validator("tickers")
    @classmethod
    def tickers_non_empty(cls, v: list[str] | None) -> list[str] | None:
        if v is not None and len(v) == 0:
            raise ValueError("tickers list must not be empty")
        return v


class FeatureExplanation(BaseModel):
    feature: str
    shap_value: float


class PredictionItem(BaseModel):
    ticker: str
    signal: str
    confidence: float
    explanation: dict[str, Any]


class PredictResponse(BaseModel):
    run_date: dt.date
    model: str
    predictions: list[PredictionItem]
    warnings: list[str] = []


class PredictionRecord(BaseModel):
    run_date: dt.date
    ticker: str
    signal: str
    confidence: float
    model: str
    top_feature_1: str | None = None
    top_feature_2: str | None = None
    top_feature_3: str | None = None
    top_feature_4: str | None = None
    top_feature_5: str | None = None
    shap_1: float | None = None
    shap_2: float | None = None
    shap_3: float | None = None
    shap_4: float | None = None
    shap_5: float | None = None
    explainer_used: str | None = None
    sentiment_available: bool = False
    predicted_at: dt.datetime
