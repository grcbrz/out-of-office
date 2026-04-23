from __future__ import annotations

from datetime import date

from pydantic import BaseModel, field_validator


class SentimentRecord(BaseModel):
    ticker: str
    date: date
    bullish_percent: float | None = None
    bearish_percent: float | None = None
    company_news_score: float | None = None
    buzz_weekly_average: float | None = None

    @field_validator("ticker")
    @classmethod
    def ticker_non_empty_uppercase(cls, v: str) -> str:
        if not v:
            raise ValueError("ticker must be non-empty")
        return v.upper()

    @field_validator("bullish_percent", "bearish_percent")
    @classmethod
    def percent_in_range(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(f"percent value must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("company_news_score", "buzz_weekly_average")
    @classmethod
    def score_non_negative(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError(f"score must be >= 0, got {v}")
        return v
