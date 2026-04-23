from __future__ import annotations

from datetime import date

from pydantic import BaseModel, field_validator


class ProcessedRecord(BaseModel):
    ticker: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float | None = None
    close_zscore: float | None = None
    volume_zscore: float | None = None
    close_outlier_flag: bool = False
    volume_outlier_flag: bool = False
    bullish_percent: float | None = None
    bearish_percent: float | None = None
    company_news_score: float | None = None
    buzz_weekly_average: float | None = None
    sentiment_available: bool = False
    is_trading_day: bool = True
    imputed_close: bool = False
    imputed_volume: bool = False
