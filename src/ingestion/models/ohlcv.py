from __future__ import annotations

from datetime import date

from pydantic import BaseModel, field_validator


class OHLCVRecord(BaseModel):
    ticker: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float | None = None

    @field_validator("ticker")
    @classmethod
    def ticker_non_empty_uppercase(cls, v: str) -> str:
        if not v:
            raise ValueError("ticker must be non-empty")
        return v.upper()

    @field_validator("date")
    @classmethod
    def date_not_future(cls, v: date) -> date:
        if v > date.today():
            raise ValueError(f"date {v} is in the future")
        return v

    @field_validator("open")
    @classmethod
    def open_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"open must be > 0, got {v}")
        return v

    @field_validator("high")
    @classmethod
    def high_valid(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"high must be > 0, got {v}")
        return v

    @field_validator("low")
    @classmethod
    def low_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"low must be > 0, got {v}")
        return v

    @field_validator("volume")
    @classmethod
    def volume_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"volume must be >= 0, got {v}")
        return v

    @field_validator("vwap")
    @classmethod
    def vwap_positive_if_present(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError(f"vwap must be > 0 if present, got {v}")
        return v
