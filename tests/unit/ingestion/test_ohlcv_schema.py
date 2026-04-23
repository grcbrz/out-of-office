from __future__ import annotations

from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from src.ingestion.models.ohlcv import OHLCVRecord


def _valid() -> dict:
    return {
        "ticker": "AAPL",
        "date": date(2024, 1, 2),
        "open": 100.0,
        "high": 110.0,
        "low": 90.0,
        "close": 105.0,
        "volume": 1_000_000,
        "vwap": 103.5,
    }


def test_valid_record_passes():
    rec = OHLCVRecord(**_valid())
    assert rec.ticker == "AAPL"


def test_ticker_uppercased():
    rec = OHLCVRecord(**{**_valid(), "ticker": "aapl"})
    assert rec.ticker == "AAPL"


def test_empty_ticker_rejected():
    with pytest.raises(ValidationError):
        OHLCVRecord(**{**_valid(), "ticker": ""})


def test_future_date_rejected():
    with pytest.raises(ValidationError):
        OHLCVRecord(**{**_valid(), "date": date.today() + timedelta(days=1)})


def test_open_zero_rejected():
    with pytest.raises(ValidationError):
        OHLCVRecord(**{**_valid(), "open": 0.0})


def test_open_negative_rejected():
    with pytest.raises(ValidationError):
        OHLCVRecord(**{**_valid(), "open": -1.0})


def test_volume_negative_rejected():
    with pytest.raises(ValidationError):
        OHLCVRecord(**{**_valid(), "volume": -1})


def test_volume_zero_accepted():
    rec = OHLCVRecord(**{**_valid(), "volume": 0})
    assert rec.volume == 0


def test_vwap_none_accepted():
    rec = OHLCVRecord(**{**_valid(), "vwap": None})
    assert rec.vwap is None


def test_vwap_zero_rejected():
    with pytest.raises(ValidationError):
        OHLCVRecord(**{**_valid(), "vwap": 0.0})


def test_vwap_negative_rejected():
    with pytest.raises(ValidationError):
        OHLCVRecord(**{**_valid(), "vwap": -5.0})
