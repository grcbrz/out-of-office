from __future__ import annotations

from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from src.serving.schemas import PredictRequest


def test_future_date_rejected():
    with pytest.raises(ValidationError):
        PredictRequest(predict_date=date.today() + timedelta(days=1))


def test_today_accepted():
    req = PredictRequest(predict_date=date.today())
    assert req.predict_date == date.today()


def test_empty_tickers_rejected():
    with pytest.raises(ValidationError):
        PredictRequest(tickers=[])


def test_none_tickers_accepted():
    req = PredictRequest(tickers=None)
    assert req.tickers is None
