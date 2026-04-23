from __future__ import annotations

import csv
import datetime as dt


from src.serving.persistence import append_prediction_csv
from src.serving.schemas import PredictionRecord


def _make_record(ticker="AAPL") -> PredictionRecord:
    return PredictionRecord(
        run_date=dt.date(2026, 1, 2),
        ticker=ticker,
        signal="BUY",
        confidence=0.7,
        model="nhits",
        explainer_used="shap",
        predicted_at=dt.datetime(2026, 1, 2, 20, 0, tzinfo=dt.timezone.utc),
    )


def test_append_creates_csv(tmp_path):
    rec = _make_record()
    append_prediction_csv(rec, predictions_dir=tmp_path)
    dest = tmp_path / "2026-01-02.csv"
    assert dest.exists()


def test_append_writes_header_once(tmp_path):
    for ticker in ("AAPL", "MSFT"):
        append_prediction_csv(_make_record(ticker), predictions_dir=tmp_path)
    dest = tmp_path / "2026-01-02.csv"
    with dest.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert rows[0]["ticker"] == "AAPL"
    assert rows[1]["ticker"] == "MSFT"
