from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.ingestion.pipeline import IngestionPipeline


def _make_pipeline(tmp_path: Path) -> IngestionPipeline:
    pipeline = IngestionPipeline(
        polygon_api_key="poly_key",
        raw_dir=tmp_path / "data" / "raw",
    )
    # Replace real HTTP clients and heavy model with mocks
    pipeline._polygon = MagicMock()
    pipeline._finbert = None   # skip FinBERT in tests; fallback uses Polygon labels
    # Tests that don't set this explicitly use the dynamic Polygon path
    pipeline._fixed_universe = []
    return pipeline


def _stub_universe(pipeline: IngestionPipeline, tickers: list[str]) -> None:
    pipeline._polygon.resolve_universe.return_value = tickers


def _stub_ohlcv(pipeline: IngestionPipeline, ticker: str, records: list) -> None:
    pipeline._polygon.fetch_ohlcv.return_value = records


def _stub_sentiment(pipeline: IngestionPipeline, ticker: str, record) -> None:
    pipeline._polygon.fetch_news_sentiment.return_value = record


def test_universe_failure_aborts_run(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    pipeline._polygon.resolve_universe.side_effect = Exception("API down")
    with pytest.raises(RuntimeError, match="universe resolution failed"):
        pipeline.run(date(2024, 1, 2), date(2024, 1, 2))
    pipeline._polygon.fetch_ohlcv.assert_not_called()


def test_ohlcv_failure_excludes_ticker_and_writes_alert(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    _stub_universe(pipeline, ["AAPL", "MSFT"])
    pipeline._polygon.fetch_ohlcv.side_effect = Exception("timeout")
    pipeline.run(date(2024, 1, 2), date(2024, 1, 2))
    alert_path = tmp_path / "data" / "raw" / "alerts" / "2024-01-02.json"
    assert alert_path.exists()
    alert = json.loads(alert_path.read_text())
    assert "AAPL" in alert["ohlcv_failed"]
    assert "MSFT" in alert["ohlcv_failed"]


def test_sentiment_failure_leaves_no_file_so_next_run_retries(tmp_path):
    from src.ingestion.models.ohlcv import OHLCVRecord

    pipeline = _make_pipeline(tmp_path)
    _stub_universe(pipeline, ["AAPL"])

    pipeline._polygon.fetch_ohlcv.return_value = [
        OHLCVRecord(ticker="AAPL", date=date(2024, 1, 2),
                    open=100.0, high=110.0, low=90.0, close=105.0, volume=1000000)
    ]
    pipeline._polygon.fetch_news_sentiment.side_effect = Exception("sentiment API down")

    pipeline.run(date(2024, 1, 2), date(2024, 1, 2))

    assert (tmp_path / "data" / "raw" / "ohlcv" / "AAPL" / "2024-01-02.csv").exists()
    # No file written on failure — next nightly will retry
    assert not (tmp_path / "data" / "raw" / "sentiment" / "AAPL" / "2024-01-02.csv").exists()


def test_fixed_universe_bypasses_polygon_resolution(tmp_path):
    from src.ingestion.models.ohlcv import OHLCVRecord
    from src.ingestion.models.sentiment import SentimentRecord

    pipeline = _make_pipeline(tmp_path)
    pipeline._fixed_universe = ["AAPL", "NVDA"]

    pipeline._polygon.fetch_ohlcv.return_value = [
        OHLCVRecord(ticker="AAPL", date=date(2024, 1, 2),
                    open=100.0, high=110.0, low=90.0, close=105.0, volume=1_000_000)
    ]
    pipeline._polygon.fetch_news_sentiment.return_value = SentimentRecord(
        ticker="AAPL", date=date(2024, 1, 2),
    )

    pipeline.run(date(2024, 1, 2), date(2024, 1, 2))
    pipeline._polygon.resolve_universe.assert_not_called()


def test_idempotency_skips_existing_files(tmp_path):
    from src.ingestion.models.ohlcv import OHLCVRecord
    from src.ingestion.models.sentiment import SentimentRecord

    pipeline = _make_pipeline(tmp_path)
    _stub_universe(pipeline, ["AAPL"])

    pipeline._polygon.fetch_ohlcv.return_value = [
        OHLCVRecord(ticker="AAPL", date=date(2024, 1, 2),
                    open=100.0, high=110.0, low=90.0, close=105.0, volume=1000000)
    ]
    pipeline._polygon.fetch_news_sentiment.return_value = SentimentRecord(
        ticker="AAPL", date=date(2024, 1, 2),
    )

    pipeline.run(date(2024, 1, 2), date(2024, 1, 2))
    call_count_after_first = pipeline._polygon.fetch_ohlcv.call_count

    pipeline.run(date(2024, 1, 2), date(2024, 1, 2))
    assert pipeline._polygon.fetch_ohlcv.call_count == call_count_after_first


def test_run_metadata_written(tmp_path):
    from src.ingestion.models.ohlcv import OHLCVRecord
    from src.ingestion.models.sentiment import SentimentRecord

    pipeline = _make_pipeline(tmp_path)
    _stub_universe(pipeline, ["AAPL"])

    pipeline._polygon.fetch_ohlcv.return_value = [
        OHLCVRecord(ticker="AAPL", date=date(2024, 1, 2),
                    open=100.0, high=110.0, low=90.0, close=105.0, volume=1000000)
    ]
    pipeline._polygon.fetch_news_sentiment.return_value = SentimentRecord(
        ticker="AAPL", date=date(2024, 1, 2),
    )

    pipeline.run(date(2024, 1, 2), date(2024, 1, 2))
    meta_path = tmp_path / "data" / "raw" / "runs" / "2024-01-02.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["run_date"] == "2024-01-02"
    assert "universe_size" in meta
    assert "ohlcv_success" in meta
