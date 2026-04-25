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
        alphavantage_api_key="av_key",
        raw_dir=tmp_path / "data" / "raw",
    )
    # Replace real HTTP clients with mocks
    pipeline._polygon = MagicMock()
    pipeline._alphavantage = MagicMock()
    return pipeline


def _stub_universe(pipeline: IngestionPipeline, tickers: list[str]) -> None:
    pipeline._polygon.resolve_universe.return_value = tickers


def _stub_ohlcv(pipeline: IngestionPipeline, ticker: str, records: list) -> None:
    pipeline._polygon.fetch_ohlcv.return_value = records


def _stub_sentiment(pipeline: IngestionPipeline, ticker: str, record) -> None:
    pipeline._alphavantage.fetch_sentiment.return_value = record


def test_universe_failure_aborts_run(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    pipeline._polygon.resolve_universe.side_effect = Exception("API down")
    with pytest.raises(RuntimeError, match="universe resolution failed"):
        pipeline.run(date(2024, 1, 2), date(2024, 1, 2))
    # No OHLCV calls were made
    pipeline._polygon.fetch_ohlcv.assert_not_called()


def test_ohlcv_failure_excludes_ticker_and_writes_alert(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    _stub_universe(pipeline, ["AAPL", "MSFT"])
    pipeline._polygon.fetch_ohlcv.side_effect = Exception("timeout")
    pipeline._alphavantage.fetch_sentiment.return_value = MagicMock(
        ticker="MSFT", date=date(2024, 1, 2),
        bullish_percent=None, bearish_percent=None,
        company_news_score=None, article_count=None,
    )
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

    ohlcv_record = OHLCVRecord(
        ticker="AAPL", date=date(2024, 1, 2),
        open=100.0, high=110.0, low=90.0, close=105.0, volume=1000000,
    )
    pipeline._polygon.fetch_ohlcv.return_value = [ohlcv_record]
    pipeline._alphavantage.fetch_sentiment.side_effect = Exception("sentiment API down")

    pipeline.run(date(2024, 1, 2), date(2024, 1, 2))

    ohlcv_path = tmp_path / "data" / "raw" / "ohlcv" / "AAPL" / "2024-01-02.csv"
    assert ohlcv_path.exists()

    # No file written on failure — next nightly will retry instead of seeing a
    # permanently-null sentinel and skipping forever.
    sentiment_path = tmp_path / "data" / "raw" / "sentiment" / "AAPL" / "2024-01-02.csv"
    assert not sentiment_path.exists()


def test_idempotency_skips_existing_files(tmp_path):
    from src.ingestion.models.ohlcv import OHLCVRecord
    from src.ingestion.models.sentiment import SentimentRecord

    pipeline = _make_pipeline(tmp_path)
    _stub_universe(pipeline, ["AAPL"])

    ohlcv_record = OHLCVRecord(
        ticker="AAPL", date=date(2024, 1, 2),
        open=100.0, high=110.0, low=90.0, close=105.0, volume=1000000,
    )
    pipeline._polygon.fetch_ohlcv.return_value = [ohlcv_record]

    null_sentiment = SentimentRecord(ticker="AAPL", date=date(2024, 1, 2))
    pipeline._alphavantage.fetch_sentiment.return_value = null_sentiment

    pipeline.run(date(2024, 1, 2), date(2024, 1, 2))
    call_count_after_first = pipeline._polygon.fetch_ohlcv.call_count

    pipeline.run(date(2024, 1, 2), date(2024, 1, 2))
    # Second run should not have called fetch_ohlcv again
    assert pipeline._polygon.fetch_ohlcv.call_count == call_count_after_first


def test_run_metadata_written(tmp_path):
    from src.ingestion.models.ohlcv import OHLCVRecord
    from src.ingestion.models.sentiment import SentimentRecord

    pipeline = _make_pipeline(tmp_path)
    _stub_universe(pipeline, ["AAPL"])

    pipeline._polygon.fetch_ohlcv.return_value = [
        OHLCVRecord(
            ticker="AAPL", date=date(2024, 1, 2),
            open=100.0, high=110.0, low=90.0, close=105.0, volume=1000000,
        )
    ]
    pipeline._alphavantage.fetch_sentiment.return_value = SentimentRecord(
        ticker="AAPL", date=date(2024, 1, 2)
    )

    pipeline.run(date(2024, 1, 2), date(2024, 1, 2))
    meta_path = tmp_path / "data" / "raw" / "runs" / "2024-01-02.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["run_date"] == "2024-01-02"
    assert "universe_size" in meta
    assert "ohlcv_success" in meta
