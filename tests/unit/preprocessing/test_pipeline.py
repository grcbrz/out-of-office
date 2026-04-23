from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.preprocessing.pipeline import PreprocessingPipeline


def _write_universe(raw_dir: Path, run_date: date, tickers: list[str]) -> None:
    path = raw_dir / "universe" / f"{run_date}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker"])
        for t in tickers:
            writer.writerow([t])


def _write_ohlcv_csv(raw_dir: Path, ticker: str, run_date: date) -> None:
    path = raw_dir / "ohlcv" / ticker / f"{run_date}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ticker", "date", "open", "high", "low", "close", "volume", "vwap"])
        writer.writeheader()
        for i in range(70):
            d = f"202{i//365 + 3}-{(i//30) % 12 + 1:02d}-{(i % 28) + 1:02d}"
            writer.writerow({
                "ticker": ticker, "date": f"2023-{(i % 11) + 1:02d}-{(i % 28) + 1:02d}",
                "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0,
                "volume": 1_000_000, "vwap": 101.0,
            })


def test_pipeline_skips_missing_universe(tmp_path):
    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"
    pipeline = PreprocessingPipeline(raw_dir=raw_dir, processed_dir=proc_dir)
    pipeline.run(date(2026, 4, 23))
    # No error raised; processed/runs should not exist
    assert not (proc_dir / "runs").exists()


def test_pipeline_skips_ticker_without_ohlcv(tmp_path):
    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"
    run_date = date(2026, 4, 23)

    _write_universe(raw_dir, run_date, ["AAPL"])
    pipeline = PreprocessingPipeline(raw_dir=raw_dir, processed_dir=proc_dir)

    mock_cal = MagicMock()
    mock_cal.schedule.return_value = pd.DataFrame()
    pipeline._calendar = mock_cal

    pipeline.run(run_date)
    # AAPL has no ohlcv → skipped; meta written but tickers_processed=0
    meta_path = proc_dir / "runs" / f"{run_date}.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["tickers_processed"] == 0
    assert meta["tickers_skipped"] == 1


def test_pipeline_idempotent_when_dest_exists(tmp_path):
    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"
    run_date = date(2026, 4, 23)

    _write_universe(raw_dir, run_date, ["AAPL"])

    # Pre-create the destination
    dest = proc_dir / "AAPL" / f"{run_date}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("ticker,date\nAAPL,2026-04-23\n")

    pipeline = PreprocessingPipeline(raw_dir=raw_dir, processed_dir=proc_dir)
    mock_cal = MagicMock()
    mock_cal.schedule.return_value = pd.DataFrame()
    pipeline._calendar = mock_cal

    pipeline.run(run_date)
    meta_path = proc_dir / "runs" / f"{run_date}.json"
    meta = json.loads(meta_path.read_text())
    # Idempotent: counts as processed (0 outliers)
    assert meta["tickers_processed"] == 1
