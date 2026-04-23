from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path


from src.features.pipeline import FeaturePipeline, _distribution, _is_nan


def _write_universe(raw_dir: Path, run_date: date, tickers: list[str]) -> None:
    path = raw_dir / "universe" / f"{run_date}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker"])
        for t in tickers:
            writer.writerow([t])


def _write_processed_csv(processed_dir: Path, ticker: str, run_date: date, n: int = 80) -> Path:
    path = processed_dir / ticker / f"{run_date}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    import numpy as np
    rng = np.random.default_rng(0)
    base = 100.0
    rows = []
    for i in range(n):
        from datetime import timedelta
        d = date(2023, 1, 2) + timedelta(days=i)
        rows.append({
            "ticker": ticker,
            "date": str(d),
            "open": base,
            "high": base + 1,
            "low": base - 1,
            "close": base + rng.normal(0, 0.5),
            "volume": 1_000_000,
            "vwap": base,
            "close_zscore": 0.0,
            "volume_zscore": 0.0,
            "close_outlier_flag": False,
            "volume_outlier_flag": False,
            "bullish_percent": None,
            "bearish_percent": None,
            "company_news_score": None,
            "buzz_weekly_average": None,
            "sentiment_available": False,
            "is_trading_day": True,
            "imputed_close": False,
            "imputed_volume": False,
        })

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_pipeline_skips_missing_universe(tmp_path):
    proc_dir = tmp_path / "processed"
    feat_dir = tmp_path / "features"
    run_date = date(2026, 4, 23)
    pipeline = FeaturePipeline(processed_dir=proc_dir, features_dir=feat_dir)
    pipeline.run(run_date)
    # Runs JSON is written, but no ticker feature files
    meta_path = feat_dir / "runs" / f"{run_date}.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["tickers_processed"] == 0


def test_pipeline_skips_ticker_without_processed_file(tmp_path):
    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"
    feat_dir = tmp_path / "features"
    run_date = date(2026, 4, 23)

    _write_universe(raw_dir, run_date, ["AAPL"])
    pipeline = FeaturePipeline(processed_dir=proc_dir, features_dir=feat_dir)

    # Patch _load_universe to use raw_dir
    pipeline._processed_dir = proc_dir

    pipeline.run(run_date)
    meta = json.loads((feat_dir / "runs" / f"{run_date}.json").read_text())
    assert meta["tickers_skipped"] == 1


def test_pipeline_idempotent_when_features_exist(tmp_path):
    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"
    feat_dir = tmp_path / "features"
    run_date = date(2026, 4, 23)

    _write_universe(raw_dir, run_date, ["AAPL"])
    _write_processed_csv(proc_dir, "AAPL", run_date)

    # Pre-create dest
    dest = feat_dir / "AAPL" / f"{run_date}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("ticker,date\nAAPL,2026-04-23\n")

    pipeline = FeaturePipeline(processed_dir=proc_dir, features_dir=feat_dir)
    pipeline.run(run_date)
    meta = json.loads((feat_dir / "runs" / f"{run_date}.json").read_text())
    assert meta["tickers_processed"] == 1


def test_pipeline_processes_valid_ticker(tmp_path):
    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"
    feat_dir = tmp_path / "features"
    run_date = date(2026, 4, 23)

    _write_universe(raw_dir, run_date, ["AAPL"])
    _write_processed_csv(proc_dir, "AAPL", run_date, n=100)

    pipeline = FeaturePipeline(processed_dir=proc_dir, features_dir=feat_dir)
    pipeline.run(run_date)

    meta_path = feat_dir / "runs" / f"{run_date}.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["tickers_processed"] == 1


def test_distribution_empty():
    assert _distribution([]) == {}


def test_distribution_counts():
    targets = ["BUY", "BUY", "SELL", "HOLD"]
    d = _distribution(targets)
    assert d["BUY"] == 0.5
    assert d["SELL"] == 0.25
    assert d["HOLD"] == 0.25


def test_is_nan_with_float_nan():
    import math
    assert _is_nan(math.nan)


def test_is_nan_with_normal_values():
    assert not _is_nan(1.0)
    assert not _is_nan("hello")
    assert not _is_nan(None)
