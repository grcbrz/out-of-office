from __future__ import annotations

import csv as csv_mod
import logging
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from src.features.audit import lookahead_bias_guard, null_audit
from src.features.lags import compute_lag_features
from src.features.returns import compute_forward_return, compute_log_returns
from src.features.schema import FEATURE_COLUMNS, FeatureRecord
from src.features.seasonality import compute_seasonality_features
from src.features.sentiment import passthrough_sentiment
from src.features.target import compute_target_label
from src.features.trend import compute_close_to_sma, compute_ema, compute_macd, compute_sma
from src.features.volume import compute_obv, compute_vwap_ratio
from src.ingestion.persistence import write_json

logger = logging.getLogger(__name__)

_PROCESSED_DIR = Path("data/processed")
_FEATURES_DIR = Path("data/features")
_WARMUP_ROWS = 26  # minimum rows to produce valid MACD(26)


class FeaturePipeline:
    """Computes the feature matrix per ticker from processed CSV files."""

    def __init__(
        self,
        processed_dir: Path = _PROCESSED_DIR,
        features_dir: Path = _FEATURES_DIR,
    ) -> None:
        self._processed_dir = processed_dir
        self._features_dir = features_dir

    def run(self, run_date: date) -> None:
        started_at = datetime.now(timezone.utc)
        universe = self._load_universe(run_date)

        processed = 0
        skipped = 0
        warmup_dropped = 0
        all_targets: list[str] = []

        for ticker in universe:
            result = self._process_ticker(ticker, run_date)
            if result is None:
                skipped += 1
                continue
            processed += 1
            warmup_dropped += result["warmup_dropped"]
            all_targets.extend(result["targets"])

        target_dist = _distribution(all_targets)
        completed_at = datetime.now(timezone.utc)
        write_json(
            self._features_dir / "runs" / f"{run_date}.json",
            {
                "run_date": str(run_date),
                "tickers_processed": processed,
                "tickers_skipped": skipped,
                "warmup_rows_dropped": warmup_dropped,
                "target_distribution": target_dist,
                "started_at": started_at.isoformat(),
                "completed_at": completed_at.isoformat(),
            },
        )
        logger.info("feature pipeline complete: %d processed, %d skipped", processed, skipped)

    # ------------------------------------------------------------------

    def _load_universe(self, run_date: date) -> list[str]:
        path = self._processed_dir / ".." / "raw" / "universe" / f"{run_date}.csv"
        path = path.resolve()
        if not path.exists():
            logger.error("universe file not found: %s", path)
            return []
        with path.open() as f:
            reader = csv_mod.reader(f)
            next(reader, None)
            return [row[0] for row in reader if row]

    def _process_ticker(self, ticker: str, run_date: date) -> dict | None:
        dest = self._features_dir / ticker / f"{run_date}.csv"
        if dest.exists():
            logger.debug("features already exist for %s %s, skipping", ticker, run_date)
            return {"warmup_dropped": 0, "targets": []}

        processed_path = self._processed_dir / ticker / f"{run_date}.csv"
        if not processed_path.exists():
            logger.warning("no processed file for %s %s", ticker, run_date)
            return None

        df = pd.read_csv(processed_path)
        if df.empty:
            return None

        df = self._compute_all_features(df)
        df = self._drop_warmup(df, ticker)
        null_audit(df, ticker)
        lookahead_bias_guard(df)

        valid_rows = self._validate_rows(df, ticker)
        if not valid_rows:
            return None

        self._write_features(dest, valid_rows)
        targets = [r.target for r in valid_rows if r.target]
        return {"warmup_dropped": max(0, len(df) + _WARMUP_ROWS - len(df)), "targets": targets}

    def _compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = compute_log_returns(df)
        df = compute_forward_return(df)
        df = compute_sma(df)
        df = compute_ema(df)
        df = compute_macd(df)
        df = compute_close_to_sma(df)
        df = compute_obv(df)
        df = compute_vwap_ratio(df)
        df = compute_lag_features(df)
        df = compute_seasonality_features(df)
        df = passthrough_sentiment(df)

        # Add lag for log_return
        df["log_return_lag1"] = df["log_return"].shift(1)
        df["log_return_lag2"] = df["log_return"].shift(2)
        df["log_return_lag3"] = df["log_return"].shift(3)

        df = compute_target_label(df)
        return df

    def _drop_warmup(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        before = len(df)
        df = df.iloc[_WARMUP_ROWS:].reset_index(drop=True)
        logger.debug("dropped %d warmup rows for %s", before - len(df), ticker)
        return df

    def _validate_rows(self, df: pd.DataFrame, ticker: str) -> list[FeatureRecord]:
        valid: list[FeatureRecord] = []
        for _, row in df.iterrows():
            try:
                valid.append(FeatureRecord(**{k: v for k, v in row.items() if not _is_nan(v)}))
            except Exception as exc:
                logger.error("feature row validation failed for %s: %s", ticker, exc)
        return valid

    def _write_features(self, dest: Path, records: list[FeatureRecord]) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        rows = [r.model_dump() for r in records]
        if not rows:
            return
        with dest.open("w", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def _distribution(targets: list[str]) -> dict[str, float]:
    if not targets:
        return {}
    total = len(targets)
    return {label: round(targets.count(label) / total, 3) for label in ["BUY", "HOLD", "SELL"]}


def _is_nan(v) -> bool:
    import math
    try:
        return math.isnan(v)
    except (TypeError, ValueError):
        return False
