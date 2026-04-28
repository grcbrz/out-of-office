from __future__ import annotations

import csv
import logging
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

from src.preprocessing.imputer import fill_volume, forward_fill_close
from src.preprocessing.loader import load_ohlcv, load_sentiment
from src.preprocessing.merger import merge_ohlcv_sentiment
from src.preprocessing.outlier import flag_outliers
from src.preprocessing.validator import ProcessedRecord
from src.ingestion.persistence import write_json

logger = logging.getLogger(__name__)

_RAW_DIR = Path("data/raw")
_PROCESSED_DIR = Path("data/processed")


class PreprocessingPipeline:
    """Transforms raw OHLCV + sentiment CSVs into validated processed CSVs per ticker."""

    def __init__(
        self,
        raw_dir: Path = _RAW_DIR,
        processed_dir: Path = _PROCESSED_DIR,
    ) -> None:
        self._raw_dir = raw_dir
        self._processed_dir = processed_dir
        self._calendar = mcal.get_calendar("NYSE")

    def run(self, run_date: date) -> None:
        started_at = datetime.now(timezone.utc)
        universe = self._load_universe(run_date)
        if not universe:
            logger.warning("empty universe for %s; nothing to preprocess", run_date)
            return

        schedule = self._calendar.schedule(
            start_date="2020-01-01", end_date=str(run_date)
        )

        processed = 0
        skipped_detail: list[dict] = []
        total_close_outliers = 0
        total_volume_outliers = 0
        total_imputed_close = 0
        total_imputed_volume = 0

        for ticker in universe:
            result = self._process_ticker(ticker, run_date, schedule)
            if result is None:
                skipped_detail.append({"ticker": ticker, "reason": "processing failed"})
                continue
            processed += 1
            total_close_outliers += result["close_outliers"]
            total_volume_outliers += result["volume_outliers"]
            total_imputed_close += result["imputed_close"]
            total_imputed_volume += result["imputed_volume"]

        completed_at = datetime.now(timezone.utc)
        meta = {
            "run_date": str(run_date),
            "tickers_processed": processed,
            "tickers_skipped": len(skipped_detail),
            "skipped_detail": skipped_detail,
            "outliers_flagged": {"close": total_close_outliers, "volume": total_volume_outliers},
            "imputed_rows": {"close": total_imputed_close, "volume": total_imputed_volume},
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
        }
        write_json(self._processed_dir / "runs" / f"{run_date}.json", meta)
        logger.info("preprocessing complete: %d processed, %d skipped", processed, len(skipped_detail))

    # ------------------------------------------------------------------

    def _load_universe(self, run_date: date) -> list[str]:
        universe_path = self._raw_dir / "universe" / f"{run_date}.csv"
        if not universe_path.exists():
            logger.error("universe file not found: %s", universe_path)
            return []
        with universe_path.open() as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            return [row[0] for row in reader if row]

    def _process_ticker(
        self, ticker: str, run_date: date, schedule: pd.DataFrame
    ) -> dict | None:
        dest = self._processed_dir / ticker / f"{run_date}.csv"
        if dest.exists():
            logger.debug("processed file already exists for %s %s, skipping", ticker, run_date)
            return {"close_outliers": 0, "volume_outliers": 0, "imputed_close": 0, "imputed_volume": 0}

        ohlcv_path = self._raw_dir / "ohlcv" / ticker / f"{run_date}.csv"
        if not ohlcv_path.exists():
            logger.warning("no raw ohlcv for %s %s", ticker, run_date)
            return None

        ohlcv_records = load_ohlcv(ohlcv_path)
        if not ohlcv_records:
            return None

        ohlcv_df = pd.DataFrame([r.model_dump() for r in ohlcv_records])

        sentiment_path = self._raw_dir / "sentiment" / ticker / f"{run_date}.csv"
        sent_df = pd.DataFrame()
        if sentiment_path.exists():
            sent_records = load_sentiment(sentiment_path)
            if sent_records:
                sent_df = pd.DataFrame([r.model_dump() for r in sent_records])

        try:
            ohlcv_df = forward_fill_close(ohlcv_df, schedule)
            ohlcv_df = fill_volume(ohlcv_df)
            ohlcv_df = flag_outliers(ohlcv_df)
            merged = merge_ohlcv_sentiment(ohlcv_df, sent_df if not sent_df.empty else pd.DataFrame())
        except Exception as exc:
            logger.error("processing failed for %s: %s", ticker, exc)
            return None

        valid_rows = self._validate_rows(merged, ticker)
        if not valid_rows:
            return None

        self._write_processed(dest, valid_rows)
        return {
            "close_outliers": int(merged["close_outlier_flag"].sum()),
            "volume_outliers": int(merged["volume_outlier_flag"].sum()),
            "imputed_close": int(merged.get("imputed_close", pd.Series([False])).sum()),
            "imputed_volume": int(merged.get("imputed_volume", pd.Series([False])).sum()),
        }

    def _validate_rows(self, df: pd.DataFrame, ticker: str) -> list[ProcessedRecord]:
        import math
        valid: list[ProcessedRecord] = []
        for _, row in df.iterrows():
            row_dict = {
                k: (None if isinstance(v, float) and math.isnan(v) else v)
                for k, v in row.to_dict().items()
            }
            try:
                valid.append(ProcessedRecord(**row_dict))
            except Exception as exc:
                logger.error("processed row validation failed for %s: %s", ticker, exc)
        return valid

    def _write_processed(self, dest: Path, records: list[ProcessedRecord]) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        rows = [r.model_dump() for r in records]
        if not rows:
            return
        import csv as csv_mod
        with dest.open("w", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
