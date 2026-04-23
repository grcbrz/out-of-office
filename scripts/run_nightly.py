"""Nightly batch orchestrator — full pipeline wiring (Specs 01–07)."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("nightly")

_STATUS_PATH = Path("data/monitoring/status.json")
_RETRAIN_CADENCE_DAYS = 21


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nightly batch pipeline")
    parser.add_argument(
        "--start-date",
        required=True,
        type=date.fromisoformat,
        help="OHLCV fetch start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--run-date",
        type=date.fromisoformat,
        default=date.today(),
        help="Run date (default: today)",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retraining regardless of cadence or drift flags",
    )
    return parser.parse_args()


def _read_status() -> dict:
    if _STATUS_PATH.exists():
        return json.loads(_STATUS_PATH.read_text())
    return {"retraining_required": False, "last_monitored": None}


def _should_retrain(run_date: date, force: bool) -> bool:
    if force:
        return True
    status = _read_status()
    if status.get("retraining_required", False):
        logger.info("drift/degradation flag set — unscheduled retraining triggered")
        return True
    last = status.get("last_monitored")
    if last is None:
        return True
    days_since = (run_date - date.fromisoformat(last)).days
    return days_since >= _RETRAIN_CADENCE_DAYS


def main() -> None:
    args = _parse_args()
    run_date = args.run_date

    polygon_key = os.environ.get("POLYGON_API_KEY", "")
    finnhub_key = os.environ.get("FINNHUB_API_KEY", "")

    if not polygon_key or not finnhub_key:
        logger.critical("POLYGON_API_KEY and FINNHUB_API_KEY must be set in environment")
        sys.exit(1)

    logger.info("=== nightly run: %s (start_date=%s) ===", run_date, args.start_date)

    # 1. Ingestion
    from src.ingestion.pipeline import IngestionPipeline
    IngestionPipeline(polygon_key, finnhub_key).run(run_date, args.start_date)
    logger.info("ingestion complete")

    # 2. Preprocessing
    from src.preprocessing.pipeline import PreprocessingPipeline
    PreprocessingPipeline().run(run_date)
    logger.info("preprocessing complete")

    # 3. Feature Engineering
    from src.features.pipeline import FeaturePipeline
    FeaturePipeline().run(run_date)
    logger.info("feature engineering complete")

    # 4. Monitoring (before training decision)
    from src.monitoring.pipeline import MonitoringPipeline
    import numpy as np
    import pandas as pd
    # Load feature and prediction data for monitoring (empty stubs if not present)
    predictions_csv = Path(f"data/predictions/{run_date}.csv")
    predictions_df = pd.read_csv(predictions_csv) if predictions_csv.exists() else pd.DataFrame(columns=["ticker", "signal", "run_date"])

    ohlcv_files = list(Path("data/raw/ohlcv").glob("*/*.csv"))
    ohlcv_frames = [pd.read_csv(f) for f in ohlcv_files] if ohlcv_files else []
    ohlcv_df = pd.concat(ohlcv_frames, ignore_index=True) if ohlcv_frames else pd.DataFrame(columns=["ticker", "date", "close"])

    MonitoringPipeline().run(
        run_date=run_date,
        reference_feature_stats={},
        current_feature_stats={},
        reference_signal_counts={"BUY": 1, "HOLD": 1, "SELL": 1},
        current_signal_counts={"BUY": 1, "HOLD": 1, "SELL": 1},
        predictions_df=predictions_df,
        ohlcv_df=ohlcv_df,
    )
    logger.info("monitoring complete")

    # 5. Training + Evaluation (conditional)
    if _should_retrain(run_date, args.force_retrain):
        from src.models.training_pipeline import TrainingPipeline
        TrainingPipeline().run(run_date)
        logger.info("training complete")

        from src.evaluation.pipeline import EvaluationPipeline
        try:
            EvaluationPipeline().run(run_date)
            logger.info("evaluation complete")
            # Reset drift flag after successful retraining + quality gate pass
            from src.monitoring.persistence import update_status
            update_status(_STATUS_PATH, retraining_required=False)
        except Exception as exc:
            logger.error("evaluation/quality gate failed: %s — retraining flag kept", exc)
    else:
        logger.info("retraining skipped (cadence not reached, no drift flag)")

    # 6. Prediction
    from scripts.prediction_client import PredictionClient
    api_token = os.environ.get("API_TOKEN", "")
    if api_token:
        PredictionClient().run(run_date)
        logger.info("prediction complete")
    else:
        logger.warning("API_TOKEN not set — skipping prediction client")

    logger.info("=== nightly run complete: %s ===", run_date)


if __name__ == "__main__":
    main()
