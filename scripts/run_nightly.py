"""Nightly batch orchestrator — full pipeline wiring (Specs 01–07)."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

import httpx

# Add repo root to sys.path so src imports work when running this script
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("nightly")

_STATUS_PATH = Path("data/monitoring/status.json")
_PRODUCTION_DIR = Path("models/production")
_FEATURES_DIR = Path("data/features")
_PREDICTIONS_DIR = Path("data/predictions")
_RAW_OHLCV_DIR = Path("data/raw/ohlcv")
_RETRAIN_CADENCE_DAYS = 21
_CURRENT_WINDOW_DAYS = 21


def _default_start_date() -> date:
    from datetime import timedelta
    return date.today() - timedelta(days=730)  # 2 years — Polygon free-tier limit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nightly batch pipeline")
    parser.add_argument(
        "--start-date",
        required=False,
        default=None,
        type=date.fromisoformat,
        help="OHLCV fetch start date in YYYY-MM-DD format (default: 2 years ago)",
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


def _resolve_production_artifact() -> Path | None:
    if not _PRODUCTION_DIR.exists():
        return None
    candidates = [p for p in _PRODUCTION_DIR.iterdir() if p.is_dir() and (p / "metadata.json").exists()]
    return candidates[0] if candidates else None


def _load_recent_predictions(window_end: date, window_days: int):
    import pandas as pd
    from datetime import timedelta
    if not _PREDICTIONS_DIR.exists():
        return pd.DataFrame(columns=["ticker", "signal", "run_date"])
    cutoff = window_end - timedelta(days=window_days)
    frames = []
    for csv in _PREDICTIONS_DIR.glob("*.csv"):
        try:
            file_date = date.fromisoformat(csv.stem)
        except ValueError:
            continue
        if cutoff <= file_date <= window_end:
            frames.append(pd.read_csv(csv))
    if not frames:
        return pd.DataFrame(columns=["ticker", "signal", "run_date"])
    return pd.concat(frames, ignore_index=True)


def _signal_counts(predictions_df) -> dict[str, int]:
    if predictions_df.empty:
        return {"BUY": 0, "HOLD": 0, "SELL": 0}
    counts = predictions_df["signal"].value_counts().to_dict()
    return {label: int(counts.get(label, 0)) for label in ("BUY", "HOLD", "SELL")}


def _run_monitoring(run_date: date) -> None:
    import pandas as pd
    from datetime import timedelta
    from src.features.schema import CONTINUOUS_FEATURE_COLUMNS
    from src.monitoring.persistence import load_feature_window, load_monitoring_reference
    from src.monitoring.pipeline import MonitoringPipeline

    artifact = _resolve_production_artifact()
    if artifact is None:
        logger.warning("monitoring skipped — no production model artifact yet")
        return

    try:
        reference = load_monitoring_reference(artifact)
    except FileNotFoundError:
        logger.warning("monitoring skipped — production artifact missing monitoring_reference.json")
        return

    ref_start = date.fromisoformat(reference["training_start_date"])
    ref_end = date.fromisoformat(reference["training_end_date"])
    cur_end = run_date
    cur_start = run_date - timedelta(days=_CURRENT_WINDOW_DAYS * 2)  # calendar slack for ~21 trading days

    reference_stats = load_feature_window(_FEATURES_DIR, ref_start, ref_end, CONTINUOUS_FEATURE_COLUMNS)
    current_stats = load_feature_window(_FEATURES_DIR, cur_start, cur_end, CONTINUOUS_FEATURE_COLUMNS)

    predictions_df = _load_recent_predictions(run_date, _CURRENT_WINDOW_DAYS)
    current_signal_counts = _signal_counts(predictions_df)

    ohlcv_files = list(_RAW_OHLCV_DIR.glob("*/*.csv"))
    ohlcv_df = (
        pd.concat([pd.read_csv(f) for f in ohlcv_files], ignore_index=True)
        if ohlcv_files else pd.DataFrame(columns=["ticker", "date", "close"])
    )

    MonitoringPipeline().run(
        run_date=run_date,
        reference_feature_stats=reference_stats,
        current_feature_stats=current_stats,
        reference_signal_counts=reference["signal_counts"],
        current_signal_counts=current_signal_counts,
        predictions_df=predictions_df,
        ohlcv_df=ohlcv_df,
    )


def main() -> None:
    args = _parse_args()
    run_date = args.run_date
    start_date = args.start_date or _default_start_date()

    polygon_key = os.environ.get("POLYGON_API_KEY", "")
    alphavantage_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")

    if not polygon_key or not alphavantage_key:
        logger.critical("POLYGON_API_KEY and ALPHA_VANTAGE_API_KEY must be set in environment")
        sys.exit(1)

    logger.info("=== nightly run: %s (start_date=%s) ===", run_date, start_date)

    # 1. Ingestion
    from src.ingestion.pipeline import IngestionPipeline
    IngestionPipeline(polygon_key, alphavantage_key).run(run_date, start_date)
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
    _run_monitoring(run_date)
    logger.info("monitoring complete")

    # 5. Training + Evaluation (conditional)
    if _should_retrain(run_date, args.force_retrain):
        import pandas as pd
        from src.models.training_pipeline import TrainingPipeline
        feature_files = list(Path("data/features").glob("*/*.csv"))
        if not feature_files:
            logger.error("no feature files found — skipping training")
        else:
            global_df = pd.concat(
                [pd.read_csv(f) for f in feature_files], ignore_index=True
            )
            TrainingPipeline().run(global_df)
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
        try:
            PredictionClient().run(run_date)
            logger.info("prediction complete")
        except httpx.ConnectError:
            logger.warning("prediction skipped — API server is not running (start with 'make run')")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 503:
                logger.warning("prediction skipped — server running but no model loaded yet (run training first)")
            else:
                logger.warning("prediction failed with HTTP %d", exc.response.status_code)
    else:
        logger.warning("API_TOKEN not set — skipping prediction client")

    logger.info("=== nightly run complete: %s ===", run_date)


if __name__ == "__main__":
    main()
