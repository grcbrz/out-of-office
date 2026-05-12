"""Nightly batch orchestrator — full pipeline wiring (Specs 01–07)."""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path
from contextlib import contextmanager

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

# Memory monitoring threshold (8GB)
_MEMORY_LIMIT_MB = 8192

@contextmanager
def memory_monitor(stage_name: str):
    import psutil
    """Monitor memory usage before and after a stage, force GC if needed."""
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024
    logger.info(f"Memory before {stage_name}: {mem_before:.1f} MB")

    yield

    gc.collect()
    mem_after = process.memory_info().rss / 1024 / 1024
    delta = mem_after - mem_before
    logger.info(f"Memory after {stage_name}: {mem_after:.1f} MB (Δ: {delta:+.1f} MB)")

    if mem_after > _MEMORY_LIMIT_MB:
        logger.warning(f"High memory usage after {stage_name}: {mem_after:.1f} MB")

    # Force aggressive GC if memory is high
    if mem_after > _MEMORY_LIMIT_MB * 0.8:
        logger.info("Forcing aggressive garbage collection")
        for _ in range(3):
            gc.collect()

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
    candidates = sorted(
        [p for p in _PRODUCTION_DIR.iterdir() if p.is_dir() and (p / "metadata.json").exists()],
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
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
    result = pd.concat(frames, ignore_index=True)
    # Clean up intermediate frames
    del frames
    return result


def _reload_server(api_token: str) -> None:
    if not api_token:
        return
    try:
        with httpx.Client(base_url="http://127.0.0.1:8000", timeout=30) as client:
            client.post("/reload", headers={"Authorization": f"Bearer {api_token}"})
        logger.info("server model reloaded")
    except Exception as exc:
        logger.warning("server reload skipped: %s", exc)


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

    # Detect universe change: if the saved reference tickers differ from the
    # current fixed_universe, drift stats are meaningless (different companies).
    # Skip feature/prediction drift and let cadence-based retraining handle it.
    universe_changed = False
    ref_tickers = reference.get("tickers")
    if ref_tickers is not None:
        import yaml as _yaml
        _ing_cfg = _yaml.safe_load(Path("configs/ingestion.yaml").read_text())
        current_tickers = sorted(_ing_cfg.get("fixed_universe") or [])
        if sorted(ref_tickers) != current_tickers:
            logger.warning(
                "universe changed (reference has %d tickers, current has %d) — "
                "skipping drift detection this run; will reset after next successful training",
                len(ref_tickers), len(current_tickers),
            )
            universe_changed = True

    if universe_changed:
        reference_stats: dict = {}
        current_stats: dict = {}
    else:
        reference_stats = load_feature_window(_FEATURES_DIR, ref_start, ref_end, CONTINUOUS_FEATURE_COLUMNS)
        current_stats = load_feature_window(_FEATURES_DIR, cur_start, cur_end, CONTINUOUS_FEATURE_COLUMNS)

    predictions_df = _load_recent_predictions(run_date, _CURRENT_WINDOW_DAYS)
    current_signal_counts = _signal_counts(predictions_df)

    ohlcv_files = list(_RAW_OHLCV_DIR.glob("*/*.csv"))
    if ohlcv_files:
        # Load in chunks to avoid memory spike
        chunks = []
        for ohlcv_file in ohlcv_files[:10]:  # Limit to avoid memory issues
            chunks.append(pd.read_csv(ohlcv_file))
        ohlcv_df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=["ticker", "date", "close"])
        del chunks
    else:
        ohlcv_df = pd.DataFrame(columns=["ticker", "date", "close"])

    MonitoringPipeline().run(
        run_date=run_date,
        reference_feature_stats=reference_stats,
        current_feature_stats=current_stats,
        reference_signal_counts={} if universe_changed else reference["signal_counts"],
        current_signal_counts=current_signal_counts,
        predictions_df=predictions_df,
        ohlcv_df=ohlcv_df,
    )
    # Clean up
    del reference_stats, current_stats, predictions_df, ohlcv_df
    gc.collect()


def _build_training_pipeline():
    """Instantiate TrainingPipeline from configs/training.yaml walk_forward section."""
    import yaml
    from src.models.training_pipeline import TrainingPipeline

    config_path = Path("configs/training.yaml")
    wf: dict = {}
    tr: dict = {}
    if config_path.exists():
        with config_path.open() as f:
            cfg = yaml.safe_load(f) or {}
        wf = cfg.get("walk_forward", {})
        tr = cfg.get("training", {})

    return TrainingPipeline(
        train_window=wf.get("train_window", 120),
        step_size=wf.get("step_size", 21),
        random_seed=tr.get("random_seed", 42),
        weight_half_life_days=tr.get("weight_half_life_days", 0),
    )


def _load_features_chunked(feature_files, chunk_size: int = 5):
    """Load feature files in chunks to manage memory."""
    import pandas as pd
    if not feature_files:
        return None

    all_dfs = []
    for i in range(0, len(feature_files), chunk_size):
        chunk_files = feature_files[i:i + chunk_size]
        chunk_dfs = []
        for f in chunk_files:
            try:
                # Only load necessary columns if possible
                df = pd.read_csv(f)
                chunk_dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")

        if chunk_dfs:
            chunk_df = pd.concat(chunk_dfs, ignore_index=True)
            all_dfs.append(chunk_df)
            del chunk_dfs
            gc.collect()

    if not all_dfs:
        return None

    result = pd.concat(all_dfs, ignore_index=True)
    del all_dfs
    gc.collect()

    logger.info(f"Loaded {len(feature_files)} feature files, resulting shape: {result.shape}")
    return result


def main() -> None:
    args = _parse_args()
    run_date = args.run_date
    start_date = args.start_date or _default_start_date()

    polygon_key = os.environ.get("POLYGON_API_KEY", "")
    api_token = os.environ.get("API_TOKEN", "")

    # Set environment variables for memory optimization
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    if not polygon_key:
        logger.critical("POLYGON_API_KEY must be set in environment")
        sys.exit(1)

    logger.info("=== nightly run: %s (start_date=%s) ===", run_date, start_date)

    # 1. Ingestion
    with memory_monitor("ingestion"):
        from src.ingestion.pipeline import IngestionPipeline
        IngestionPipeline(polygon_key).run(run_date, start_date)
        # Force cleanup of any cached data
        gc.collect()
    logger.info("ingestion complete")

    # 2. Preprocessing
    with memory_monitor("preprocessing"):
        from src.preprocessing.pipeline import PreprocessingPipeline
        PreprocessingPipeline().run(run_date)
        gc.collect()
    logger.info("preprocessing complete")

    # 3. Feature Engineering
    with memory_monitor("feature engineering"):
        from src.features.pipeline import FeaturePipeline
        FeaturePipeline().run(run_date)
        gc.collect()
    logger.info("feature engineering complete")

    # 4. Monitoring (before training decision)
    with memory_monitor("monitoring"):
        _run_monitoring(run_date)
        gc.collect()
    logger.info("monitoring complete")

    # 5. Training + Evaluation (conditional)
    if _should_retrain(run_date, args.force_retrain):
        with memory_monitor("training"):
            import pandas as pd

            feature_files = list(Path("data/features").glob("*/*.csv"))
            if not feature_files:
                logger.error("no feature files found — skipping training")
            else:
                # Load features in chunks to avoid memory explosion
                global_df = _load_features_chunked(feature_files, chunk_size=3)

                if global_df is not None:
                    training_pipeline = _build_training_pipeline()
                    training_pipeline.run(global_df)

                    # Clean up large DataFrame
                    del global_df
                    gc.collect()

                    logger.info("training complete")

                    # 5a. Evaluation
                    from src.evaluation.pipeline import EvaluationPipeline
                    try:
                        EvaluationPipeline().run(run_date)
                        logger.info("evaluation complete")
                        # Reset drift flag after successful retraining + quality gate pass
                        from src.monitoring.persistence import update_status
                        update_status(_STATUS_PATH, retraining_required=False)
                        _reload_server(api_token)
                    except Exception as exc:
                        logger.error("evaluation/quality gate failed: %s — retraining flag kept", exc)
                else:
                    logger.error("failed to load feature data")
    else:
        logger.info("retraining skipped (cadence not reached, no drift flag)")

    # 6. Prediction
    with memory_monitor("prediction"):
        from scripts.prediction_client import PredictionClient
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

        gc.collect()

    # Final memory cleanup
    gc.collect()
    logger.info("=== nightly run complete: %s ===", run_date)


if __name__ == "__main__":
    main()
