"""Nightly batch orchestrator.

Full pipeline wiring is added incrementally as each spec is implemented.
Currently wires: IngestionPipeline (Spec 01).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("nightly")


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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    polygon_key = os.environ.get("POLYGON_API_KEY", "")
    finnhub_key = os.environ.get("FINNHUB_API_KEY", "")

    if not polygon_key or not finnhub_key:
        logger.critical("POLYGON_API_KEY and FINNHUB_API_KEY must be set in environment")
        sys.exit(1)

    from src.ingestion.pipeline import IngestionPipeline

    logger.info("=== nightly run: %s (start_date=%s) ===", args.run_date, args.start_date)
    IngestionPipeline(polygon_key, finnhub_key).run(args.run_date, args.start_date)
    logger.info("=== ingestion complete ===")

    # Specs 02–07 will be wired here as implemented.


if __name__ == "__main__":
    main()
