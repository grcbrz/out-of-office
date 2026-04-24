from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.features.schema import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

_TICKER_MAP_PATH = Path("configs/ticker_map.json")

TARGET_ENCODING = {"SELL": 0, "HOLD": 1, "BUY": 2}


class TrainingDataError(Exception):
    """Raised when NaN values remain in feature data after imputation."""


class DataPreparer:
    """Handles ticker encoding, null imputation, and class weight computation."""

    def __init__(self, ticker_map_path: Path = _TICKER_MAP_PATH) -> None:
        self._ticker_map_path = ticker_map_path
        self._ticker_map: dict[str, int] = self._load_ticker_map()
        self._imputation_params: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Ticker encoding
    # ------------------------------------------------------------------

    def encode_tickers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ticker_id column using the stable ticker map. Updates map for new tickers."""
        df = df.copy()
        new_tickers = [t for t in df["ticker"].unique() if t not in self._ticker_map]
        for ticker in sorted(new_tickers):
            self._ticker_map[ticker] = len(self._ticker_map)
        self._save_ticker_map()
        df["ticker_id"] = df["ticker"].map(self._ticker_map)
        return df

    # ------------------------------------------------------------------
    # Imputation (fit on train, apply to train + val)
    # ------------------------------------------------------------------

    def fit_imputation(self, train_df: pd.DataFrame) -> None:
        """Compute median imputation params from training fold only."""
        self._imputation_params = {}
        imputable = [c for c in FEATURE_COLUMNS if c in train_df.columns and train_df[c].isna().any()]
        for col in imputable:
            median = train_df[col].median()
            # All-null column (e.g. sentiment unavailable for entire fold): default to 0.0
            self._imputation_params[col] = float(median) if pd.notna(median) else 0.0

    def apply_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted imputation params. Raises TrainingDataError if NaN remains."""
        df = df.copy()
        for col, median in self._imputation_params.items():
            if col in df.columns:
                df[col] = df[col].fillna(median)
        remaining_null_cols = [
            c for c in FEATURE_COLUMNS if c in df.columns and df[c].isna().any()
        ]
        if remaining_null_cols:
            raise TrainingDataError(
                f"NaN remains in feature columns after imputation: {remaining_null_cols}"
            )
        return df

    def get_imputation_params(self) -> dict[str, float]:
        return dict(self._imputation_params)

    # ------------------------------------------------------------------
    # Class weights
    # ------------------------------------------------------------------

    def compute_class_weights(self, targets: pd.Series) -> dict[int, float]:
        """Inverse-frequency class weights, computed per training fold."""
        counts = targets.value_counts()
        total = len(targets)
        weights: dict[int, float] = {}
        for label, encoded in TARGET_ENCODING.items():
            count = counts.get(label, 1)
            weights[encoded] = total / (len(TARGET_ENCODING) * count)
        return weights

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load_ticker_map(self) -> dict[str, int]:
        if self._ticker_map_path.exists():
            with self._ticker_map_path.open() as f:
                return json.load(f)
        return {}

    def _save_ticker_map(self) -> None:
        self._ticker_map_path.parent.mkdir(parents=True, exist_ok=True)
        with self._ticker_map_path.open("w") as f:
            json.dump(self._ticker_map, f, indent=2)
