from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Fold:
    index: int
    train: pd.DataFrame
    val: pd.DataFrame
    is_final: bool

    def __del__(self):
        """Ensure DataFrames are cleaned up when Fold is garbage collected."""
        if hasattr(self, 'train'):
            del self.train
        if hasattr(self, 'val'):
            del self.val


def validate_features(df: pd.DataFrame, target_cols: Optional[list] = None) -> pd.DataFrame:
    """Validate and clean feature DataFrame to prevent NaN warnings.

    Column dropping is restricted to columns that are NOT in FEATURE_COLUMNS.
    Expected feature columns — including sentiment value columns that are
    legitimately sparse early in data collection — are always preserved so
    that fit_imputation() can handle them correctly (imputing to 0.0 when
    an entire training fold has no sentiment data).

    Dropping expected columns here caused the 7 sentiment value columns
    (bullish_percent, bearish_percent, company_news_score, article_count,
    positive_insights, negative_insights, neutral_insights) to be silently
    removed from every training fold, so the model never learned to use them.
    """
    if df.empty:
        return df

    from src.features.schema import FEATURE_COLUMNS
    _protected = frozenset(FEATURE_COLUMNS)

    # Drop only UNEXPECTED columns (not in FEATURE_COLUMNS) that are all NaN
    # or mostly NaN. Never drop expected feature columns regardless of null rate.
    unexpected_cols = [c for c in df.columns if c not in _protected]
    if unexpected_cols:
        unexpected_df = df[unexpected_cols]
        threshold = len(df) * 0.5
        cols_to_drop = [
            c for c in unexpected_cols
            if df[c].isna().all() or df[c].notna().sum() < threshold
        ]
        if cols_to_drop:
            logger.debug("dropping %d unexpected all-null/sparse columns: %s",
                         len(cols_to_drop), cols_to_drop)
            df = df.drop(columns=cols_to_drop)

    # Remove rows with any NaN in target columns (if specified)
    if target_cols:
        existing_targets = [col for col in target_cols if col in df.columns]
        if existing_targets:
            before_count = len(df)
            df = df.dropna(subset=existing_targets)
            if len(df) < before_count:
                logger.warning("removed %d rows with NaN in target columns",
                                before_count - len(df))

    return df


def generate_folds(df: pd.DataFrame, train_window: int, step_size: int) -> list[Fold]:
    """Produce sliding walk-forward folds from a date-sorted global dataset.

    Memory-optimized version - preserves original behavior but reduces memory usage.

    Each fold has a fixed-size training window and a validation window equal
    to step_size. No temporal leakage: val dates are strictly after train dates.

    Args:
        df: Global feature DataFrame sorted ascending by date.
        train_window: Number of trading days in each training window.
        step_size: Number of trading days in each validation window (= step).

    Returns:
        List of Fold objects. Empty list if fewer than train_window + step_size rows exist.
    """
    # Early validation to prevent NaN warnings
    if df.empty:
        logger.warning("empty DataFrame provided to generate_folds")
        return []

    # Ensure date column exists
    if "date" not in df.columns:
        logger.error("DataFrame missing 'date' column")
        return []

    # Validate data before processing
    df = validate_features(df)

    # Use numpy array for dates (more memory efficient)
    dates = df["date"].unique()
    dates = np.sort(dates)  # Ensure sorted
    n = len(dates)

    if n < train_window + step_size:
        logger.warning(
            "not enough dates (%d) for even one fold (need %d)",
            n, train_window + step_size,
        )
        return []

    folds: list[Fold] = []
    start = 0

    # Pre-convert to numpy array for faster boolean masking
    df_date_array = df["date"].values

    while start + train_window + step_size <= n:
        train_dates = set(dates[start: start + train_window])
        val_dates = set(dates[start + train_window: start + train_window + step_size])

        # Use numpy isin for faster masking (same result as pandas isin)
        train_mask = np.isin(df_date_array, list(train_dates))
        val_mask = np.isin(df_date_array, list(val_dates))

        # Create copies (necessary to avoid view issues later)
        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()

        # Skip empty folds
        if train_df.empty or val_df.empty:
            logger.warning(f"Skipping fold {len(folds)}: empty train or val")
            start += step_size
            del train_mask, val_mask, train_dates, val_dates
            continue

        folds.append(Fold(index=len(folds), train=train_df, val=val_df, is_final=False))

        start += step_size

        # Clean up masks to free memory
        del train_mask, val_mask, train_dates, val_dates

        # Periodic garbage collection during fold generation
        if len(folds) % 3 == 0:
            gc.collect()

    if folds:
        folds[-1].is_final = True

    logger.info("generated %d walk-forward folds", len(folds))
    return folds
