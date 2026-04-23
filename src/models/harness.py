from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Fold:
    index: int
    train: pd.DataFrame
    val: pd.DataFrame
    is_final: bool


def generate_folds(df: pd.DataFrame, train_window: int, step_size: int) -> list[Fold]:
    """Produce sliding walk-forward folds from a date-sorted global dataset.

    Each fold has a fixed-size training window and a validation window equal
    to step_size. No temporal leakage: val dates are strictly after train dates.

    Args:
        df: Global feature DataFrame sorted ascending by date.
        train_window: Number of trading days in each training window.
        step_size: Number of trading days in each validation window (= step).

    Returns:
        List of Fold objects. Empty list if fewer than train_window + step_size rows exist.
    """
    dates = df["date"].unique()
    dates = sorted(dates)
    n = len(dates)

    if n < train_window + step_size:
        logger.warning(
            "not enough dates (%d) for even one fold (need %d)",
            n, train_window + step_size,
        )
        return []

    folds: list[Fold] = []
    start = 0
    while start + train_window + step_size <= n:
        train_dates = set(dates[start: start + train_window])
        val_dates = set(dates[start + train_window: start + train_window + step_size])
        train_df = df[df["date"].isin(train_dates)].reset_index(drop=True)
        val_df = df[df["date"].isin(val_dates)].reset_index(drop=True)
        folds.append(Fold(index=len(folds), train=train_df, val=val_df, is_final=False))
        start += step_size

    if folds:
        folds[-1].is_final = True
    logger.info("generated %d walk-forward folds", len(folds))
    return folds
