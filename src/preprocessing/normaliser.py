from __future__ import annotations

import pandas as pd

from src.preprocessing.outlier import _rolling_zscore


def compute_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling z-scores for close and volume.

    This is a thin wrapper around the shared _rolling_zscore used in outlier.py.
    The zscore columns are written directly (no separate persistence of stats).
    """
    df = df.copy()
    df["close_zscore"] = _rolling_zscore(df["close"], "close")
    df["volume_zscore"] = _rolling_zscore(df["volume"].astype(float), "volume")
    return df
