from __future__ import annotations

import numpy as np
import pandas as pd


def compute_time_decay_weights(dates: pd.Series, half_life_days: int) -> np.ndarray:
    """Return exponential decay weights aligned to dates.

    The most recent date in the series receives weight 1.0. A date exactly
    half_life_days earlier receives weight 0.5. The relationship is:

        w(t) = exp(-ln(2) / half_life_days * days_ago)

    This makes recent data proportionally more influential without discarding
    older rows entirely. Useful when retraining frequently with a large fixed
    window where most rows are shared between consecutive runs.
    """
    dates_dt = pd.to_datetime(dates)
    max_date = dates_dt.max()
    days_ago = (max_date - dates_dt).dt.days.to_numpy()
    decay_rate = np.log(2) / half_life_days
    return np.exp(-decay_rate * days_ago)
