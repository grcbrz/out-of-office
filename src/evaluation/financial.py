from __future__ import annotations

import numpy as np
import pandas as pd

_TRADING_DAYS_PER_YEAR = 252
_SIGNAL_TO_DIRECTION = {"BUY": 1, "SELL": -1, "HOLD": 0}


def compute_financial_metrics(
    signals: pd.Series, forward_returns: pd.Series
) -> dict:
    """Compute Sharpe ratio, max drawdown, hit rate on BUY/SELL signals.

    forward_returns must never be used as model input — this function is for
    diagnostic evaluation only.
    """
    directions = signals.map(_SIGNAL_TO_DIRECTION).fillna(0)
    daily_returns = directions * forward_returns

    hit_rate = _hit_rate(signals, forward_returns)
    sharpe = _sharpe(daily_returns)
    max_dd = _max_drawdown(daily_returns)
    signal_dist = _signal_distribution(signals)

    return {
        "hit_rate": hit_rate,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "signal_distribution": signal_dist,
    }


def _hit_rate(signals: pd.Series, forward_returns: pd.Series) -> float:
    """% of BUY/SELL signals where direction was correct. Excludes HOLD."""
    directional = signals[signals.isin(["BUY", "SELL"])]
    if directional.empty:
        return float("nan")
    correct = (
        ((directional == "BUY") & (forward_returns[directional.index] > 0)) |
        ((directional == "SELL") & (forward_returns[directional.index] < 0))
    )
    return float(correct.mean())


def _sharpe(daily_returns: pd.Series) -> float:
    std = daily_returns.std()
    if std == 0 or pd.isna(std):
        return 0.0
    return float(daily_returns.mean() / std * np.sqrt(_TRADING_DAYS_PER_YEAR))


def _max_drawdown(daily_returns: pd.Series) -> float:
    cumulative = (1 + daily_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return float(drawdown.min())


def _signal_distribution(signals: pd.Series) -> dict[str, float]:
    if signals.empty:
        return {}
    counts = signals.value_counts(normalize=True)
    return {label: float(counts.get(label, 0.0)) for label in ["BUY", "HOLD", "SELL"]}
