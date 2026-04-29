from __future__ import annotations

import numpy as np
import pandas as pd

_TRADING_DAYS_PER_YEAR = 252
_SIGNAL_TO_DIRECTION = {"BUY": 1, "SELL": -1, "HOLD": 0}


def compute_financial_metrics(
    signals: pd.Series,
    forward_log_returns: pd.Series,
) -> dict:
    """Sharpe ratio, max drawdown, hit rate on BUY/SELL signals.

    Inputs use **log** forward returns (matches ``compute_forward_return`` in
    the feature layer). Sharpe and drawdown require **simple** returns to
    compound correctly — log returns can be < -1, and ``(1 + log_r).cumprod()``
    can go negative which is meaningless for portfolio-equity dynamics.

    Strategy returns are constructed as ``direction * simple_return``:
    a SELL signal flips the sign (modelled as a short position with no
    financing cost). Hit rate uses the sign of the **log** return — sign is
    invariant under the log → simple transformation so the choice is
    cosmetic.
    """
    directions = signals.map(_SIGNAL_TO_DIRECTION).fillna(0).astype(float)

    simple_returns = np.expm1(forward_log_returns.astype(float))
    strategy_returns = directions * simple_returns

    return {
        "hit_rate": _hit_rate(signals, forward_log_returns),
        "sharpe_ratio": _sharpe(strategy_returns),
        "max_drawdown": _max_drawdown(strategy_returns),
        "signal_distribution": _signal_distribution(signals),
    }


def _hit_rate(signals: pd.Series, forward_log_returns: pd.Series) -> float:
    """% of BUY/SELL signals where the realised direction was correct.
    Excludes HOLD. NaN if no BUY/SELL emitted.
    """
    directional = signals[signals.isin(["BUY", "SELL"])]
    if directional.empty:
        return float("nan")
    aligned = forward_log_returns[directional.index]
    correct = (
        ((directional == "BUY") & (aligned > 0))
        | ((directional == "SELL") & (aligned < 0))
    )
    return float(correct.mean())


def _sharpe(strategy_returns: pd.Series) -> float:
    std = strategy_returns.std()
    if std == 0 or pd.isna(std):
        return 0.0
    return float(strategy_returns.mean() / std * np.sqrt(_TRADING_DAYS_PER_YEAR))


def _max_drawdown(strategy_returns: pd.Series) -> float:
    """Worst peak-to-trough drawdown of cumulative equity (simple-return compounding).

    Returns 0.0 if no data, otherwise a value in (-1, 0].
    """
    if strategy_returns.empty:
        return 0.0
    cumulative = (1.0 + strategy_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return float(drawdown.min())


def _signal_distribution(signals: pd.Series) -> dict[str, float]:
    if signals.empty:
        return {}
    counts = signals.value_counts(normalize=True)
    return {label: float(counts.get(label, 0.0)) for label in ["BUY", "HOLD", "SELL"]}
