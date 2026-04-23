from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class DegradationResult:
    hit_rate: float | None
    triggered: bool
    consecutive_windows_below: int


def compute_hit_rate(predictions_df: pd.DataFrame, ohlcv_df: pd.DataFrame) -> float | None:
    """
    Hit rate = correct directional signals / total BUY+SELL signals.

    predictions_df columns: ticker, signal, run_date
    ohlcv_df columns: ticker, date, close
    """
    active = predictions_df[predictions_df["signal"].isin(["BUY", "SELL"])].copy()
    if active.empty:
        return None

    active = active.copy()
    active["run_date"] = pd.to_datetime(active["run_date"]).dt.date

    ohlcv = ohlcv_df[["ticker", "date", "close"]].copy()
    ohlcv["date"] = pd.to_datetime(ohlcv["date"]).dt.date

    # For each ticker get next-day return: close[t+1] / close[t] - 1
    ohlcv = ohlcv.sort_values(["ticker", "date"])
    ohlcv["next_close"] = ohlcv.groupby("ticker")["close"].shift(-1)
    ohlcv["return"] = ohlcv["next_close"] / ohlcv["close"] - 1
    ohlcv = ohlcv.dropna(subset=["return"])

    merged = active.merge(
        ohlcv.rename(columns={"date": "run_date"})[["ticker", "run_date", "return"]],
        on=["ticker", "run_date"],
        how="inner",
    )

    if merged.empty:
        return None

    correct = (
        ((merged["signal"] == "BUY") & (merged["return"] > 0))
        | ((merged["signal"] == "SELL") & (merged["return"] < 0))
    ).sum()

    return float(correct / len(merged))


class DegradationDetector:
    def __init__(
        self,
        hit_rate_threshold: float = 0.45,
        consecutive_windows_required: int = 2,
    ) -> None:
        self._threshold = hit_rate_threshold
        self._required = consecutive_windows_required

    def run(
        self,
        predictions_df: pd.DataFrame,
        ohlcv_df: pd.DataFrame,
        previous_consecutive_windows: int,
    ) -> DegradationResult:
        hit_rate = compute_hit_rate(predictions_df, ohlcv_df)

        if hit_rate is None or hit_rate >= self._threshold:
            return DegradationResult(
                hit_rate=hit_rate,
                triggered=False,
                consecutive_windows_below=0,
            )

        consecutive = previous_consecutive_windows + 1
        triggered = consecutive >= self._required

        return DegradationResult(
            hit_rate=hit_rate,
            triggered=triggered,
            consecutive_windows_below=consecutive,
        )
