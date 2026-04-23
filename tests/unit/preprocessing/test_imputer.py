from __future__ import annotations

import pandas as pd
import pandas_market_calendars as mcal
import pytest

from src.preprocessing.imputer import fill_volume, forward_fill_close


def _schedule():
    cal = mcal.get_calendar("NYSE")
    return cal.schedule(start_date="2024-01-01", end_date="2024-12-31")


def _df(dates, closes, volumes=None):
    if volumes is None:
        volumes = [1000.0] * len(dates)
    return pd.DataFrame({
        "ticker": ["AAPL"] * len(dates),
        "date": pd.to_datetime(dates).date,
        "open": [100.0] * len(dates),
        "high": [110.0] * len(dates),
        "low": [90.0] * len(dates),
        "close": closes,
        "volume": volumes,
    })


def test_forward_fill_within_limit():
    df = _df(["2024-01-02", "2024-01-03", "2024-01-04"],
             [100.0, None, 102.0])
    result = forward_fill_close(df, _schedule())
    assert len(result) == 3
    assert result.loc[result["date"] == pd.Timestamp("2024-01-03").date(), "close"].iloc[0] == 100.0
    assert result.loc[result["date"] == pd.Timestamp("2024-01-03").date(), "imputed_close"].iloc[0] == True


def test_forward_fill_exceeds_limit_drops_row(caplog):
    """Three consecutive missing trading days exceeds the 2-day limit; rows dropped."""
    df = _df(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"],
        [100.0, None, None, None, 105.0],
    )
    import logging
    with caplog.at_level(logging.ERROR):
        result = forward_fill_close(df, _schedule())
    # The third consecutive fill (2024-01-05) should be dropped
    dates_in_result = list(result["date"])
    assert pd.Timestamp("2024-01-05").date() not in dates_in_result


def test_volume_zero_fill():
    df = _df(["2024-01-02", "2024-01-03"], [100.0, 101.0], [1000.0, None])
    result = fill_volume(df)
    assert result.loc[1, "volume"] == 0
    assert result.loc[1, "imputed_volume"] == True


def test_volume_present_not_flagged():
    df = _df(["2024-01-02"], [100.0], [5000.0])
    result = fill_volume(df)
    assert result.loc[0, "imputed_volume"] == False
