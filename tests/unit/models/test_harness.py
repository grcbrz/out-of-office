from __future__ import annotations

import pandas as pd

from src.models.harness import generate_folds


def _df(n_dates: int, n_tickers: int = 2) -> pd.DataFrame:
    """Build a minimal global dataset with n_dates unique dates and n_tickers tickers."""
    dates = pd.bdate_range("2020-01-02", periods=n_dates).date
    rows = []
    for d in dates:
        for t in [f"T{i}" for i in range(n_tickers)]:
            rows.append({"date": d, "ticker": t, "target": "HOLD"})
    return pd.DataFrame(rows)


def test_fold_count():
    """For 500 dates, 252 train + 21 val, step 21 → floor((500-252)/21) folds."""
    df = _df(500)
    folds = generate_folds(df, train_window=252, step_size=21)
    expected = (500 - 252) // 21
    assert len(folds) == expected


def test_fold_generation_no_leakage():
    """Val dates must never appear in the corresponding train window."""
    df = _df(400)
    folds = generate_folds(df, train_window=252, step_size=21)
    for fold in folds:
        train_dates = set(fold.train["date"])
        val_dates = set(fold.val["date"])
        assert train_dates.isdisjoint(val_dates), "train/val date overlap detected"


def test_not_enough_data_returns_empty(caplog):
    df = _df(10)
    folds = generate_folds(df, train_window=252, step_size=21)
    assert folds == []


def test_last_fold_marked_final():
    df = _df(400)
    folds = generate_folds(df, train_window=252, step_size=21)
    assert folds[-1].is_final is True
    for fold in folds[:-1]:
        assert fold.is_final is False
