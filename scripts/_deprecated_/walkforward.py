"""Rolling walk-forward harness.

Splits the long H4 history into non-overlapping test windows, runs the
strategy spec on each window, and aggregates per-fold metrics. The train
window is exposed for any future calibration step the proposer might use,
but the harness itself does not fit anything -- specs come in fully
parameterised.

Defaults:
    train_months = 12, test_months = 3, step_months = 3
    -> ~25 non-overlapping test folds across 2018-06 -> 2026-04.

Aggregate metrics:
    folds                 -- number of folds
    median_sharpe         -- median annualised Sharpe across fold tests
    pct_positive_folds    -- share of folds with positive total return
    avg_total_return      -- mean fold total return
    median_total_return   -- median fold total return
    worst_fold_dd         -- worst per-fold max drawdown
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from strategy import Trade, run_h4_sim, trades_to_metrics


@dataclass
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _month_floor(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts.year, ts.month, 1, tz="UTC")


def make_folds(
    h4: pd.DataFrame,
    train_months: int = 12,
    test_months: int = 3,
    step_months: int = 3,
) -> list[Fold]:
    if len(h4) == 0:
        return []
    start = _month_floor(h4["time"].iloc[0])
    end = h4["time"].iloc[-1]
    folds: list[Fold] = []
    cursor = start
    while True:
        train_end = cursor + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > end:
            break
        folds.append(Fold(
            train_start=cursor,
            train_end=train_end,
            test_start=train_end,
            test_end=test_end,
        ))
        cursor = cursor + pd.DateOffset(months=step_months)
    return folds


def slice_window(h4: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return h4[(h4["time"] >= start) & (h4["time"] < end)].reset_index(drop=True)


def walk_forward(spec: dict, h4: pd.DataFrame,
                 train_months: int = 12, test_months: int = 3,
                 step_months: int = 3) -> dict:
    folds = make_folds(h4, train_months, test_months, step_months)
    rows = []
    for f in folds:
        # Strategy needs warmup bars from before `test_start` for any
        # rolling indicator (ATR, MA). We feed the full slice ending at
        # test_end and then trim trades to those whose entry_time is in
        # [test_start, test_end).
        full = slice_window(h4, f.train_start, f.test_end)
        if len(full) < 30:
            continue
        trades = run_h4_sim(spec, full)
        trades = [t for t in trades if f.test_start <= t.entry_time < f.test_end]
        m = trades_to_metrics(spec.get("id", "?"), trades)
        rows.append({
            "test_start": f.test_start,
            "test_end": f.test_end,
            "trades": m["trades"],
            "win_rate": m["win_rate"],
            "total_return": m["total_return"],
            "sharpe_ann": m["sharpe_ann"],
            "max_drawdown": m["max_drawdown"],
        })

    if not rows:
        return {"folds": 0, "median_sharpe": 0.0, "pct_positive_folds": 0.0,
                "avg_total_return": 0.0, "median_total_return": 0.0,
                "worst_fold_dd": 0.0, "fold_table": pd.DataFrame()}

    df = pd.DataFrame(rows)
    return {
        "folds": int(len(df)),
        "median_sharpe": round(float(df["sharpe_ann"].median()), 3),
        "pct_positive_folds": round(float((df["total_return"] > 0).mean()), 4),
        "avg_total_return": round(float(df["total_return"].mean()), 4),
        "median_total_return": round(float(df["total_return"].median()), 4),
        "worst_fold_dd": round(float(df["max_drawdown"].min()), 4),
        "fold_table": df,
    }
