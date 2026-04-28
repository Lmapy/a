"""Regime/filter helpers used by the executor's _apply_filters and the
report generators (which want the same definitions to label trades).

Most of these are inlined into execution/executor.py for performance,
but this module owns the canonical definitions and labelling logic
used in regime-breakdown reports.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.constants import SESSION_HOURS


def session_label(ts: pd.Timestamp) -> str:
    h = ts.hour
    if h in SESSION_HOURS["ny"]:
        return "ny"
    if h in SESSION_HOURS["london"]:
        return "london"
    if h in SESSION_HOURS["asia"]:
        return "asia"
    return "off"


def atr_percentile(h4: pd.DataFrame, window: int = 100, atr_n: int = 14) -> pd.Series:
    h, l, c = h4["high"].values, h4["low"].values, h4["close"].values
    pc = np.concatenate(([np.nan], c[:-1]))
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    atr = pd.Series(tr).rolling(atr_n, min_periods=atr_n).mean()
    return atr.rolling(window, min_periods=window).rank(pct=True)


def trend_label(h4: pd.DataFrame, ma_n: int = 50) -> pd.Series:
    ma = h4["close"].rolling(ma_n).mean().shift(1)
    prev_close = h4["close"].shift(1)
    sign = np.sign(prev_close - ma)
    return pd.Series(sign).map({1: "up", -1: "down", 0: "flat"}).fillna("na")


# `vwap_distance_z` previously lived here. Removed in Batch F: requires
# real volume the harness does not have on disk. The OHLC-only proxy
# is `atr_distance_from_session_mean` (Batch G); call sites can build
# it from the session-anchored TWAP / typical-price mean.


def regime_breakdown(trades: list, h4: pd.DataFrame) -> pd.DataFrame:
    """Tabulate trade outcome by session and trend label."""
    if not trades:
        return pd.DataFrame()
    h4i = h4.set_index("time")
    rows = []
    for t in trades:
        bucket = pd.Timestamp(t.h4_bucket) if t.h4_bucket is not None else pd.Timestamp(t.entry_time).floor("4h")
        rows.append({
            "session": session_label(bucket),
            "win": t.ret > 0,
            "ret": t.ret,
        })
    df = pd.DataFrame(rows)
    out = df.groupby("session").agg(
        trades=("ret", "size"),
        win_rate=("win", "mean"),
        avg_ret_bp=("ret", lambda s: s.mean() * 10_000),
        total_ret=("ret", "sum"),
    ).round(4).reset_index()
    return out
