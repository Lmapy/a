"""Multi-year holdout segmentation (M15-aware).

Segments the long H4 series into yearly slices and re-runs the spec on
each slice using the canonical M15-aware executor (the same one
walk-forward and the holdout proper use). The 6-month warm-up window
is preserved so MA/ATR indicators stabilise before the test slice.

Strategy must be profitable in ≥ N yearly slices to pass.

Phase-2 hardening: this module previously called
`validation.walkforward.run_h4_only` which silently downgraded M15
entries to H4 open. That helper is gone. This module now goes through
`run_executor_on_window` -- same executor as walk-forward and holdout.
"""
from __future__ import annotations

import pandas as pd

from analytics.trade_metrics import basic
from validation.walkforward import run_executor_on_window, check_compatibility


def yearly_segments(spec, h4_long: pd.DataFrame, m15_long: pd.DataFrame,
                    min_positive_years: int = 3) -> dict:
    status, reason = check_compatibility(spec)
    if status != "ok":
        return {
            "yearly": [],
            "n_years": 0,
            "n_positive_years": 0,
            "regime_consistency_score": 0.0,
            "min_positive_years_required": min_positive_years,
            "passes_yearly_consistency": False,
            "compatibility": status,
            "compatibility_reason": reason,
        }

    df = h4_long.copy()
    df["year"] = df["time"].dt.year
    out = []
    for y, g in df.groupby("year"):
        if len(g) < 100:
            continue
        # warm-up: include 6 months prior so MA/ATR are populated
        warm_start = pd.Timestamp(y - 1, 7, 1, tz="UTC")
        ystart = pd.Timestamp(y, 1, 1, tz="UTC")
        yend = pd.Timestamp(y + 1, 1, 1, tz="UTC")
        h4_window = h4_long[(h4_long["time"] >= warm_start)
                            & (h4_long["time"] <  yend)].reset_index(drop=True)
        m15_window = m15_long[(m15_long["time"] >= warm_start)
                              & (m15_long["time"] <  yend)].reset_index(drop=True)
        if len(m15_window) == 0:
            continue
        trades = run_executor_on_window(spec, h4_window, m15_window)
        ytrades = [t for t in trades if ystart <= t.entry_time < yend]
        m = basic(ytrades)
        out.append({
            "year": int(y),
            "trades": int(m.get("trades", 0)),
            "win_rate": float(m.get("win_rate", 0.0)),
            "total_return": float(m.get("total_return", 0.0)),
            "sharpe_ann": float(m.get("sharpe_ann", 0.0)),
            "max_drawdown": float(m.get("max_drawdown", 0.0)),
            "profit_factor": float(m.get("profit_factor", 0.0)),
        })
    n_pos = sum(1 for r in out if r["total_return"] > 0)
    consistency = round(n_pos / len(out), 4) if out else 0.0
    return {
        "yearly": out,
        "n_years": len(out),
        "n_positive_years": n_pos,
        "regime_consistency_score": consistency,
        "min_positive_years_required": min_positive_years,
        "passes_yearly_consistency": n_pos >= min_positive_years,
        "compatibility": "ok",
    }
