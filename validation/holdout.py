"""Multi-year holdout segmentation.

Segments the long H4 series into yearly slices and re-runs the spec on
each slice (H4-only sim). Returns per-year performance plus a regime
consistency score = share of years with positive total_return.

Strategy must be profitable in ≥ N segments (default 3) to pass.
"""
from __future__ import annotations

import pandas as pd

from analytics.trade_metrics import basic
from validation.walkforward import run_h4_only


def yearly_segments(spec, h4_long: pd.DataFrame, min_positive_years: int = 3) -> dict:
    df = h4_long.copy()
    df["year"] = df["time"].dt.year
    out = []
    for y, g in df.groupby("year"):
        if len(g) < 100:
            continue
        # warm-up: include 6 months prior so MA/ATR are populated
        warm_start = pd.Timestamp(y - 1, 7, 1, tz="UTC")
        full = h4_long[(h4_long["time"] >= warm_start) & (h4_long["time"] < pd.Timestamp(y + 1, 1, 1, tz="UTC"))]
        trades = run_h4_only(spec, full.reset_index(drop=True))
        ystart = pd.Timestamp(y, 1, 1, tz="UTC")
        yend = pd.Timestamp(y + 1, 1, 1, tz="UTC")
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
    }
