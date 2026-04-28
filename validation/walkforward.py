"""Walk-forward harness. ≥20 disjoint test folds.

Slimmer than v1's: 9-month train, 3-month test, 3-month step (~28 folds
on 2018-06 -> 2026-04). The H4-only flat-cost simulation is reused so we
can evaluate hundreds of specs cheaply and gate on multi-fold stats.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analytics.trade_metrics import basic
from core.types import Spec, FoldResult


@dataclass
class WFConfig:
    train_months: int = 9
    test_months: int = 3
    step_months: int = 3
    min_folds: int = 20


def _month_floor(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts.year, ts.month, 1, tz="UTC")


def make_folds(h4: pd.DataFrame, cfg: WFConfig) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    if len(h4) == 0:
        return []
    start = _month_floor(h4["time"].iloc[0])
    end = h4["time"].iloc[-1]
    out = []
    cursor = start
    while True:
        train_end = cursor + pd.DateOffset(months=cfg.train_months)
        test_end = train_end + pd.DateOffset(months=cfg.test_months)
        if test_end > end:
            break
        out.append((cursor, train_end, train_end, test_end))
        cursor = cursor + pd.DateOffset(months=cfg.step_months)
    return out


def _slice(h4: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return h4[(h4["time"] >= start) & (h4["time"] < end)].reset_index(drop=True)


def run_h4_only(spec: Spec, h4: pd.DataFrame) -> list:
    """Reuses the v1 strategy.run_h4_sim for the walk-forward H4 simulation
    (no M15 needed). v1's runner is robust and vectorised; we treat it
    as a kernel here."""
    import scripts.strategy as v1
    s = spec.to_json()
    s["entry"] = {"type": s["entry"].get("type", "h4_open")}
    if s["entry"]["type"] == "fib_limit_entry":
        s["entry"] = {"type": "m15_retrace_fib", "level": spec.entry.get("level", 0.5)}
    elif s["entry"]["type"] == "touch_entry":
        s["entry"] = {"type": "h4_open"}
    elif s["entry"]["type"] in ("reaction_close", "minor_structure_break", "delayed_entry_1",
                                "delayed_entry_2", "sweep_reclaim", "zone_midpoint_limit"):
        # H4-only walk-forward can't replay M15 entries; treat as h4_open
        # (the M15 entry refinement is tested in the holdout where M15 is available)
        s["entry"] = {"type": "h4_open"}
    return v1.run_h4_sim(s, h4)


def walk_forward(spec: Spec, h4: pd.DataFrame, cfg: WFConfig | None = None) -> dict:
    cfg = cfg or WFConfig()
    folds = make_folds(h4, cfg)
    out_folds: list[FoldResult] = []
    for k, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        full = _slice(h4, tr_s, te_e)
        if len(full) < 30:
            continue
        trades = run_h4_only(spec, full)
        # restrict to test window
        tr = [t for t in trades if te_s <= t.entry_time < te_e]
        m = basic(tr)
        out_folds.append(FoldResult(
            fold_id=k,
            train_start=tr_s, train_end=tr_e,
            test_start=te_s, test_end=te_e,
            trades=int(m.get("trades", 0)),
            win_rate=float(m.get("win_rate", 0.0)),
            total_return=float(m.get("total_return", 0.0)),
            sharpe_ann=float(m.get("sharpe_ann", 0.0)),
            profit_factor=float(m.get("profit_factor", 0.0)),
            max_drawdown=float(m.get("max_drawdown", 0.0)),
        ))
    if not out_folds:
        return {"folds": 0, "median_sharpe": 0.0, "pct_positive_folds": 0.0,
                "fold_records": []}
    sharpes = np.array([f.sharpe_ann for f in out_folds])
    rets = np.array([f.total_return for f in out_folds])
    return {
        "folds": len(out_folds),
        "median_sharpe": round(float(np.median(sharpes)), 3),
        "pct_positive_folds": round(float((rets > 0).mean()), 4),
        "median_total_return": round(float(np.median(rets)), 4),
        "worst_fold_dd": round(float(min(f.max_drawdown for f in out_folds)), 4),
        "min_folds_required": cfg.min_folds,
        "meets_min_folds": len(out_folds) >= cfg.min_folds,
        "fold_records": [f.__dict__ for f in out_folds],
    }
