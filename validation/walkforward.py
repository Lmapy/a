"""Walk-forward harness — M15-aware (Phase 2 hardening).

This was the worst hidden risk in the pipeline: the old runner silently
rewrote any M15 entry model to `h4_open`, so walk-forward tested a
strategy *different* from the one being certified. Holdout and
walk-forward must use the same executor on the same data resolutions or
"certified" means nothing.

Now:
  * Folds are built from the `research_train` slice of H4.
  * Both H4 and M15 are sliced to each fold.
  * `validation.walkforward.run_executor` calls
    `execution.executor.run(spec, h4_window, m15_window)` -- the same
    function holdout uses.
  * Entry-model + entry-timeframe compatibility is checked up front.
    If the spec's entry timeframe is not wired into the executor, the
    runner returns `status="data_unavailable"` or
    `status="resolution_limited"` -- it does NOT fall back to
    `h4_open`. Such specs cannot certify.

The 9-month train / 3-month test / 3-month step layout is unchanged;
only the inner kernel is.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analytics.trade_metrics import basic
from core.types import Spec, FoldResult
from entry_models.compatibility import compatibility_status, ENTRY_MODEL_TIMEFRAME_MAP
from execution.executor import ExecutionModel, run as run_executor


# Timeframes the executor currently consumes natively.
# H4 is the bias frame; M15 is the entry sub-frame passed into the
# entry-model registry. M5/M3/M1 are NOT yet wired through the
# executor: a spec that demands one of those will get
# `data_unavailable` rather than a silent downgrade.
SUPPORTED_ENTRY_TIMEFRAMES = {"M15"}


@dataclass
class WFConfig:
    train_months: int = 9
    test_months: int = 3
    step_months: int = 3
    min_folds: int = 20


def _month_floor(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts.year, ts.month, 1, tz="UTC")


def make_folds(h4: pd.DataFrame, cfg: WFConfig
               ) -> list[tuple[pd.Timestamp, pd.Timestamp,
                               pd.Timestamp, pd.Timestamp]]:
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


def _slice(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df.iloc[0:0].reset_index(drop=True) if df is not None else None
    return df[(df["time"] >= start) & (df["time"] < end)].reset_index(drop=True)


def check_compatibility(spec: Spec) -> tuple[str, str]:
    """Resolve entry-model compatibility against the executor's current
    timeframe support.

    Returns (status, reason). status ∈ {"ok", "data_unavailable",
    "resolution_limited", "unknown_model"}.
    """
    entry_type = spec.entry.get("type", "")
    entry_tf = getattr(spec, "entry_timeframe", "M15")
    if entry_type not in ENTRY_MODEL_TIMEFRAME_MAP:
        return "unknown_model", f"entry model '{entry_type}' not in compatibility map"
    if entry_tf not in SUPPORTED_ENTRY_TIMEFRAMES:
        return "data_unavailable", (
            f"entry_timeframe={entry_tf} requires sub-bars not yet wired "
            f"into the executor (supported: {sorted(SUPPORTED_ENTRY_TIMEFRAMES)})")
    status = compatibility_status(entry_type, entry_tf, data_available=True)
    if status != "ok":
        return status, f"entry_model={entry_type} on {entry_tf} -> {status}"
    return "ok", ""


def run_executor_on_window(spec: Spec, h4_window: pd.DataFrame,
                           m15_window: pd.DataFrame,
                           execution: ExecutionModel | None = None) -> list:
    """Run the canonical executor on a (h4, m15) slice.

    This is the SAME function holdout uses. The whole point of this
    module is to ensure walk-forward goes through this kernel unchanged.
    """
    return run_executor(spec, h4_window, m15_window,
                         execution or ExecutionModel())


def walk_forward(spec: Spec,
                 h4: pd.DataFrame,
                 m15: pd.DataFrame,
                 cfg: WFConfig | None = None,
                 execution: ExecutionModel | None = None) -> dict:
    """Walk forward through `h4` (and matching `m15` sub-bars).

    Returns a dict carrying fold metrics, certification gates, and a
    `compatibility` field describing why a spec with an unsupported
    entry timeframe was skipped.
    """
    cfg = cfg or WFConfig()
    status, reason = check_compatibility(spec)
    if status != "ok":
        return {
            "folds": 0,
            "median_sharpe": 0.0,
            "pct_positive_folds": 0.0,
            "fold_records": [],
            "compatibility": status,
            "compatibility_reason": reason,
            "min_folds_required": cfg.min_folds,
            "meets_min_folds": False,
        }

    folds = make_folds(h4, cfg)
    out_folds: list[FoldResult] = []
    for k, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        # Inner runner consumes both train and test bars (so signals + filters
        # have full lookback before the test window begins). Trades are then
        # filtered to those whose entry falls inside the test window.
        h4_full = _slice(h4, tr_s, te_e)
        m15_full = _slice(m15, tr_s, te_e)
        if h4_full is None or len(h4_full) < 30:
            continue
        if m15_full is None or len(m15_full) < 30 * 4:
            # need ~4 M15 sub-bars per H4 bar at a minimum
            continue
        trades = run_executor_on_window(spec, h4_full, m15_full, execution)
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
        return {
            "folds": 0,
            "median_sharpe": 0.0,
            "pct_positive_folds": 0.0,
            "fold_records": [],
            "compatibility": "ok",
            "min_folds_required": cfg.min_folds,
            "meets_min_folds": False,
        }

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
        "compatibility": "ok",
        "fold_records": [f.__dict__ for f in out_folds],
    }
