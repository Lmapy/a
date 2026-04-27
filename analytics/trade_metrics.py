"""Aggregate metrics over a list of Trade objects.

Computes the basic profit/loss/risk numbers AND trade-level diagnostics
(MAE, MFE, time-to-TP/SL distributions, entry efficiency, near-miss
TPs, consecutive-loss patterns, biggest-trade dependence).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from core.constants import H4_BARS_PER_YEAR
from core.types import Trade


def _safe(x: float, fallback: float = 0.0) -> float:
    return float(x) if math.isfinite(x) else fallback


def basic(trades: list[Trade]) -> dict:
    if not trades:
        return {"trades": 0}
    rets = np.array([t.ret for t in trades], dtype=float)
    pnls = np.array([t.pnl for t in trades], dtype=float)
    eq = np.cumprod(1 + rets)
    wins = int((rets > 0).sum())
    losses = int((rets < 0).sum())
    gross_win = float(pnls[pnls > 0].sum()) if (pnls > 0).any() else 0.0
    gross_loss = float(-pnls[pnls < 0].sum()) if (pnls < 0).any() else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0
    sd = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    sharpe = (rets.mean() / sd) * math.sqrt(H4_BARS_PER_YEAR) if sd > 0 else 0.0
    peak = np.maximum.accumulate(eq) if len(eq) else np.array([1.0])
    dd = float((eq / peak - 1.0).min()) if len(eq) else 0.0
    biggest_share = float(abs(pnls).max() / abs(pnls).sum()) if abs(pnls).sum() > 0 else 0.0
    return {
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / len(rets), 4),
        "avg_ret_bp": round(float(rets.mean() * 10_000), 2),
        "median_ret_bp": round(float(np.median(rets) * 10_000), 2),
        "total_return": round(float(eq[-1] - 1.0), 4),
        "sharpe_ann": round(_safe(sharpe), 3),
        "profit_factor": round(_safe(pf), 3),
        "max_drawdown": round(dd, 4),
        "biggest_trade_share": round(biggest_share, 4),
    }


def excursion(trades: list[Trade]) -> dict:
    """MAE / MFE distributions and entry efficiency.

    entry_efficiency = (entry distance to MFE) / (range from MAE to MFE)
    Higher = entered closer to the worst point of the trade (good).
    """
    if not trades:
        return {}
    mae = np.array([t.mae for t in trades], dtype=float)
    mfe = np.array([t.mfe for t in trades], dtype=float)
    rng = mae + mfe
    with np.errstate(divide="ignore", invalid="ignore"):
        eff = np.where(rng > 0, mae / rng, 0.0)  # MAE share of total swing
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_arr = mae / np.where(mfe > 0, mfe, np.nan)
    ratio_mean = float(np.nanmean(ratio_arr)) if np.isfinite(ratio_arr).any() else 0.0
    return {
        "mae_mean": round(float(mae.mean()), 4),
        "mae_p90": round(float(np.percentile(mae, 90)), 4),
        "mfe_mean": round(float(mfe.mean()), 4),
        "mfe_p90": round(float(np.percentile(mfe, 90)), 4),
        "mae_to_mfe_ratio_mean": round(ratio_mean, 4),
        "entry_efficiency_mean": round(float(eff.mean()), 4),  # 0 = perfect entry, 1 = catastrophic
    }


def timings(trades: list[Trade]) -> dict:
    if not trades:
        return {}
    tp = np.array([t.time_to_tp_min for t in trades if t.time_to_tp_min is not None], dtype=float)
    sl = np.array([t.time_to_sl_min for t in trades if t.time_to_sl_min is not None], dtype=float)
    near = int(sum(1 for t in trades if t.near_miss_tp))
    return {
        "time_to_tp_mean_min": round(float(tp.mean()), 1) if len(tp) else None,
        "time_to_tp_median_min": round(float(np.median(tp)), 1) if len(tp) else None,
        "time_to_sl_mean_min": round(float(sl.mean()), 1) if len(sl) else None,
        "time_to_sl_median_min": round(float(np.median(sl)), 1) if len(sl) else None,
        "near_miss_tp_count": near,
        "near_miss_tp_rate": round(near / len(trades), 4),
    }


def loss_patterns(trades: list[Trade]) -> dict:
    if not trades:
        return {}
    rets = np.array([t.ret for t in trades])
    losses = (rets <= 0).astype(int)
    # max consecutive loss streak
    max_streak = cur = 0
    for x in losses:
        cur = cur + 1 if x else 0
        max_streak = max(max_streak, cur)
    # worst N-trade rolling drawdown in % terms
    worst_window_3 = float(pd.Series(rets).rolling(3).sum().min()) if len(rets) >= 3 else 0.0
    worst_window_5 = float(pd.Series(rets).rolling(5).sum().min()) if len(rets) >= 5 else 0.0
    return {
        "max_consec_losses": int(max_streak),
        "worst_3trade_window_ret": round(worst_window_3, 4),
        "worst_5trade_window_ret": round(worst_window_5, 4),
    }


def all_metrics(trades: list[Trade], window_weeks: float | None = None) -> dict:
    out = {**basic(trades), **excursion(trades), **timings(trades), **loss_patterns(trades)}
    if window_weeks and window_weeks > 0:
        out["trades_per_week"] = round(len(trades) / window_weeks, 2)
    return out
