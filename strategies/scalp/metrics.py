"""Metrics + CSV / JSON writers for the CBR scalp engine."""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.scalp.config import CBRGoldScalpConfig


def compute_metrics(trades: list, setups: list) -> dict:
    """Aggregate metrics over a list of TradeRow + SetupRow."""
    n = len(trades)
    if n == 0:
        return {"total_trades": 0,
                 "n_setups_logged": len(setups),
                 "note": "no trades produced — inspect setups.csv for skip reasons"}

    rs = np.array([t.r_result for t in trades], dtype=float)
    pnls = np.array([t.pnl for t in trades], dtype=float)
    durs = np.array([t.duration_minutes for t in trades], dtype=float)
    wins = rs > 0
    losses = rs < 0

    win_rate = float(wins.mean())
    avg_r = float(rs.mean())
    median_r = float(np.median(rs))
    total_r = float(rs.sum())
    avg_win = float(rs[wins].mean()) if wins.any() else 0.0
    avg_loss = float(rs[losses].mean()) if losses.any() else 0.0
    gross_win_r = float(rs[wins].sum()) if wins.any() else 0.0
    gross_loss_r = float(-rs[losses].sum()) if losses.any() else 0.0
    pf = (gross_win_r / gross_loss_r) if gross_loss_r > 0 else \
        float("inf") if gross_win_r > 0 else 0.0
    expectancy = avg_r       # in R; equiv to win_rate*avg_win + loss_rate*avg_loss

    # consecutive streaks
    cur_w = max_w = cur_l = max_l = 0
    for r in rs:
        if r > 0:
            cur_w += 1; cur_l = 0
        elif r < 0:
            cur_l += 1; cur_w = 0
        else:
            cur_w = cur_l = 0
        max_w = max(max_w, cur_w)
        max_l = max(max_l, cur_l)

    # equity in R, peak-to-trough drawdown
    eq = np.cumsum(rs)
    peak = np.maximum.accumulate(eq) if len(eq) else np.array([0.0])
    dd = (eq - peak)
    max_dd_r = float(dd.min()) if len(dd) else 0.0

    # biggest-trade dependence
    abs_total = float(np.abs(rs).sum())
    biggest_share = float(np.abs(rs).max() / abs_total) if abs_total > 0 else 0.0

    # session / direction breakdowns
    by_dir = _by_group(trades, lambda t: "long" if t.direction > 0 else "short")
    by_dow = _by_group(trades, lambda t: t.day_of_week)
    by_trigger = _by_group(trades, lambda t: t.trigger_kind)

    # quality buckets
    qbuckets = {"0_25": [], "25_50": [], "50_75": [], "75_100": []}
    for t in trades:
        q = t.expansion_quality
        if q < 25: qbuckets["0_25"].append(t.r_result)
        elif q < 50: qbuckets["25_50"].append(t.r_result)
        elif q < 75: qbuckets["50_75"].append(t.r_result)
        else: qbuckets["75_100"].append(t.r_result)
    by_quality = {k: {"trades": len(v), "avg_r": (sum(v)/len(v) if v else 0.0)}
                   for k, v in qbuckets.items()}

    # MAE/MFE
    avg_mae = float(np.mean([t.mae for t in trades]))
    avg_mfe = float(np.mean([t.mfe for t in trades]))

    # ambiguous bars
    ambiguous_count = sum(1 for t in trades
                           if "ambiguous" in (t.exit_reason or ""))

    # setup funnel
    n_setups = len(setups)
    n_orders = sum(1 for s in setups if getattr(s, "order_placed", False))
    n_fills = sum(1 for s in setups if getattr(s, "order_filled", False))
    skip_hist: dict = {}
    for s in setups:
        r = getattr(s, "skipped_reason", "") or ""
        if r:
            skip_hist[r] = skip_hist.get(r, 0) + 1

    return {
        "total_trades": n,
        "win_rate": round(win_rate, 4),
        "avg_r": round(avg_r, 4),
        "median_r": round(median_r, 4),
        "total_r": round(total_r, 3),
        "profit_factor": round(pf, 3) if math.isfinite(pf) else "inf",
        "expectancy_r": round(expectancy, 4),
        "avg_winner_r": round(avg_win, 4),
        "avg_loser_r": round(avg_loss, 4),
        "max_drawdown_r": round(max_dd_r, 3),
        "max_consecutive_wins": max_w,
        "max_consecutive_losses": max_l,
        "biggest_trade_share": round(biggest_share, 4),
        "avg_mae": round(avg_mae, 5),
        "avg_mfe": round(avg_mfe, 5),
        "avg_duration_min": round(float(durs.mean()), 1),
        "ambiguous_same_bar_count": ambiguous_count,
        "by_direction": by_dir,
        "by_day_of_week": by_dow,
        "by_trigger": by_trigger,
        "by_expansion_quality": by_quality,
        "setups_logged": n_setups,
        "orders_placed": n_orders,
        "orders_filled": n_fills,
        "skip_reasons": skip_hist,
    }


def _by_group(trades, key) -> dict:
    out: dict = {}
    for t in trades:
        k = key(t)
        out.setdefault(k, []).append(t.r_result)
    return {k: {"trades": len(v),
                 "avg_r": round(sum(v) / len(v), 4) if v else 0.0,
                 "total_r": round(sum(v), 3)}
            for k, v in out.items()}


def write_outputs(*, results: dict, cfg: CBRGoldScalpConfig,
                   output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trades = results["trades"]
    setups = results["setups"]
    metrics = compute_metrics(trades, setups)

    pd.DataFrame([t.to_dict() for t in trades]).to_csv(
        output_dir / "trades.csv", index=False)
    pd.DataFrame([s.to_dict() for s in setups]).to_csv(
        output_dir / "setups.csv", index=False)
    cfg.write_used(output_dir / "config_used.json")
    (output_dir / "validation_report.json").write_text(
        json.dumps(results["validation"], indent=2, default=str))

    summary = {
        "name": cfg.name,
        "symbol": cfg.symbol,
        "primary_timeframe": cfg.primary_timeframe,
        "bias_timeframe": cfg.bias_timeframe,
        "n_m1_bars": results.get("n_m1_bars"),
        "n_h1_bars": results.get("n_h1_bars"),
        "runtime_s": results.get("runtime_s"),
        "metrics": metrics,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    return summary
