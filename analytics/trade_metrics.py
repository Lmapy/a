"""Aggregate metrics over a list of Trade objects.

Computes the basic profit/loss/risk numbers AND trade-level diagnostics
(MAE, MFE, time-to-TP/SL distributions, entry efficiency, near-miss
TPs, consecutive-loss patterns, biggest-trade dependence).

Phase 6 hardening: the legacy `sharpe_ann` annualised by sqrt(1560)
H4 bars per year, which over-states Sharpe by sqrt(1560/n_trades_per_year)
for sparse-trade strategies (a 200-trade-per-year system gets a ~2.8x
inflation). The dict now exposes three explicit Sharpe flavours:

  sharpe_h4_bar_ann   sqrt(H4_BARS_PER_YEAR) -- legacy, kept as
                      `sharpe_ann` for back-compat in callers that
                      have not been migrated yet
  sharpe_trade_ann    annualised by trade frequency -- correct for
                      sparse strategies
  sharpe_daily_ann    daily-equity Sharpe annualised by sqrt(252)
  sharpe_weekly_ann   weekly-equity Sharpe annualised by sqrt(52)

`sharpe_ann` continues to mean the H4-bar form so existing leaderboards
load unchanged; new code should use the explicit names. The basic dict
now also carries `time_under_water`, `expectancy_R`, `worst_day_ret`,
and `worst_week_ret`.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from core.constants import H4_BARS_PER_YEAR
from core.types import Trade

TRADING_DAYS_PER_YEAR = 252
TRADING_WEEKS_PER_YEAR = 52


def _safe(x: float, fallback: float = 0.0) -> float:
    return float(x) if math.isfinite(x) else fallback


def _equity_curve(rets: np.ndarray) -> np.ndarray:
    if len(rets) == 0:
        return np.array([1.0])
    return np.cumprod(1.0 + rets)


def _max_drawdown(eq: np.ndarray) -> float:
    if len(eq) == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


def _time_under_water_share(eq: np.ndarray) -> float:
    """Fraction of points where equity is below its prior peak."""
    if len(eq) == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    under = (eq < peak).astype(float)
    return float(under.mean())


def _trade_frequency(trades: list[Trade]) -> tuple[float, float]:
    """Return (trades_per_year, trades_per_day) on the actual span."""
    if len(trades) < 2:
        return 0.0, 0.0
    span_seconds = (pd.Timestamp(trades[-1].entry_time)
                    - pd.Timestamp(trades[0].entry_time)).total_seconds()
    if span_seconds <= 0:
        return 0.0, 0.0
    span_years = span_seconds / (365.25 * 86400.0)
    span_days = span_seconds / 86400.0
    tpy = len(trades) / span_years if span_years > 0 else 0.0
    tpd = len(trades) / span_days if span_days > 0 else 0.0
    return tpy, tpd


def _sharpe_ann_with_factor(rets: np.ndarray, factor: float) -> float:
    if len(rets) < 2:
        return 0.0
    sd = float(rets.std(ddof=1))
    if sd <= 0:
        return 0.0
    return float(rets.mean() / sd) * math.sqrt(factor)


def _daily_returns(trades: list[Trade]) -> np.ndarray:
    if not trades:
        return np.array([], dtype=float)
    df = pd.DataFrame({
        "date": [pd.Timestamp(t.entry_time).normalize() for t in trades],
        "ret": [float(t.ret) for t in trades],
    })
    return df.groupby("date")["ret"].sum().values


def _weekly_returns(trades: list[Trade]) -> np.ndarray:
    if not trades:
        return np.array([], dtype=float)
    # `to_period("W")` strips tz; do the floor manually to keep tz-aware.
    weeks = []
    for t in trades:
        ts = pd.Timestamp(t.entry_time)
        # Monday at 00:00 of the same week (UTC)
        monday = ts - pd.Timedelta(days=ts.weekday())
        weeks.append(monday.normalize())
    df = pd.DataFrame({"week": weeks, "ret": [float(t.ret) for t in trades]})
    return df.groupby("week")["ret"].sum().values


def basic(trades: list[Trade]) -> dict:
    if not trades:
        return {"trades": 0}
    rets = np.array([t.ret for t in trades], dtype=float)
    pnls = np.array([t.pnl for t in trades], dtype=float)
    eq = _equity_curve(rets)
    wins = int((rets > 0).sum())
    losses = int((rets < 0).sum())
    gross_win = float(pnls[pnls > 0].sum()) if (pnls > 0).any() else 0.0
    gross_loss = float(-pnls[pnls < 0].sum()) if (pnls < 0).any() else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0
    dd = _max_drawdown(eq)
    biggest_share = float(abs(pnls).max() / abs(pnls).sum()) if abs(pnls).sum() > 0 else 0.0

    # Sharpe variants
    sharpe_h4 = _sharpe_ann_with_factor(rets, H4_BARS_PER_YEAR)
    tpy, _ = _trade_frequency(trades)
    sharpe_trade = _sharpe_ann_with_factor(rets, tpy) if tpy > 0 else 0.0
    daily = _daily_returns(trades)
    weekly = _weekly_returns(trades)
    sharpe_daily = _sharpe_ann_with_factor(daily, TRADING_DAYS_PER_YEAR) \
                    if len(daily) >= 2 else 0.0
    sharpe_weekly = _sharpe_ann_with_factor(weekly, TRADING_WEEKS_PER_YEAR) \
                     if len(weekly) >= 2 else 0.0

    # Risk-of-ruin-style metrics
    tuw = _time_under_water_share(eq)
    worst_day = float(daily.min()) if len(daily) else 0.0
    worst_week = float(weekly.min()) if len(weekly) else 0.0

    # Expectancy in R: mean return per stop-distance proxy. We don't
    # know stop distance per-trade in general, so use the
    # in-distribution average loss as the R unit (a common convention).
    losses_arr = rets[rets < 0]
    r_unit = float(-losses_arr.mean()) if len(losses_arr) else 0.0
    expectancy_R = float(rets.mean() / r_unit) if r_unit > 0 else 0.0

    return {
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / len(rets), 4),
        "avg_ret_bp": round(float(rets.mean() * 10_000), 2),
        "median_ret_bp": round(float(np.median(rets) * 10_000), 2),
        "total_return": round(float(eq[-1] - 1.0), 4),
        # legacy alias (H4-bar annualisation); kept for backward compat
        "sharpe_ann": round(_safe(sharpe_h4), 3),
        # explicit, disambiguated names
        "sharpe_h4_bar_ann": round(_safe(sharpe_h4), 3),
        "sharpe_trade_ann": round(_safe(sharpe_trade), 3),
        "sharpe_daily_ann": round(_safe(sharpe_daily), 3),
        "sharpe_weekly_ann": round(_safe(sharpe_weekly), 3),
        "trades_per_year": round(tpy, 2),
        "n_trading_days": int(len(daily)),
        "n_trading_weeks": int(len(weekly)),
        "profit_factor": round(_safe(pf), 3),
        "max_drawdown": round(dd, 4),
        "biggest_trade_share": round(biggest_share, 4),
        "time_under_water_share": round(tuw, 4),
        "worst_day_ret": round(worst_day, 4),
        "worst_week_ret": round(worst_week, 4),
        "expectancy_R": round(_safe(expectancy_R), 4),
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
