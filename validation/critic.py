"""Agent 05 — Robustness Critic.

Stacks several "try to kill the strategy" tests on top of the existing
walk-forward + statistical layers in validation/. Returns a dict with
pass/fail per test and a short explanation, so the certifier can refuse
strategies that only work because of one outlier or one good week.

Tests:
  - top_trade_dependency        remove top 1 / top 5% of trades; result must
                                stay positive
  - worst_week                  worst 7-day window must not exceed account DD
  - consecutive_loss            longest losing streak must be tolerable
  - session_split               profitable in >=2 of {asia, london, ny}
  - removed_top_year            with the single best calendar year removed,
                                strategy still positive on remainder
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.types import Trade
from regime.filters import session_label


@dataclass
class CriticResult:
    passes_critic: bool
    failure_reasons: list[str]
    detail: dict


def _total_return(rets: np.ndarray) -> float:
    if len(rets) == 0:
        return 0.0
    return float(np.cumprod(1 + rets)[-1] - 1.0)


def top_trade_dependency(trades: list[Trade], detail: dict) -> list[str]:
    fails: list[str] = []
    if not trades:
        return ["no trades to test top-trade dependency"]
    rets = np.array([t.ret for t in trades], dtype=float)
    base = _total_return(rets)
    detail["base_total_return"] = round(base, 6)

    # Drop top 1 trade by absolute return contribution.
    abs_rets = np.abs(rets)
    top1 = int(np.argmax(abs_rets))
    rets_no1 = np.delete(rets, top1)
    after_no1 = _total_return(rets_no1)
    detail["total_return_minus_top1"] = round(after_no1, 6)
    if after_no1 <= 0:
        fails.append(f"profit disappears after removing top 1 trade "
                     f"({base:.4f} -> {after_no1:.4f})")

    # Drop top 5% by abs contribution.
    n_drop = max(1, int(np.ceil(len(rets) * 0.05)))
    drop_idxs = np.argsort(abs_rets)[-n_drop:]
    rets_no5 = np.delete(rets, drop_idxs)
    after_no5 = _total_return(rets_no5)
    detail["total_return_minus_top5pct"] = round(after_no5, 6)
    detail["n_dropped_top5pct"] = int(n_drop)
    if after_no5 <= 0:
        fails.append(f"profit disappears after removing top 5% trades "
                     f"({base:.4f} -> {after_no5:.4f}, n_dropped={n_drop})")

    # Single-trade share of profit.
    pos = rets[rets > 0]
    if len(pos) > 0:
        share = float(pos.max() / pos.sum()) if pos.sum() > 0 else 0.0
        detail["biggest_winner_share_of_winners"] = round(share, 4)
        if share > 0.40:
            fails.append(f"single biggest winner is {share:.0%} of all winning "
                         "PnL — concentration risk")
    return fails


def worst_week(trades: list[Trade], detail: dict, dd_floor: float = -0.10) -> list[str]:
    """Worst 7-day window's PnL must not breach `dd_floor`."""
    if not trades:
        return []
    df = pd.DataFrame([{"t": t.exit_time, "ret": t.ret} for t in trades])
    df = df.set_index("t").sort_index()
    weekly = df["ret"].rolling("7D").sum()
    worst = float(weekly.min())
    detail["worst_7d_window_ret"] = round(worst, 4)
    if worst < dd_floor:
        return [f"worst 7-day window {worst:.4f} breaches {dd_floor:.2f} floor"]
    return []


def consecutive_loss(trades: list[Trade], detail: dict, max_streak: int = 6) -> list[str]:
    if not trades:
        return []
    rets = np.array([t.ret for t in trades])
    streak = cur = 0
    for r in rets:
        cur = cur + 1 if r <= 0 else 0
        streak = max(streak, cur)
    detail["max_consecutive_losses"] = int(streak)
    if streak > max_streak:
        return [f"longest losing streak {streak} > {max_streak} max"]
    return []


def session_split(trades: list[Trade], detail: dict) -> list[str]:
    if not trades:
        return []
    rows = []
    for t in trades:
        bucket = pd.Timestamp(t.h4_bucket) if t.h4_bucket is not None \
                 else pd.Timestamp(t.entry_time).floor("4h")
        rows.append({"session": session_label(bucket), "ret": t.ret})
    df = pd.DataFrame(rows)
    by = df.groupby("session")["ret"].sum().to_dict()
    detail["session_total_return"] = {k: round(float(v), 4) for k, v in by.items()}
    n_sessions_with_data = len(by)
    n_pos = sum(1 for v in by.values() if v > 0)
    detail["sessions_positive"] = int(n_pos)
    if n_sessions_with_data >= 2 and n_pos < 2:
        return [f"profit concentrated in 1 session "
                f"({by}); needs >=2 positive sessions"]
    return []


def removed_top_year(trades: list[Trade], detail: dict) -> list[str]:
    if not trades:
        return []
    df = pd.DataFrame([{"y": t.entry_time.year, "ret": t.ret} for t in trades])
    by_year = df.groupby("y")["ret"].sum()
    if len(by_year) < 2:
        return []
    best_y = int(by_year.idxmax())
    remainder = float(df.loc[df["y"] != best_y, "ret"].sum())
    detail["best_year"] = int(best_y)
    detail["remainder_total_return_minus_best_year"] = round(remainder, 4)
    if remainder <= 0:
        return [f"all profit comes from year {best_y}; remainder {remainder:.4f}"]
    return []


def time_segment_holdout(trades: list[Trade], detail: dict) -> list[str]:
    """Split the holdout into early/mid/late thirds; require positive
    return in >=2 of 3 segments."""
    if len(trades) < 9:
        return []
    rets = pd.Series([t.ret for t in trades])
    n = len(rets)
    bounds = [0, n // 3, 2 * n // 3, n]
    seg_returns = []
    for i, (s, e) in enumerate(zip(bounds[:-1], bounds[1:])):
        seg = rets.iloc[s:e]
        seg_ret = float((1 + seg).prod() - 1.0) if len(seg) else 0.0
        seg_returns.append(seg_ret)
    detail["segment_returns"] = [round(x, 4) for x in seg_returns]
    n_pos = sum(1 for x in seg_returns if x > 0)
    if n_pos < 2:
        return [f"only {n_pos}/3 holdout segments positive "
                f"(early={seg_returns[0]:+.4f}, mid={seg_returns[1]:+.4f}, "
                f"late={seg_returns[2]:+.4f})"]
    return []


def direction_bias(trades: list[Trade], detail: dict) -> list[str]:
    """Both long and short must contribute positive PnL (or one of them
    must have at least one trade and not be hugely negative)."""
    if not trades:
        return []
    rets_long = [t.ret for t in trades if t.direction > 0]
    rets_short = [t.ret for t in trades if t.direction < 0]
    long_total = float(np.prod([1 + r for r in rets_long]) - 1.0) if rets_long else 0.0
    short_total = float(np.prod([1 + r for r in rets_short]) - 1.0) if rets_short else 0.0
    detail["long_total_return"] = round(long_total, 4)
    detail["short_total_return"] = round(short_total, 4)
    detail["n_long_trades"] = len(rets_long)
    detail["n_short_trades"] = len(rets_short)
    fails = []
    # If we have both sides, require both not catastrophically negative.
    if rets_long and rets_short:
        if long_total < -0.05 and short_total > 0:
            fails.append(f"long-side total return {long_total:+.4f} is "
                         "materially negative — direction bias risk")
        if short_total < -0.05 and long_total > 0:
            fails.append(f"short-side total return {short_total:+.4f} is "
                         "materially negative — direction bias risk")
    return fails


def cluster_of_wins(trades: list[Trade], detail: dict) -> list[str]:
    """Detect whether all profit accumulates in a tight calendar window.
    Reject if >70% of all positive PnL lands inside a single rolling
    7-day window."""
    if not trades:
        return []
    df = pd.DataFrame([{"t": t.exit_time, "ret": t.ret} for t in trades]).set_index("t").sort_index()
    pos = df[df["ret"] > 0]
    if pos.empty:
        return []
    total_pos = float(pos["ret"].sum())
    rolling_max = pos["ret"].rolling("7D").sum().max()
    if total_pos > 0:
        share = float(rolling_max) / total_pos
    else:
        share = 0.0
    detail["cluster_max_7d_share_of_positive_pnl"] = round(share, 4)
    if share > 0.70:
        return [f">70% of positive PnL ({share:.0%}) clusters in a single "
                "rolling 7-day window — reject"]
    return []


def run_critic(trades: list[Trade], dd_floor: float = -0.10,
               max_streak: int = 6) -> CriticResult:
    detail: dict = {}
    fails: list[str] = []
    fails += top_trade_dependency(trades, detail)
    fails += worst_week(trades, detail, dd_floor=dd_floor)
    fails += consecutive_loss(trades, detail, max_streak=max_streak)
    fails += session_split(trades, detail)
    fails += removed_top_year(trades, detail)
    fails += time_segment_holdout(trades, detail)
    fails += direction_bias(trades, detail)
    fails += cluster_of_wins(trades, detail)
    return CriticResult(
        passes_critic=not fails,
        failure_reasons=fails,
        detail=detail,
    )
