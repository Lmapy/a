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


def run_critic(trades: list[Trade], dd_floor: float = -0.10,
               max_streak: int = 6) -> CriticResult:
    detail: dict = {}
    fails: list[str] = []
    fails += top_trade_dependency(trades, detail)
    fails += worst_week(trades, detail, dd_floor=dd_floor)
    fails += consecutive_loss(trades, detail, max_streak=max_streak)
    fails += session_split(trades, detail)
    fails += removed_top_year(trades, detail)
    return CriticResult(
        passes_critic=not fails,
        failure_reasons=fails,
        detail=detail,
    )
