"""Trade and equity-curve statistics."""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from .results import TradeRecord


def compute_statistics(
    trades: list[TradeRecord],
    equity_curve: list[tuple[datetime, float]],
    starting_balance: float,
) -> dict[str, Any]:
    """Compute performance statistics from trades and equity curve.

    Returns a dict with keys:
        total_trades, winning_trades, losing_trades, win_rate,
        avg_win, avg_loss, largest_win, largest_loss,
        profit_factor, expectancy, total_net_pnl, total_commission,
        max_drawdown_dollars, max_drawdown_pct, avg_drawdown_dollars,
        sharpe_ratio, sortino_ratio, calmar_ratio, ulcer_index,
        max_consec_wins, max_consec_losses,
        avg_trade_duration_s, avg_mae, avg_mfe,
        trading_days.
    """
    stats: dict[str, Any] = {}

    # ── Trade counts ───────────────────────────────────────────────────────
    total = len(trades)
    winners = [t for t in trades if t.net_pnl > 0]
    losers  = [t for t in trades if t.net_pnl <= 0]

    stats["total_trades"]   = total
    stats["winning_trades"] = len(winners)
    stats["losing_trades"]  = len(losers)
    stats["win_rate"]       = len(winners) / total if total else 0.0

    # ── PnL ───────────────────────────────────────────────────────────────
    gross_wins   = sum(t.gross_pnl for t in winners)
    gross_losses = abs(sum(t.gross_pnl for t in losers))
    avg_win  = sum(t.net_pnl for t in winners) / len(winners) if winners else 0.0
    avg_loss = sum(t.net_pnl for t in losers)  / len(losers)  if losers  else 0.0

    stats["avg_win"]    = avg_win
    stats["avg_loss"]   = avg_loss
    stats["largest_win"]  = max((t.net_pnl for t in winners), default=0.0)
    stats["largest_loss"] = min((t.net_pnl for t in losers),  default=0.0)
    stats["profit_factor"]    = gross_wins / gross_losses if gross_losses else float("inf")
    stats["total_net_pnl"]    = sum(t.net_pnl for t in trades)
    stats["total_commission"]  = sum(t.commission for t in trades)

    # Expectancy: average dollar gained per trade (accounts for win rate)
    stats["expectancy"] = (
        stats["win_rate"] * avg_win + (1 - stats["win_rate"]) * avg_loss
        if total else 0.0
    )

    # ── Drawdown from equity curve ─────────────────────────────────────────
    if equity_curve:
        equities = [eq for _, eq in equity_curve]
        peak   = equities[0]
        max_dd = 0.0
        dd_sum = 0.0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = peak - eq
            dd_sum += dd
            if dd > max_dd:
                max_dd = dd
        stats["max_drawdown_dollars"] = max_dd
        stats["max_drawdown_pct"]     = max_dd / starting_balance * 100
        stats["avg_drawdown_dollars"] = dd_sum / len(equities)
    else:
        stats["max_drawdown_dollars"] = 0.0
        stats["max_drawdown_pct"]     = 0.0
        stats["avg_drawdown_dollars"] = 0.0

    # ── Risk ratios ───────────────────────────────────────────────────────
    sharpe, sortino, ann_return = _compute_risk_ratios(equity_curve, starting_balance)
    stats["sharpe_ratio"]  = sharpe
    stats["sortino_ratio"] = sortino

    # Calmar: annualised return / max drawdown (higher = better)
    max_dd = stats["max_drawdown_dollars"]
    stats["calmar_ratio"] = (
        round(ann_return / (max_dd / starting_balance), 4)
        if max_dd > 0 else float("inf")
    )

    # Ulcer index: RMS of drawdown depths (measures pain, not just peak loss)
    stats["ulcer_index"] = _ulcer_index(equity_curve)

    # ── Consecutive streak stats ───────────────────────────────────────────
    max_cw, max_cl = _consecutive_streaks(trades)
    stats["max_consec_wins"]   = max_cw
    stats["max_consec_losses"] = max_cl

    # ── Duration & excursion ──────────────────────────────────────────────
    stats["avg_trade_duration_s"] = (
        sum(t.duration_seconds for t in trades) / total if total else 0.0
    )
    stats["avg_mae"] = sum(t.mae for t in trades) / total if total else 0.0
    stats["avg_mfe"] = sum(t.mfe for t in trades) / total if total else 0.0

    # ── Trading days ──────────────────────────────────────────────────────
    if equity_curve:
        stats["trading_days"] = len({ts.date() for ts, _ in equity_curve})
    else:
        stats["trading_days"] = 0

    return stats


def _consecutive_streaks(trades: list[TradeRecord]) -> tuple[int, int]:
    """Return (max_consecutive_wins, max_consecutive_losses)."""
    max_w = max_l = cur_w = cur_l = 0
    for t in trades:
        if t.net_pnl > 0:
            cur_w += 1
            cur_l  = 0
            max_w  = max(max_w, cur_w)
        else:
            cur_l += 1
            cur_w  = 0
            max_l  = max(max_l, cur_l)
    return max_w, max_l


def _ulcer_index(equity_curve: list[tuple[datetime, float]]) -> float:
    """Ulcer index: sqrt of mean squared drawdown from rolling peak.

    Lower is better. Captures both depth and duration of drawdowns.
    """
    if len(equity_curve) < 2:
        return 0.0
    equities = [eq for _, eq in equity_curve]
    peak = equities[0]
    sq_sum = 0.0
    for eq in equities:
        if eq > peak:
            peak = eq
        pct_dd = (peak - eq) / peak * 100 if peak > 0 else 0.0
        sq_sum += pct_dd ** 2
    return round(math.sqrt(sq_sum / len(equities)), 4)


def _compute_risk_ratios(
    equity_curve: list[tuple[datetime, float]],
    starting_balance: float,
) -> tuple[float, float, float]:
    """Compute annualised Sharpe, Sortino, and CAGR from the equity curve.

    Uses daily close-of-bar returns. Returns (sharpe, sortino, ann_return).
    """
    if len(equity_curve) < 2:
        return 0.0, 0.0, 0.0

    # Group equity values by date, take last value of each day
    daily: dict = {}
    for ts, eq in equity_curve:
        daily[ts.date()] = eq

    sorted_days = sorted(daily.keys())
    if len(sorted_days) < 2:
        return 0.0, 0.0, 0.0

    prev_eq = starting_balance
    returns: list[float] = []
    for d in sorted_days:
        eq = daily[d]
        if prev_eq != 0:
            returns.append((eq - prev_eq) / prev_eq)
        prev_eq = eq

    if not returns:
        return 0.0, 0.0, 0.0

    mean_r   = sum(returns) / len(returns)
    variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
    std      = math.sqrt(variance) if variance > 0 else 0.0
    ann      = math.sqrt(252)

    sharpe = (mean_r / std) * ann if std > 0 else 0.0

    down_sq  = [(r - mean_r) ** 2 for r in returns if r < mean_r]
    down_std = math.sqrt(sum(down_sq) / len(down_sq)) if down_sq else 0.0
    sortino  = (mean_r / down_std) * ann if down_std > 0 else 0.0

    # Annualised return (CAGR-style from mean daily return)
    ann_return = mean_r * 252

    return round(sharpe, 4), round(sortino, 4), round(ann_return, 6)
