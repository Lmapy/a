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
        avg_win, avg_loss, profit_factor, total_net_pnl,
        max_drawdown_dollars, max_drawdown_pct,
        sharpe_ratio, sortino_ratio,
        avg_trade_duration_s, avg_mae, avg_mfe,
        trading_days.
    """
    stats: dict[str, Any] = {}

    # ── Trade counts ───────────────────────────────────────────────────────
    total = len(trades)
    winners = [t for t in trades if t.net_pnl > 0]
    losers = [t for t in trades if t.net_pnl <= 0]

    stats["total_trades"] = total
    stats["winning_trades"] = len(winners)
    stats["losing_trades"] = len(losers)
    stats["win_rate"] = len(winners) / total if total else 0.0

    # ── PnL ───────────────────────────────────────────────────────────────
    gross_wins = sum(t.gross_pnl for t in winners)
    gross_losses = abs(sum(t.gross_pnl for t in losers))
    stats["avg_win"] = sum(t.net_pnl for t in winners) / len(winners) if winners else 0.0
    stats["avg_loss"] = sum(t.net_pnl for t in losers) / len(losers) if losers else 0.0
    stats["profit_factor"] = gross_wins / gross_losses if gross_losses else float("inf")
    stats["total_net_pnl"] = sum(t.net_pnl for t in trades)
    stats["total_commission"] = sum(t.commission for t in trades)

    # ── Drawdown from equity curve ─────────────────────────────────────────
    if equity_curve:
        equities = [eq for _, eq in equity_curve]
        peak = equities[0]
        max_dd = 0.0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_dd:
                max_dd = dd
        stats["max_drawdown_dollars"] = max_dd
        stats["max_drawdown_pct"] = max_dd / starting_balance * 100
    else:
        stats["max_drawdown_dollars"] = 0.0
        stats["max_drawdown_pct"] = 0.0

    # ── Sharpe / Sortino from equity curve ────────────────────────────────
    sharpe, sortino = _compute_risk_ratios(equity_curve, starting_balance)
    stats["sharpe_ratio"] = sharpe
    stats["sortino_ratio"] = sortino

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


def _compute_risk_ratios(
    equity_curve: list[tuple[datetime, float]],
    starting_balance: float,
) -> tuple[float, float]:
    """Compute annualised Sharpe and Sortino from the equity curve.

    Uses daily close-of-bar returns. If no data or all returns are zero,
    returns (0.0, 0.0).
    """
    if len(equity_curve) < 2:
        return 0.0, 0.0

    # Group equity values by date, take last value of each day
    daily: dict = {}
    for ts, eq in equity_curve:
        d = ts.date()
        daily[d] = eq

    sorted_days = sorted(daily.keys())
    if len(sorted_days) < 2:
        return 0.0, 0.0

    prev_eq = starting_balance
    returns: list[float] = []
    for d in sorted_days:
        eq = daily[d]
        if prev_eq != 0:
            returns.append((eq - prev_eq) / prev_eq)
        prev_eq = eq

    if not returns:
        return 0.0, 0.0

    mean_r = sum(returns) / len(returns)
    variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
    std = math.sqrt(variance) if variance > 0 else 0.0

    annualise = math.sqrt(252)

    sharpe = (mean_r / std) * annualise if std > 0 else 0.0

    down_sq = [(r - mean_r) ** 2 for r in returns if r < mean_r]
    down_std = math.sqrt(sum(down_sq) / len(down_sq)) if down_sq else 0.0
    sortino = (mean_r / down_std) * annualise if down_std > 0 else 0.0

    return round(sharpe, 4), round(sortino, 4)
