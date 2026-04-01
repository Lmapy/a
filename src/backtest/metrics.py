from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.models import TradeRecord


def compute_metrics(trades: list[TradeRecord], daily_pnls: list[float] | None = None) -> dict:
    """Compute comprehensive trading performance metrics."""
    if not trades:
        return {"total_trades": 0, "net_pnl": 0}

    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    net_pnl = sum(pnls)
    total_commission = sum(t.commission for t in trades)

    metrics = {
        "total_trades": len(trades),
        "net_pnl": net_pnl,
        "gross_pnl": net_pnl + total_commission,
        "total_commission": total_commission,
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": len(wins) / len(trades) if trades else 0,
        "avg_win": np.mean(wins) if wins else 0,
        "avg_loss": np.mean(losses) if losses else 0,
        "largest_win": max(pnls) if pnls else 0,
        "largest_loss": min(pnls) if pnls else 0,
        "profit_factor": abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf"),
        "expectancy": np.mean(pnls) if pnls else 0,
    }

    # Max drawdown from trade-by-trade equity
    equity = np.cumsum(pnls)
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity - running_max
    metrics["max_drawdown"] = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0

    # Daily metrics if available
    if daily_pnls and len(daily_pnls) > 1:
        daily = np.array(daily_pnls)
        metrics["trading_days"] = len(daily_pnls)
        metrics["green_days"] = int(np.sum(daily > 0))
        metrics["red_days"] = int(np.sum(daily <= 0))
        metrics["avg_daily_pnl"] = float(np.mean(daily))
        metrics["daily_pnl_std"] = float(np.std(daily))

        # Sharpe ratio (annualized, ~252 trading days)
        if np.std(daily) > 0:
            metrics["sharpe_ratio"] = float(np.mean(daily) / np.std(daily) * np.sqrt(252))
        else:
            metrics["sharpe_ratio"] = 0

        # Sortino ratio (using downside deviation)
        downside = daily[daily < 0]
        if len(downside) > 0 and np.std(downside) > 0:
            metrics["sortino_ratio"] = float(np.mean(daily) / np.std(downside) * np.sqrt(252))
        else:
            metrics["sortino_ratio"] = 0

    # Strategy breakdown
    strategy_pnl: dict[str, list[float]] = {}
    for t in trades:
        strategy_pnl.setdefault(t.strategy_name, []).append(t.pnl)

    metrics["by_strategy"] = {
        name: {
            "trades": len(pnl_list),
            "net_pnl": sum(pnl_list),
            "win_rate": sum(1 for p in pnl_list if p > 0) / len(pnl_list) if pnl_list else 0,
            "avg_pnl": np.mean(pnl_list),
        }
        for name, pnl_list in strategy_pnl.items()
    }

    return metrics


def format_metrics(metrics: dict) -> str:
    """Format metrics dict as a human-readable string."""
    lines = [
        "=" * 60,
        "PERFORMANCE METRICS",
        "=" * 60,
        f"Total Trades:      {metrics.get('total_trades', 0)}",
        f"Net P&L:           ${metrics.get('net_pnl', 0):,.2f}",
        f"Win Rate:          {metrics.get('win_rate', 0):.1%}",
        f"Avg Win:           ${metrics.get('avg_win', 0):,.2f}",
        f"Avg Loss:          ${metrics.get('avg_loss', 0):,.2f}",
        f"Profit Factor:     {metrics.get('profit_factor', 0):.2f}",
        f"Expectancy:        ${metrics.get('expectancy', 0):,.2f}",
        f"Max Drawdown:      ${metrics.get('max_drawdown', 0):,.2f}",
        f"Largest Win:       ${metrics.get('largest_win', 0):,.2f}",
        f"Largest Loss:      ${metrics.get('largest_loss', 0):,.2f}",
    ]

    if "trading_days" in metrics:
        lines.extend([
            "",
            f"Trading Days:      {metrics['trading_days']}",
            f"Green / Red Days:  {metrics.get('green_days', 0)} / {metrics.get('red_days', 0)}",
            f"Avg Daily P&L:     ${metrics.get('avg_daily_pnl', 0):,.2f}",
            f"Sharpe Ratio:      {metrics.get('sharpe_ratio', 0):.2f}",
            f"Sortino Ratio:     {metrics.get('sortino_ratio', 0):.2f}",
        ])

    if "by_strategy" in metrics:
        lines.append("")
        lines.append("BY STRATEGY:")
        for name, stats in metrics["by_strategy"].items():
            lines.append(
                f"  {name}: {stats['trades']} trades, "
                f"${stats['net_pnl']:,.2f}, "
                f"{stats['win_rate']:.1%} win rate"
            )

    lines.append("=" * 60)
    return "\n".join(lines)
