"""Performance metrics and prop firm pass/fail analysis."""

import numpy as np
import pandas as pd
from backtest.engine import BacktestResult, TradeRecord
from risk.prop_firm_rules import EvaluationStatus
from config import RULES


def calculate_metrics(result: BacktestResult) -> dict:
    """Calculate comprehensive performance metrics from backtest results."""
    trades = result.trades
    tracker = result.tracker

    if not trades:
        return {
            "total_trades": 0,
            "status": tracker.status,
            "cumulative_pnl": 0.0,
        }

    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    win_rate = len(wins) / len(pnls) if pnls else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf")

    # Max consecutive losses
    max_consec_loss = 0
    current_streak = 0
    for p in pnls:
        if p <= 0:
            current_streak += 1
            max_consec_loss = max(max_consec_loss, current_streak)
        else:
            current_streak = 0

    # Max drawdown from equity curve
    equity = result.equity_curve
    if equity:
        peak = equity[0]
        max_dd = 0
        for val in equity:
            peak = max(peak, val)
            dd = peak - val
            max_dd = max(max_dd, dd)
    else:
        max_dd = 0

    # Sharpe ratio (daily returns)
    daily_pnls = tracker.daily_pnl_history
    if len(daily_pnls) > 1:
        sharpe = np.mean(daily_pnls) / np.std(daily_pnls) * np.sqrt(252) if np.std(daily_pnls) > 0 else 0
        downside = [p for p in daily_pnls if p < 0]
        sortino = (
            np.mean(daily_pnls) / np.std(downside) * np.sqrt(252)
            if downside and np.std(downside) > 0 else 0
        )
    else:
        sharpe = 0
        sortino = 0

    # Trade duration
    durations = []
    for t in trades:
        if hasattr(t.entry_time, "timestamp") and hasattr(t.exit_time, "timestamp"):
            dur = (t.exit_time - t.entry_time).total_seconds() / 60
            durations.append(dur)
    avg_duration_min = np.mean(durations) if durations else 0

    return {
        "status": tracker.status,
        "passed": tracker.status == EvaluationStatus.PASSED,
        "cumulative_pnl": round(total_pnl, 2),
        "total_trades": len(trades),
        "win_rate": round(win_rate, 4),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "max_consecutive_losses": max_consec_loss,
        "max_drawdown": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "trading_days": tracker.trading_days,
        "best_day_pnl": round(tracker.best_day_pnl, 2),
        "avg_duration_min": round(avg_duration_min, 1),
        "daily_pnl_history": [round(x, 2) for x in daily_pnls],
        "trades_per_day": round(len(trades) / max(tracker.trading_days, 1), 2),
    }


def print_metrics(metrics: dict):
    """Print formatted metrics report."""
    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS")
    print("=" * 60)

    status_icon = "PASS" if metrics.get("passed") else "FAIL"
    print(f"  Status:              [{status_icon}] {metrics.get('status', 'N/A')}")
    print(f"  Cumulative P&L:      ${metrics.get('cumulative_pnl', 0):,.2f}")
    print(f"  Trading Days:        {metrics.get('trading_days', 0)}")
    print(f"  Total Trades:        {metrics.get('total_trades', 0)}")
    print(f"  Trades/Day:          {metrics.get('trades_per_day', 0):.1f}")
    print()
    print(f"  Win Rate:            {metrics.get('win_rate', 0) * 100:.1f}%")
    print(f"  Avg Win:             ${metrics.get('avg_win', 0):,.2f}")
    print(f"  Avg Loss:            ${metrics.get('avg_loss', 0):,.2f}")
    print(f"  Profit Factor:       {metrics.get('profit_factor', 0):.2f}")
    print(f"  Max Consec Losses:   {metrics.get('max_consecutive_losses', 0)}")
    print()
    print(f"  Max Drawdown:        ${metrics.get('max_drawdown', 0):,.2f}")
    print(f"  Best Day P&L:        ${metrics.get('best_day_pnl', 0):,.2f}")
    print(f"  Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  Sortino Ratio:       {metrics.get('sortino_ratio', 0):.2f}")
    print(f"  Avg Trade Duration:  {metrics.get('avg_duration_min', 0):.0f} min")
    print()

    # Prop firm checks
    print("  ── Prop Firm Checks ──")
    pnl = metrics.get("cumulative_pnl", 0)
    dd = metrics.get("max_drawdown", 0)
    best = metrics.get("best_day_pnl", 0)
    days = metrics.get("trading_days", 0)
    max_day = RULES.profit_target * RULES.consistency_pct

    print(f"  Profit Target:       ${pnl:,.2f} / ${RULES.profit_target:,.2f}  "
          f"{'OK' if pnl >= RULES.profit_target else 'NOT MET'}")
    print(f"  Max Drawdown:        ${dd:,.2f} / ${RULES.max_trailing_drawdown:,.2f}  "
          f"{'OK' if dd < RULES.max_trailing_drawdown else 'EXCEEDED'}")
    print(f"  Consistency:         ${best:,.2f} / ${max_day:,.2f}  "
          f"{'OK' if best <= max_day else 'EXCEEDED'}")
    print(f"  Trading Days:        {days} / {RULES.max_trading_days}  "
          f"{'OK' if days <= RULES.max_trading_days else 'EXCEEDED'}")
    print("=" * 60)


def print_trade_log(trades: list[TradeRecord], max_rows: int = 50):
    """Print detailed trade log."""
    print(f"\n{'#':>3} {'Dir':>5} {'Entry':>10} {'Exit':>10} {'SL':>10} "
          f"{'TP':>10} {'Ctrs':>5} {'P&L':>10} {'Reason':>12} {'Score':>5}")
    print("-" * 95)

    for idx, t in enumerate(trades[:max_rows]):
        print(f"{idx+1:>3} {t.direction:>5} {t.entry_price:>10.2f} {t.exit_price:>10.2f} "
              f"{t.stop_loss:>10.2f} {t.take_profit:>10.2f} {t.contracts:>5} "
              f"${t.pnl:>9.2f} {t.exit_reason:>12} {t.confluence_score:>5}")

    if len(trades) > max_rows:
        print(f"  ... and {len(trades) - max_rows} more trades")
