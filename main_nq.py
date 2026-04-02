#!/usr/bin/env python3
"""NQ Mean-Reversion + Morning Momentum - Runner."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from config import DATA_DIR, RESULTS_DIR, PARAMS
from instruments import NQ as NQ_INST
from data.resample import tag_sessions
from strategy.nq_strategy import generate_nq_signals, NQSignal, NQDirection
from risk.position_sizing import calculate_adaptive_risk
from risk.prop_firm_rules import PropFirmTracker, EvaluationStatus
from backtest.engine import BacktestResult, TradeRecord
from backtest.metrics import calculate_metrics, print_metrics


def nq_position_size(sl_distance_points, risk_dollars):
    """MNQ: $0.50/tick, $0.25 tick size, $2.00/point."""
    dollar_per_point = 2.0  # per MNQ contract
    risk_per_contract = sl_distance_points * dollar_per_point
    if risk_per_contract <= 0:
        return 0
    return max(1, min(int(risk_dollars / risk_per_contract), 50))


def nq_pnl(entry, exit_price, contracts, is_long, exit_is_sl):
    """Calculate MNQ P&L with realistic slippage."""
    tick = 0.25
    commission = 0.62 * 2 * contracts  # $0.62/side per MNQ
    dollar_per_point = 2.0

    # Entry: market order, 1 tick slippage
    eff_entry = entry + tick if is_long else entry - tick

    if exit_is_sl:
        sl_slip = 2 * tick  # 2 ticks SL slippage
        eff_exit = (exit_price - sl_slip) if is_long else (exit_price + sl_slip)
    else:
        eff_exit = exit_price  # limit TP

    if is_long:
        raw = (eff_exit - eff_entry) * dollar_per_point * contracts
    else:
        raw = (eff_entry - eff_exit) * dollar_per_point * contracts

    return raw - commission


def run_nq_backtest(candles_5min, max_days=22):
    result = BacktestResult()
    tracker = result.tracker

    signals = generate_nq_signals(candles_5min)
    if not signals:
        tracker.end_evaluation()
        return result

    closes = candles_5min["close"].values
    highs = candles_5min["high"].values
    lows = candles_5min["low"].values
    timestamps = candles_5min.index

    ct = candles_5min.index.tz_localize("UTC").tz_convert("US/Central") if candles_5min.index.tz is None else candles_5min.index.tz_convert("US/Central")
    dates = ct.date

    current_date = None
    open_pos = None
    si = 0

    for i in range(len(candles_5min)):
        d = dates[i]
        if d != current_date:
            current_date = d
            tracker.start_new_day()
            if tracker.status != EvaluationStatus.ACTIVE or tracker.trading_days >= max_days:
                break

        if open_pos:
            sig, ctrs = open_pos
            is_long = sig.direction == NQDirection.LONG
            hit_sl = (lows[i] <= sig.stop_loss) if is_long else (highs[i] >= sig.stop_loss)
            hit_tp = (highs[i] >= sig.take_profit) if is_long else (lows[i] <= sig.take_profit)
            if hit_sl and hit_tp:
                hit_tp = False

            if hit_sl:
                pnl = nq_pnl(sig.entry_price, sig.stop_loss, ctrs, is_long, True)
                result.trades.append(TradeRecord(sig.timestamp, timestamps[i], sig.direction.value,
                    sig.entry_price, sig.stop_loss, sig.stop_loss, sig.take_profit,
                    ctrs, pnl, "stop_loss", 0, sig.details, tracker.trading_days))
                tracker.record_trade(pnl)
                open_pos = None
                if tracker.status != EvaluationStatus.ACTIVE: break
            elif hit_tp:
                pnl = nq_pnl(sig.entry_price, sig.take_profit, ctrs, is_long, False)
                result.trades.append(TradeRecord(sig.timestamp, timestamps[i], sig.direction.value,
                    sig.entry_price, sig.take_profit, sig.stop_loss, sig.take_profit,
                    ctrs, pnl, "take_profit", 0, sig.details, tracker.trading_days))
                tracker.record_trade(pnl)
                open_pos = None
                if tracker.status != EvaluationStatus.ACTIVE: break
            if open_pos:
                result.equity_curve.append(tracker.balance)
                continue

        if not open_pos and tracker.can_trade():
            while si < len(signals) and signals[si].index <= i:
                if signals[si].index == i:
                    sig = signals[si]
                    risk = calculate_adaptive_risk(tracker.cumulative_pnl, tracker.trading_days)
                    ctrs = nq_position_size(sig.sl_distance, risk)
                    if ctrs > 0:
                        open_pos = (sig, ctrs)
                        si += 1; break
                si += 1

        result.equity_curve.append(tracker.balance)

    tracker.end_evaluation()
    return result


def run_nq_walkforward(c5, window_days=22, step_days=22):
    results = []
    ct = c5.index.tz_localize("UTC").tz_convert("US/Central") if c5.index.tz is None else c5.index.tz_convert("US/Central")
    all_dates = sorted(set(ct.date))
    for si in range(0, len(all_dates) - window_days, step_days):
        sd, ed = all_dates[si], all_dates[min(si + window_days, len(all_dates) - 1)]
        mask = (ct.date >= sd) & (ct.date <= ed)
        w = c5[mask]
        if len(w) < 100: continue
        results.append(run_nq_backtest(w, window_days))
    return results


if __name__ == "__main__":
    import config
    config.PARAMS.risk_per_trade = 400

    # Load NQ data
    nq = pd.read_parquet(DATA_DIR / "NQ_1min.parquet")
    nq_trending = nq['2019-06-01':'2020-05-14']

    c5 = nq_trending.resample("5min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna(subset=["open"])

    print("=" * 60)
    print("  NQ MEAN-REVERSION + MORNING MOMENTUM")
    print("=" * 60)
    print(f"NQ 5-min: {len(c5):,} candles")
    print(f"Price: ${c5['close'].min():.0f} to ${c5['close'].max():.0f}")

    # Single test
    result = run_nq_backtest(c5)
    m = calculate_metrics(result)
    print_metrics(m)

    if result.trades:
        morning = [t for t in result.trades if "MORNING" in str(t.confluence_details.get("strategy", ""))]
        mr = [t for t in result.trades if "MR" in str(t.confluence_details.get("strategy", ""))]
        print(f"\n  Morning: {len(morning)} trades, MR: {len(mr)} trades")

    # Walkforward
    results = run_nq_walkforward(c5)
    p = sum(1 for r in results if r.tracker.status == EvaluationStatus.PASSED)
    prof = sum(1 for r in results if r.tracker.cumulative_pnl > 0)
    bust = sum(1 for r in results if r.tracker.status == "failed_drawdown")
    n = len(results)
    pnls = [r.tracker.cumulative_pnl for r in results]
    trades = [len(r.trades) for r in results]

    print(f"\n{'=' * 60}")
    print(f"  NQ WALKFORWARD RESULTS")
    print(f"{'=' * 60}")
    print(f"  Windows:    {n}")
    print(f"  PASSED:     {p}/{n} ({p/n*100:.1f}%)")
    print(f"  Profitable: {prof}/{n} ({prof/n*100:.1f}%)")
    print(f"  Busted:     {bust}/{n} ({bust/n*100:.1f}%)")
    print(f"  Mean P&L:   ${np.mean(pnls):.0f}")
    print(f"  Median:     ${np.median(pnls):.0f}")
    print(f"  Avg trades: {np.mean(trades):.1f}")
    print(f"{'=' * 60}")
