#!/usr/bin/env python3
"""Opening Range Breakout Strategy - Standalone Runner."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from config import DATA_DIR, RESULTS_DIR, PARAMS
from data.download_data import load_data
from data.resample import build_multi_timeframe, tag_sessions
from strategy.orb import generate_orb_signals, ORBSignal, ORBDirection
from risk.position_sizing import calculate_position_size, calculate_trade_pnl, calculate_adaptive_risk
from risk.prop_firm_rules import PropFirmTracker, EvaluationStatus
from backtest.engine import BacktestResult, TradeRecord
from backtest.metrics import calculate_metrics, print_metrics


def run_orb_backtest(candles_5min, max_days=22):
    """Run ORB strategy backtest."""
    result = BacktestResult()
    tracker = result.tracker

    signals = generate_orb_signals(candles_5min)
    if not signals:
        tracker.end_evaluation()
        return result

    closes = candles_5min["close"].values
    highs = candles_5min["high"].values
    lows = candles_5min["low"].values
    timestamps = candles_5min.index

    if candles_5min.index.tz is None:
        ct_idx = candles_5min.index.tz_localize("UTC").tz_convert("US/Central")
    else:
        ct_idx = candles_5min.index.tz_convert("US/Central")
    trading_dates = ct_idx.date

    current_date = None
    open_position = None
    signal_idx = 0

    for i in range(len(candles_5min)):
        candle_date = trading_dates[i]

        if candle_date != current_date:
            current_date = candle_date
            tracker.start_new_day()
            if tracker.status != EvaluationStatus.ACTIVE:
                break
            if tracker.trading_days >= max_days:
                break

        # Manage open position
        if open_position is not None:
            sig, contracts = open_position
            is_long = sig.direction == ORBDirection.LONG

            hit_sl = (lows[i] <= sig.stop_loss) if is_long else (highs[i] >= sig.stop_loss)
            hit_tp = (highs[i] >= sig.take_profit) if is_long else (lows[i] <= sig.take_profit)

            if hit_sl and hit_tp:
                hit_tp = False

            if hit_sl:
                pnl = calculate_trade_pnl(
                    sig.entry_price, sig.stop_loss, contracts, is_long,
                    exit_is_sl=True, entry_is_limit=False)
                result.trades.append(TradeRecord(
                    entry_time=sig.timestamp, exit_time=timestamps[i],
                    direction=sig.direction.value, entry_price=sig.entry_price,
                    exit_price=sig.stop_loss, stop_loss=sig.stop_loss,
                    take_profit=sig.take_profit, contracts=contracts,
                    pnl=pnl, exit_reason="stop_loss", confluence_score=0,
                    confluence_details=sig.details, trading_day=tracker.trading_days))
                tracker.record_trade(pnl)
                open_position = None
                if tracker.status != EvaluationStatus.ACTIVE:
                    break

            elif hit_tp:
                pnl = calculate_trade_pnl(
                    sig.entry_price, sig.take_profit, contracts, is_long,
                    exit_is_sl=False, entry_is_limit=False)
                result.trades.append(TradeRecord(
                    entry_time=sig.timestamp, exit_time=timestamps[i],
                    direction=sig.direction.value, entry_price=sig.entry_price,
                    exit_price=sig.take_profit, stop_loss=sig.stop_loss,
                    take_profit=sig.take_profit, contracts=contracts,
                    pnl=pnl, exit_reason="take_profit", confluence_score=0,
                    confluence_details=sig.details, trading_day=tracker.trading_days))
                tracker.record_trade(pnl)
                open_position = None
                if tracker.status != EvaluationStatus.ACTIVE:
                    break

            if open_position is not None:
                result.equity_curve.append(tracker.balance)
                continue

        # New signals
        if open_position is None and tracker.can_trade():
            while signal_idx < len(signals) and signals[signal_idx].index <= i:
                if signals[signal_idx].index == i:
                    sig = signals[signal_idx]
                    risk = calculate_adaptive_risk(tracker.cumulative_pnl, tracker.trading_days)
                    contracts = calculate_position_size(sig.sl_distance, risk)
                    if contracts > 0:
                        open_position = (sig, contracts)
                        signal_idx += 1
                        break
                signal_idx += 1

        result.equity_curve.append(tracker.balance)

    tracker.end_evaluation()
    return result


def run_orb_walkforward(candles_5min, window_days=22, step_days=22):
    results = []
    if candles_5min.index.tz is None:
        ct = candles_5min.index.tz_localize("UTC").tz_convert("US/Central")
    else:
        ct = candles_5min.index.tz_convert("US/Central")

    all_dates = sorted(set(ct.date))
    for start_idx in range(0, len(all_dates) - window_days, step_days):
        start_date = all_dates[start_idx]
        end_date = all_dates[min(start_idx + window_days, len(all_dates) - 1)]
        mask = (ct.date >= start_date) & (ct.date <= end_date)
        window = candles_5min[mask]
        if len(window) < 100:
            continue
        result = run_orb_backtest(window, max_days=window_days)
        results.append(result)
    return results


if __name__ == "__main__":
    import config
    config.PARAMS.risk_per_trade = 400

    raw = load_data()
    raw.attrs["_is_ohlcv"] = True
    cd = build_multi_timeframe(raw)
    c5 = cd["5min"]

    # Test on trending period
    trending = pd.read_parquet(DATA_DIR / "XAUUSD_1min_trending.parquet")
    trending.attrs["_is_ohlcv"] = True
    cd_t = build_multi_timeframe(trending)
    c5_t = cd_t["5min"]

    print("=" * 60)
    print("  OPENING RANGE BREAKOUT (ORB) STRATEGY")
    print("=" * 60)

    # Single test
    result = run_orb_backtest(c5_t)
    m = calculate_metrics(result)
    print_metrics(m)

    # Walkforward
    results = run_orb_walkforward(c5_t)
    p = sum(1 for r in results if r.tracker.status == EvaluationStatus.PASSED)
    prof = sum(1 for r in results if r.tracker.cumulative_pnl > 0)
    bust = sum(1 for r in results if r.tracker.status == "failed_drawdown")
    n = len(results)
    pnls = [r.tracker.cumulative_pnl for r in results]

    print(f"\n{'=' * 60}")
    print(f"  ORB WALKFORWARD (trending gold)")
    print(f"{'=' * 60}")
    print(f"  Windows:    {n}")
    print(f"  PASSED:     {p}/{n} ({p/n*100:.1f}%)")
    print(f"  Profitable: {prof}/{n} ({prof/n*100:.1f}%)")
    print(f"  Busted:     {bust}/{n} ({bust/n*100:.1f}%)")
    print(f"  Mean P&L:   ${np.mean(pnls):.0f}")
    print(f"  Median:     ${np.median(pnls):.0f}")
