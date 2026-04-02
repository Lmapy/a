#!/usr/bin/env python3
"""TomTrades CBR Strategy - Separate Backtest Runner.

Implements the CBR (Candle Behavior Reversal) model for gold scalping.
Based on @itstomtrades methodology: 78% WR, 2.12 R:R.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DATA_DIR, RESULTS_DIR, RULES, CONTRACT, PARAMS
from data.download_data import load_data
from data.resample import build_multi_timeframe, tag_sessions
from strategy.cbr import generate_cbr_signals, CBRSignal, CBRDirection
from risk.position_sizing import calculate_position_size, calculate_trade_pnl, calculate_adaptive_risk
from risk.prop_firm_rules import PropFirmTracker, EvaluationStatus
from backtest.metrics import calculate_metrics, print_metrics
from backtest.monte_carlo import run_monte_carlo, run_block_bootstrap, print_monte_carlo
from backtest.engine import BacktestResult, TradeRecord


def run_cbr_backtest(
    candles_1min: pd.DataFrame,
    candles_1h: pd.DataFrame,
    max_days: int = 22,
) -> BacktestResult:
    """Run CBR strategy backtest."""
    result = BacktestResult()
    tracker = result.tracker

    # Generate CBR signals
    signals = generate_cbr_signals(candles_1min, candles_1h)

    if not signals:
        print("[CBR BACKTEST] No signals generated")
        tracker.end_evaluation()
        return result

    # Simulate
    opens = candles_1min["open"].values
    closes = candles_1min["close"].values
    highs = candles_1min["high"].values
    lows = candles_1min["low"].values
    timestamps = candles_1min.index

    if candles_1min.index.tz is None:
        ct_idx = candles_1min.index.tz_localize("UTC").tz_convert("US/Central")
    else:
        ct_idx = candles_1min.index.tz_convert("US/Central")
    trading_dates = ct_idx.date

    current_date = None
    open_position = None
    signal_idx = 0

    print(f"[CBR BACKTEST] Simulating {len(signals)} signals...")

    for i in range(len(candles_1min)):
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
            signal, contracts, _ = open_position
            is_long = signal.direction == CBRDirection.LONG

            hit_sl = False
            hit_tp = False

            if is_long:
                if lows[i] <= signal.stop_loss:
                    hit_sl = True
                if highs[i] >= signal.take_profit:
                    hit_tp = True
            else:
                if highs[i] >= signal.stop_loss:
                    hit_sl = True
                if lows[i] <= signal.take_profit:
                    hit_tp = True

            # When both SL and TP hit on same candle, use OHLC sequence to determine which hit first
            if hit_sl and hit_tp:
                o = opens[i]
                if is_long:
                    # Long: SL is below, TP is above
                    # If open is closer to SL (opened low), SL likely hit first
                    # If open is closer to TP (opened high), TP likely hit first
                    if o <= signal.entry_price:
                        # Opened near/below entry -> went down first -> SL
                        hit_tp = False
                    else:
                        # Opened above entry -> went up first -> TP
                        hit_sl = False
                else:
                    # Short: SL is above, TP is below
                    if o >= signal.entry_price:
                        # Opened near/above entry -> went up first -> SL
                        hit_tp = False
                    else:
                        # Opened below entry -> went down first -> TP
                        hit_sl = False

            if hit_sl:
                pnl = calculate_trade_pnl(
                    signal.entry_price, signal.stop_loss, contracts, is_long)
                result.trades.append(TradeRecord(
                    entry_time=signal.timestamp, exit_time=timestamps[i],
                    direction=signal.direction.value,
                    entry_price=signal.entry_price, exit_price=signal.stop_loss,
                    stop_loss=signal.stop_loss, take_profit=signal.take_profit,
                    contracts=contracts, pnl=pnl, exit_reason="stop_loss",
                    confluence_score=0, confluence_details=signal.details,
                    trading_day=tracker.trading_days,
                ))
                can_continue = tracker.record_trade(pnl)
                open_position = None
                if not can_continue:
                    break

            elif hit_tp:
                pnl = calculate_trade_pnl(
                    signal.entry_price, signal.take_profit, contracts, is_long)
                result.trades.append(TradeRecord(
                    entry_time=signal.timestamp, exit_time=timestamps[i],
                    direction=signal.direction.value,
                    entry_price=signal.entry_price, exit_price=signal.take_profit,
                    stop_loss=signal.stop_loss, take_profit=signal.take_profit,
                    contracts=contracts, pnl=pnl, exit_reason="take_profit",
                    confluence_score=0, confluence_details=signal.details,
                    trading_day=tracker.trading_days,
                ))
                can_continue = tracker.record_trade(pnl)
                open_position = None
                if not can_continue:
                    break

            if open_position is not None:
                result.equity_curve.append(tracker.balance)
                continue

        # Check for new signals
        if open_position is None and tracker.can_trade():
            while signal_idx < len(signals) and signals[signal_idx].index <= i:
                if signals[signal_idx].index == i:
                    signal = signals[signal_idx]
                    adaptive_risk = calculate_adaptive_risk(
                        tracker.cumulative_pnl, tracker.trading_days)
                    contracts = calculate_position_size(
                        abs(signal.entry_price - signal.stop_loss), adaptive_risk)
                    if contracts > 0:
                        open_position = (signal, contracts, tracker.trading_days)
                        signal_idx += 1
                        break
                signal_idx += 1

        result.equity_curve.append(tracker.balance)

    tracker.end_evaluation()
    result.tracker = tracker
    print(f"[CBR BACKTEST] Complete: {len(result.trades)} trades, "
          f"Status: {tracker.status}, P&L: ${tracker.cumulative_pnl:.2f}")
    return result


def run_cbr_walkforward(
    candles_1min: pd.DataFrame,
    candles_1h: pd.DataFrame,
    window_days: int = 22,
    step_days: int = 5,
) -> list[BacktestResult]:
    """Run CBR walkforward analysis."""
    results = []

    if candles_1min.index.tz is None:
        ct_idx = candles_1min.index.tz_localize("UTC").tz_convert("US/Central")
    else:
        ct_idx = candles_1min.index.tz_convert("US/Central")

    if candles_1h.index.tz is None:
        h1_ct = candles_1h.index.tz_localize("UTC").tz_convert("US/Central")
    else:
        h1_ct = candles_1h.index.tz_convert("US/Central")

    all_dates = sorted(set(ct_idx.date))

    for start_idx in range(0, len(all_dates) - window_days, step_days):
        start_date = all_dates[start_idx]
        end_date = all_dates[min(start_idx + window_days, len(all_dates) - 1)]

        mask_5 = (ct_idx.date >= start_date) & (ct_idx.date <= end_date)
        window_5 = candles_1min[mask_5]

        mask_h1 = (h1_ct.date >= start_date) & (h1_ct.date <= end_date)
        window_h1 = candles_1h[mask_h1]

        if len(window_5) < 100 or len(window_h1) < 20:
            continue

        result = run_cbr_backtest(window_5, window_h1, max_days=window_days)
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description="TomTrades CBR Gold Strategy")
    parser.add_argument("--walkforward", action="store_true")
    parser.add_argument("--mc-sims", type=int, default=10_000)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  TOMTRADES CBR STRATEGY - GOLD SCALPING")
    print("=" * 60)

    # Load data
    raw = load_data()
    raw.attrs["_is_ohlcv"] = True
    candle_dict = build_multi_timeframe(raw)
    c1m = candle_dict["5min"]   # 5-min execution timeframe
    c1h = candle_dict["1H"]     # 1H bias timeframe

    print(f"[DATA] 5-min: {len(c1m):,} candles (execution)")
    print(f"[DATA] 1H:    {len(c1h):,} candles (bias)")

    if args.walkforward:
        # Walkforward analysis
        print("\n[WALKFORWARD] Running CBR walkforward...")
        results = run_cbr_walkforward(c1m, c1h)

        passed = sum(1 for r in results if r.tracker.status == EvaluationStatus.PASSED)
        profitable = sum(1 for r in results if r.tracker.cumulative_pnl > 0)
        busted = sum(1 for r in results if r.tracker.status == "failed_drawdown")
        pnls = [r.tracker.cumulative_pnl for r in results]

        print(f"\n{'=' * 60}")
        print(f"  CBR WALKFORWARD RESULTS")
        print(f"{'=' * 60}")
        print(f"  Windows:     {len(results)}")
        print(f"  PASSED:      {passed}/{len(results)} ({passed/len(results)*100:.1f}%)")
        print(f"  Profitable:  {profitable}/{len(results)} ({profitable/len(results)*100:.1f}%)")
        print(f"  Busted:      {busted}/{len(results)} ({busted/len(results)*100:.1f}%)")
        if pnls:
            print(f"  Mean P&L:    ${np.mean(pnls):.0f}")
            print(f"  Median P&L:  ${np.median(pnls):.0f}")
            print(f"  90th pctile: ${np.percentile(pnls, 90):.0f}")
        print(f"{'=' * 60}")

        if results:
            result = results[-1]
        else:
            return
    else:
        # Single backtest
        result = run_cbr_backtest(c1m, c1h)

    metrics = calculate_metrics(result)
    print_metrics(metrics)

    if result.trades:
        # Print trade log
        print(f"\n{'#':>3} {'Dir':>5} {'Entry':>10} {'Exit':>10} {'SL':>10} "
              f"{'TP':>10} {'Ctrs':>5} {'P&L':>10} {'Reason':>12}")
        print("-" * 85)
        for idx, t in enumerate(result.trades[:30]):
            print(f"{idx+1:>3} {t.direction:>5} {t.entry_price:>10.2f} {t.exit_price:>10.2f} "
                  f"{t.stop_loss:>10.2f} {t.take_profit:>10.2f} {t.contracts:>5} "
                  f"${t.pnl:>9.2f} {t.exit_reason:>12}")

        # Monte Carlo
        trade_pnls = [t.pnl for t in result.trades]
        mc = run_monte_carlo(trade_pnls, n_simulations=args.mc_sims,
                             trades_per_day=metrics.get("trades_per_day", 1.0))
        print_monte_carlo(mc, "CBR Trade Bootstrap")

    print(f"\n{'=' * 60}")
    print(f"  Strategy: TomTrades CBR (Candle Behavior Reversal)")
    print(f"  Contract: {CONTRACT.name} (${CONTRACT.tick_value}/tick)")
    print(f"  Data:     Real 1-min XAUUSD (FutureSharks/Oanda)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
