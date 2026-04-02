#!/usr/bin/env python3
"""Multi-Instrument Combined Strategy: Gold + NQ + ES + CL.

Runs the Market Structure strategy on all 4 instruments in parallel,
sharing a single prop firm tracker. More at-bats = higher pass rate.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from config import DATA_DIR, RESULTS_DIR, RULES, PARAMS
from instruments import ALL_INSTRUMENTS, InstrumentConfig
from data.resample import tag_sessions
from strategy.market_structure import detect_swing_points, detect_structure_breaks
from strategy.order_blocks import detect_order_blocks
from strategy.fvg import detect_fvgs
from strategy.liquidity import detect_liquidity_sweeps
from strategy.signals import generate_signals
from backtest.engine import BacktestResult, TradeRecord, _build_htf_trend_map
from risk.position_sizing import calculate_trade_pnl
from risk.prop_firm_rules import PropFirmTracker, EvaluationStatus
from backtest.metrics import calculate_metrics, print_metrics


def load_instrument(inst: InstrumentConfig, start_year=2019, end_year=2020):
    """Load 1-min data for an instrument and build timeframes."""
    path = DATA_DIR / inst.data_file
    if not path.exists():
        print(f"[SKIP] {inst.symbol}: no data at {path}")
        return None

    df = pd.read_parquet(path)
    df = df[f"{start_year}-06-01":f"{end_year}-05-14"]  # trending period

    if len(df) < 10000:
        print(f"[SKIP] {inst.symbol}: only {len(df)} bars")
        return None

    # Build 5-min and 15-min
    c5 = df.resample("5min").agg({"open": "first", "high": "max", "low": "min",
                                   "close": "last", "volume": "sum"}).dropna(subset=["open"])
    c15 = df.resample("15min").agg({"open": "first", "high": "max", "low": "min",
                                     "close": "last", "volume": "sum"}).dropna(subset=["open"])
    c5 = tag_sessions(c5)

    print(f"[{inst.symbol}] Loaded {len(c5):,} 5-min bars "
          f"({df.index.min().date()} to {df.index.max().date()}) "
          f"${df['close'].min():.0f}-${df['close'].max():.0f}")

    return {"1min": df, "5min": c5, "15min": c15, "inst": inst}


def generate_instrument_signals(data: dict):
    """Generate MS signals for one instrument."""
    c5 = data["5min"]
    c15 = data["15min"]
    inst = data["inst"]

    swings = detect_swing_points(c5, lookback=PARAMS.swing_lookback)
    htf_trend_map = _build_htf_trend_map(
        c5, c15, detect_swing_points(c15, lookback=PARAMS.swing_lookback))
    structure = detect_structure_breaks(c5, swings)
    obs = detect_order_blocks(c5, structure, PARAMS.ob_max_age_candles)
    fvgs = detect_fvgs(c5, PARAMS.fvg_min_size)
    sweeps = detect_liquidity_sweeps(c5, swings,
                                      tolerance=PARAMS.equal_level_tolerance,
                                      lookback=PARAMS.sweep_lookback_candles)

    signals = generate_signals(
        candles_entry=c5, swings_htf=swings,
        structure_events=structure, order_blocks=obs,
        fvgs=fvgs, sweeps=sweeps, htf_trend_map=htf_trend_map,
    )

    # Tag each signal with instrument info
    tagged = []
    for s in signals:
        tagged.append({
            "time": s.timestamp,
            "entry": s.entry_price,
            "sl": s.stop_loss,
            "tp": s.take_profit,
            "direction": s.direction.value,
            "sl_dist": s.sl_distance,
            "source": f"MS_{inst.symbol}",
            "inst": inst,
            "details": {**s.confluence_details, "instrument": inst.symbol},
        })

    print(f"[{inst.symbol}] {len(tagged)} MS signals")
    return tagged


def position_size_for_instrument(inst: InstrumentConfig, sl_distance: float,
                                 risk_dollars: float) -> int:
    """Calculate contracts for a given instrument."""
    if sl_distance <= 0:
        return 0

    sl_ticks = sl_distance / inst.tick_size
    risk_per_contract = sl_ticks * inst.tick_value

    if risk_per_contract <= 0:
        return 0

    contracts = int(risk_dollars / risk_per_contract)
    return max(1, min(contracts, 50))  # cap at 50 micros


def pnl_for_instrument(inst: InstrumentConfig, entry: float, exit_price: float,
                       contracts: int, is_long: bool, exit_is_sl: bool) -> float:
    """Calculate P&L for a specific instrument."""
    tick = inst.tick_size
    commission = 0.85 * 2 * contracts  # $0.85/side per micro

    # Entry: market order, 1 tick slippage
    if is_long:
        eff_entry = entry + tick
    else:
        eff_entry = entry - tick

    # Exit
    if exit_is_sl:
        sl_slip = 2 * tick  # 2 ticks SL slippage
        if is_long:
            eff_exit = exit_price - sl_slip
        else:
            eff_exit = exit_price + sl_slip
    else:
        eff_exit = exit_price  # limit TP, no slippage

    if is_long:
        raw = (eff_exit - eff_entry) * inst.oz_per_contract * contracts
    else:
        raw = (eff_entry - eff_exit) * inst.oz_per_contract * contracts

    return raw - commission


def run_multi_backtest(all_data: list[dict], max_days=22) -> BacktestResult:
    """Run all instruments on a shared prop firm tracker."""
    result = BacktestResult()
    tracker = result.tracker

    # Generate signals for all instruments
    all_signals = []
    for data in all_data:
        signals = generate_instrument_signals(data)
        all_signals.extend(signals)

    all_signals.sort(key=lambda x: x["time"])
    print(f"[MULTI] {len(all_signals)} total signals across {len(all_data)} instruments")

    # Build a unified 5-min timeline from the first instrument for date tracking
    ref = all_data[0]["5min"]
    if ref.index.tz is None:
        ct_ref = ref.index.tz_localize("UTC").tz_convert("US/Central")
    else:
        ct_ref = ref.index.tz_convert("US/Central")

    all_dates = sorted(set(ct_ref.date))

    current_date_idx = 0
    open_position = None
    signal_ptr = 0

    # Iterate through dates and signals
    for date in all_dates:
        tracker.start_new_day()
        if tracker.status != EvaluationStatus.ACTIVE:
            break
        if tracker.trading_days >= max_days:
            break

        # Process all signals for this date
        while signal_ptr < len(all_signals):
            sig = all_signals[signal_ptr]
            sig_date = sig["time"].date() if hasattr(sig["time"], 'date') else sig["time"]

            if hasattr(sig_date, '__call__'):
                sig_date = sig_date()

            # Convert to date for comparison
            if hasattr(sig["time"], 'tz_convert'):
                sig_ct = sig["time"].tz_convert("US/Central")
                sig_date = sig_ct.date()
            elif hasattr(sig["time"], 'date'):
                sig_date = sig["time"].date()

            if sig_date > date:
                break

            if sig_date < date:
                signal_ptr += 1
                continue

            # Skip if we have open position or can't trade
            if open_position is not None or not tracker.can_trade():
                signal_ptr += 1
                continue

            inst = sig["inst"]
            risk = PARAMS.risk_per_trade  # full risk per trade, not split
            contracts = position_size_for_instrument(inst, sig["sl_dist"], risk)

            if contracts > 0:
                is_long = sig["direction"] == "long"

                # Simulate: assume trade resolves same day
                # Use the R:R to determine outcome probabilistically based on
                # our known win rates. But for honest backtesting, we need candle data.
                # For now, use the 5-min candles to check SL/TP

                c5 = None
                for d in all_data:
                    if d["inst"].symbol == inst.symbol:
                        c5 = d["5min"]
                        break

                if c5 is None:
                    signal_ptr += 1
                    continue

                # Find candles after signal time
                future_mask = c5.index > sig["time"]
                future_candles = c5[future_mask].head(60)  # next 5 hours

                hit_sl = False
                hit_tp = False
                exit_price = sig["entry"]
                exit_reason = "timeout"

                for _, candle in future_candles.iterrows():
                    if is_long:
                        if candle["low"] <= sig["sl"]:
                            hit_sl = True
                            exit_price = sig["sl"]
                            break
                        if candle["high"] >= sig["tp"]:
                            hit_tp = True
                            exit_price = sig["tp"]
                            break
                    else:
                        if candle["high"] >= sig["sl"]:
                            hit_sl = True
                            exit_price = sig["sl"]
                            break
                        if candle["low"] <= sig["tp"]:
                            hit_tp = True
                            exit_price = sig["tp"]
                            break

                if hit_sl and hit_tp:
                    hit_tp = False

                if hit_sl or hit_tp:
                    pnl = pnl_for_instrument(
                        inst, sig["entry"], exit_price, contracts, is_long,
                        exit_is_sl=hit_sl)
                    exit_reason = "stop_loss" if hit_sl else "take_profit"
                else:
                    # Timeout - close at last candle close
                    if len(future_candles) > 0:
                        exit_price = future_candles["close"].iloc[-1]
                    pnl = pnl_for_instrument(
                        inst, sig["entry"], exit_price, contracts, is_long,
                        exit_is_sl=False)
                    exit_reason = "timeout"

                result.trades.append(TradeRecord(
                    entry_time=sig["time"], exit_time=sig["time"],
                    direction=sig["direction"], entry_price=sig["entry"],
                    exit_price=exit_price, stop_loss=sig["sl"], take_profit=sig["tp"],
                    contracts=contracts, pnl=pnl, exit_reason=exit_reason,
                    confluence_score=0, confluence_details=sig["details"],
                    trading_day=tracker.trading_days,
                ))
                tracker.record_trade(pnl)

                if tracker.status != EvaluationStatus.ACTIVE:
                    break

            signal_ptr += 1

        result.equity_curve.append(tracker.balance)

    tracker.end_evaluation()
    print(f"[MULTI] Complete: {len(result.trades)} trades, "
          f"Status: {tracker.status}, P&L: ${tracker.cumulative_pnl:.2f}")

    # Breakdown by instrument
    by_inst = {}
    for t in result.trades:
        inst_name = t.confluence_details.get("instrument", "?")
        by_inst.setdefault(inst_name, []).append(t)
    for inst_name, trades in sorted(by_inst.items()):
        wins = sum(1 for t in trades if t.pnl > 0)
        total_pnl = sum(t.pnl for t in trades)
        print(f"  {inst_name}: {len(trades)} trades, "
              f"WR {wins/len(trades)*100:.0f}%, P&L ${total_pnl:.0f}")

    return result


def run_multi_walkforward(all_data, window_days=22, step_days=22):
    """Walkforward across all instruments."""
    results = []
    ref = all_data[0]["5min"]

    if ref.index.tz is None:
        ct = ref.index.tz_localize("UTC").tz_convert("US/Central")
    else:
        ct = ref.index.tz_convert("US/Central")

    all_dates = sorted(set(ct.date))

    for start_idx in range(0, len(all_dates) - window_days, step_days):
        start_date = all_dates[start_idx]
        end_date = all_dates[min(start_idx + window_days, len(all_dates) - 1)]

        # Slice all instruments to this window
        window_data = []
        for data in all_data:
            c5 = data["5min"]
            c15 = data["15min"]
            inst = data["inst"]

            if c5.index.tz is None:
                c5_ct = c5.index.tz_localize("UTC").tz_convert("US/Central")
                c15_ct = c15.index.tz_localize("UTC").tz_convert("US/Central")
            else:
                c5_ct = c5.index.tz_convert("US/Central")
                c15_ct = c15.index.tz_convert("US/Central")

            w5 = c5[(c5_ct.date >= start_date) & (c5_ct.date <= end_date)]
            w15 = c15[(c15_ct.date >= start_date) & (c15_ct.date <= end_date)]

            if len(w5) < 50:
                continue

            window_data.append({"5min": w5, "15min": w15, "inst": inst})

        if len(window_data) < 2:
            continue

        result = run_multi_backtest(window_data, max_days=window_days)
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description="Multi-Instrument Strategy")
    parser.add_argument("--walkforward", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  MULTI-INSTRUMENT: Gold + NQ + ES + CL")
    print("=" * 60)

    # Load all instruments (trending period)
    all_data = []
    for inst in ALL_INSTRUMENTS:
        data = load_instrument(inst)
        if data is not None:
            all_data.append(data)

    print(f"\nLoaded {len(all_data)} instruments")

    if args.walkforward:
        results = run_multi_walkforward(all_data)
        passed = sum(1 for r in results if r.tracker.status == EvaluationStatus.PASSED)
        profitable = sum(1 for r in results if r.tracker.cumulative_pnl > 0)
        busted = sum(1 for r in results if r.tracker.status == "failed_drawdown")
        pnls = [r.tracker.cumulative_pnl for r in results]
        trades = [len(r.trades) for r in results]

        n = len(results)
        print(f"\n{'=' * 60}")
        print(f"  MULTI-INSTRUMENT WALKFORWARD")
        print(f"{'=' * 60}")
        print(f"  Windows:     {n} (non-overlapping)")
        print(f"  PASSED:      {passed}/{n} ({passed/n*100:.1f}%)")
        print(f"  Profitable:  {profitable}/{n} ({profitable/n*100:.1f}%)")
        print(f"  Busted:      {busted}/{n} ({busted/n*100:.1f}%)")
        if pnls:
            print(f"  Mean P&L:    ${np.mean(pnls):.0f}")
            print(f"  Median P&L:  ${np.median(pnls):.0f}")
            print(f"  Avg trades:  {np.mean(trades):.1f}")
        print(f"{'=' * 60}")
    else:
        result = run_multi_backtest(all_data)
        metrics = calculate_metrics(result)
        print_metrics(metrics)


if __name__ == "__main__":
    main()
