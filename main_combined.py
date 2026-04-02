#!/usr/bin/env python3
"""Combined Strategy: Market Structure + CBR running in parallel.

MS trades during NY session (trend continuation pullbacks).
CBR trades during Asia/London (mean-reversion after sweeps).
Shared prop firm tracker and risk management.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from config import DATA_DIR, RESULTS_DIR, RULES, CONTRACT, PARAMS
from data.download_data import load_data
from data.resample import build_multi_timeframe, tag_sessions
from strategy.cbr import generate_cbr_signals, CBRSignal, CBRDirection
from strategy.signals import generate_signals, TradeSignal, SignalDirection
from strategy.mean_reversion import generate_mr_signals, MRSignal, MRDirection
from strategy.market_structure import detect_swing_points, detect_structure_breaks
from strategy.order_blocks import detect_order_blocks
from strategy.fvg import detect_fvgs
from strategy.liquidity import detect_liquidity_sweeps
from strategy.sessions import compute_asian_range
from risk.position_sizing import calculate_position_size, calculate_trade_pnl, calculate_adaptive_risk
from risk.prop_firm_rules import PropFirmTracker, EvaluationStatus
from backtest.engine import BacktestResult, TradeRecord, _build_htf_trend_map
from backtest.metrics import calculate_metrics, print_metrics
from backtest.monte_carlo import run_monte_carlo, print_monte_carlo


def run_combined_backtest(
    candles_1min: pd.DataFrame,
    candles_5min: pd.DataFrame,
    candles_15min: pd.DataFrame,
    candles_1h: pd.DataFrame,
    max_days: int = 22,
) -> BacktestResult:
    """Run both strategies on shared prop firm tracker."""
    result = BacktestResult()
    tracker = result.tracker

    # ── Generate MS signals on 5-min ──
    print("[COMBINED] Generating Market Structure signals (5-min)...")
    swings = detect_swing_points(candles_5min, lookback=PARAMS.swing_lookback)
    htf_trend_map = _build_htf_trend_map(candles_5min, candles_15min,
                                          detect_swing_points(candles_15min, lookback=PARAMS.swing_lookback))
    structure_events = detect_structure_breaks(candles_5min, swings)
    order_blocks = detect_order_blocks(candles_5min, structure_events, PARAMS.ob_max_age_candles)
    fvgs = detect_fvgs(candles_5min, PARAMS.fvg_min_size)
    sweeps_ms = detect_liquidity_sweeps(candles_5min, swings,
                                        tolerance=PARAMS.equal_level_tolerance,
                                        lookback=PARAMS.sweep_lookback_candles)

    ms_signals = generate_signals(
        candles_entry=candles_5min, swings_htf=swings,
        structure_events=structure_events, order_blocks=order_blocks,
        fvgs=fvgs, sweeps=sweeps_ms, htf_trend_map=htf_trend_map,
    )
    print(f"[COMBINED] MS: {len(ms_signals)} signals")

    # ── Generate CBR signals on 1-min ──
    print("[COMBINED] Generating CBR signals (1-min)...")
    cbr_signals = generate_cbr_signals(candles_1min, candles_1h)
    print(f"[COMBINED] CBR: {len(cbr_signals)} signals")

    # ── Merge all signals into unified timeline ──
    all_signals = []

    for s in ms_signals:
        all_signals.append({
            "time": s.timestamp,
            "entry": s.entry_price,
            "sl": s.stop_loss,
            "tp": s.take_profit,
            "direction": "long" if s.direction == SignalDirection.LONG else "short",
            "sl_dist": s.sl_distance,
            "source": "MS",
            "entry_is_limit": False,  # MS uses market order
            "details": s.confluence_details,
        })

    for s in cbr_signals:
        all_signals.append({
            "time": s.timestamp,
            "entry": s.entry_price,
            "sl": s.stop_loss,
            "tp": s.take_profit,
            "direction": s.direction.value,
            "sl_dist": abs(s.entry_price - s.stop_loss),
            "source": "CBR",
            "entry_is_limit": True,  # CBR uses limit order at 50% retracement
            "details": s.details,
        })

    all_signals.sort(key=lambda x: x["time"])
    print(f"[COMBINED] Total: {len(all_signals)} merged signals")

    # ── Simulate on 1-min candles ──
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
    pending_order = None
    signal_ptr = 0

    for i in range(len(candles_1min)):
        candle_date = trading_dates[i]
        candle_time = timestamps[i]

        if candle_date != current_date:
            current_date = candle_date
            tracker.start_new_day()
            pending_order = None
            if tracker.status != EvaluationStatus.ACTIVE:
                break
            if tracker.trading_days >= max_days:
                break

        # ── Check pending limit order fill (CBR entries) ──
        if pending_order is not None and open_position is None:
            sig = pending_order
            is_long = sig["direction"] == "long"
            if is_long:
                if lows[i] <= sig["entry"]:
                    open_position = pending_order
                    pending_order = None
                elif lows[i] <= sig["sl"]:
                    pending_order = None
            else:
                if highs[i] >= sig["entry"]:
                    open_position = pending_order
                    pending_order = None
                elif highs[i] >= sig["sl"]:
                    pending_order = None
            # Expire after 120 bars (2 hours)
            if pending_order is not None:
                time_diff = (candle_time - sig["time"]).total_seconds() / 60
                if time_diff > 120:
                    pending_order = None

        # ── Manage open position ──
        if open_position is not None:
            sig = open_position
            is_long = sig["direction"] == "long"
            contracts = sig["contracts"]

            hit_sl = (lows[i] <= sig["sl"]) if is_long else (highs[i] >= sig["sl"])
            hit_tp = (highs[i] >= sig["tp"]) if is_long else (lows[i] <= sig["tp"])

            if hit_sl and hit_tp:
                hit_tp = False  # conservative

            if hit_sl:
                pnl = calculate_trade_pnl(
                    sig["entry"], sig["sl"], contracts, is_long,
                    exit_is_sl=True, entry_is_limit=sig["entry_is_limit"])
                result.trades.append(TradeRecord(
                    entry_time=sig["time"], exit_time=candle_time,
                    direction=sig["direction"], entry_price=sig["entry"],
                    exit_price=sig["sl"], stop_loss=sig["sl"], take_profit=sig["tp"],
                    contracts=contracts, pnl=pnl, exit_reason="stop_loss",
                    confluence_score=0, confluence_details=sig["details"],
                    trading_day=tracker.trading_days,
                ))
                tracker.record_trade(pnl)
                open_position = None
                if tracker.status != EvaluationStatus.ACTIVE:
                    break

            elif hit_tp:
                pnl = calculate_trade_pnl(
                    sig["entry"], sig["tp"], contracts, is_long,
                    exit_is_sl=False, entry_is_limit=sig["entry_is_limit"])
                result.trades.append(TradeRecord(
                    entry_time=sig["time"], exit_time=candle_time,
                    direction=sig["direction"], entry_price=sig["entry"],
                    exit_price=sig["tp"], stop_loss=sig["sl"], take_profit=sig["tp"],
                    contracts=contracts, pnl=pnl, exit_reason="take_profit",
                    confluence_score=0, confluence_details=sig["details"],
                    trading_day=tracker.trading_days,
                ))
                tracker.record_trade(pnl)
                open_position = None
                if tracker.status != EvaluationStatus.ACTIVE:
                    break

            if open_position is not None:
                result.equity_curve.append(tracker.balance)
                continue

        # ── Check for new signals ──
        if open_position is None and pending_order is None and tracker.can_trade():
            while signal_ptr < len(all_signals):
                sig = all_signals[signal_ptr]
                if sig["time"] > candle_time:
                    break
                if sig["time"] == candle_time or (
                    sig["time"] < candle_time and
                    (candle_time - sig["time"]).total_seconds() < 300
                ):
                    risk = PARAMS.risk_per_trade
                    contracts = calculate_position_size(sig["sl_dist"], risk)
                    if contracts > 0:
                        sig["contracts"] = contracts
                        if sig["entry_is_limit"]:
                            pending_order = sig
                        else:
                            open_position = sig
                        signal_ptr += 1
                        break
                signal_ptr += 1

        result.equity_curve.append(tracker.balance)

    tracker.end_evaluation()
    result.tracker = tracker
    print(f"[COMBINED] Complete: {len(result.trades)} trades, "
          f"Status: {tracker.status}, P&L: ${tracker.cumulative_pnl:.2f}")
    return result


def run_combined_walkforward(
    candles_1min, candles_5min, candles_15min, candles_1h,
    window_days=22, step_days=22,
):
    """Non-overlapping walkforward for combined strategy."""
    results = []

    if candles_1min.index.tz is None:
        ct_1m = candles_1min.index.tz_localize("UTC").tz_convert("US/Central")
    else:
        ct_1m = candles_1min.index.tz_convert("US/Central")

    def to_ct(idx):
        if idx.tz is None:
            return idx.tz_localize("UTC").tz_convert("US/Central")
        return idx.tz_convert("US/Central")

    ct_5 = to_ct(candles_5min.index)
    ct_15 = to_ct(candles_15min.index)
    ct_1h = to_ct(candles_1h.index)

    all_dates = sorted(set(ct_1m.date))

    for start_idx in range(0, len(all_dates) - window_days, step_days):
        start_date = all_dates[start_idx]
        end_date = all_dates[min(start_idx + window_days, len(all_dates) - 1)]

        w1m = candles_1min[(ct_1m.date >= start_date) & (ct_1m.date <= end_date)]
        w5 = candles_5min[(ct_5.date >= start_date) & (ct_5.date <= end_date)]
        w15 = candles_15min[(ct_15.date >= start_date) & (ct_15.date <= end_date)]
        w1h = candles_1h[(ct_1h.date >= start_date) & (ct_1h.date <= end_date)]

        if len(w1m) < 1000 or len(w5) < 100 or len(w15) < 20:
            continue

        result = run_combined_backtest(w1m, w5, w15, w1h, max_days=window_days)
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description="Combined MS + CBR Strategy")
    parser.add_argument("--walkforward", action="store_true")
    parser.add_argument("--mc-sims", type=int, default=10000)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  COMBINED STRATEGY: Market Structure + CBR")
    print("=" * 60)

    raw = load_data()
    raw.attrs["_is_ohlcv"] = True
    cd = build_multi_timeframe(raw)
    c1m = tag_sessions(raw.copy())
    c5 = cd["5min"]
    c15 = cd["15min"]
    c1h = cd["1H"]

    print(f"[DATA] 1-min: {len(c1m):,}, 5-min: {len(c5):,}, 15-min: {len(c15):,}, 1H: {len(c1h):,}")

    if args.walkforward:
        results = run_combined_walkforward(c1m, c5, c15, c1h)
        passed = sum(1 for r in results if r.tracker.status == EvaluationStatus.PASSED)
        profitable = sum(1 for r in results if r.tracker.cumulative_pnl > 0)
        busted = sum(1 for r in results if r.tracker.status == "failed_drawdown")
        pnls = [r.tracker.cumulative_pnl for r in results]
        trades = [len(r.trades) for r in results]

        print(f"\n{'=' * 60}")
        print(f"  COMBINED WALKFORWARD RESULTS")
        print(f"{'=' * 60}")
        print(f"  Windows:     {len(results)} (non-overlapping)")
        print(f"  PASSED:      {passed}/{len(results)} ({passed/len(results)*100:.1f}%)")
        print(f"  Profitable:  {profitable}/{len(results)} ({profitable/len(results)*100:.1f}%)")
        print(f"  Busted:      {busted}/{len(results)} ({busted/len(results)*100:.1f}%)")
        if pnls:
            print(f"  Mean P&L:    ${np.mean(pnls):.0f}")
            print(f"  Median P&L:  ${np.median(pnls):.0f}")
            print(f"  Avg trades:  {np.mean(trades):.1f}")
        print(f"{'=' * 60}")
    else:
        result = run_combined_backtest(c1m, c5, c15, c1h)
        metrics = calculate_metrics(result)
        print_metrics(metrics)

        if result.trades:
            ms_trades = [t for t in result.trades if t.confluence_details.get("structure")]
            cbr_trades = [t for t in result.trades if t.confluence_details.get("msb")]
            other = len(result.trades) - len(ms_trades) - len(cbr_trades)
            print(f"\n  Trade sources: MS={len(ms_trades)}, CBR={len(cbr_trades)}, Other={other}")

            pnls = [t.pnl for t in result.trades]
            mc = run_monte_carlo(pnls, n_simulations=args.mc_sims,
                                 trades_per_day=metrics.get("trades_per_day", 1.5))
            print_monte_carlo(mc, "Combined Bootstrap")


if __name__ == "__main__":
    main()
