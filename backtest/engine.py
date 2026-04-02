"""Backtesting engine: event-driven simulation on candle data."""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import date

from strategy.market_structure import (
    detect_swing_points, detect_structure_breaks, SwingType, Trend,
    identify_trend, SwingPoint,
)
from strategy.order_blocks import detect_order_blocks, update_order_blocks
from strategy.fvg import detect_fvgs, update_fvgs
from strategy.liquidity import detect_liquidity_sweeps
from strategy.sessions import compute_asian_range
from strategy.signals import generate_signals, TradeSignal, SignalDirection
from risk.position_sizing import calculate_position_size, calculate_trade_pnl, calculate_adaptive_risk
from risk.prop_firm_rules import PropFirmTracker, EvaluationStatus
from config import PARAMS, CONTRACT


@dataclass
class TradeRecord:
    entry_time: object
    exit_time: object
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    contracts: int
    pnl: float
    exit_reason: str
    confluence_score: int
    confluence_details: dict
    trading_day: int


@dataclass
class BacktestResult:
    trades: list[TradeRecord] = field(default_factory=list)
    tracker: PropFirmTracker = field(default_factory=PropFirmTracker)
    equity_curve: list = field(default_factory=list)


def _build_htf_trend_map(
    candles_entry: pd.DataFrame,
    candles_htf: pd.DataFrame,
    swings_htf: list,
) -> dict[int, str]:
    """Build a map from entry candle index -> HTF trend ('bullish'/'bearish'/'neutral').

    For each entry candle, finds the corresponding HTF candle and determines
    the trend based on HTF swing points confirmed by that time.
    """
    trend_map = {}

    # Get HTF timestamps for mapping
    htf_times = candles_htf.index
    entry_times = candles_entry.index

    sh = [s for s in swings_htf if s.swing_type == SwingType.HIGH]
    sl = [s for s in swings_htf if s.swing_type == SwingType.LOW]

    # Pre-compute trend at each HTF candle index
    htf_trends = {}
    for htf_idx in range(len(candles_htf)):
        available_sh = [s for s in sh if s.index <= htf_idx]
        available_sl = [s for s in sl if s.index <= htf_idx]
        htf_trends[htf_idx] = identify_trend(available_sh, available_sl)

    # Map entry candles to HTF trend
    htf_ptr = 0
    for i in range(len(candles_entry)):
        # Advance HTF pointer to match entry time
        while htf_ptr < len(htf_times) - 1 and htf_times[htf_ptr + 1] <= entry_times[i]:
            htf_ptr += 1

        if htf_ptr in htf_trends:
            trend_map[i] = htf_trends[htf_ptr].value
        else:
            trend_map[i] = "neutral"

    return trend_map


def run_backtest(
    candles_5min: pd.DataFrame,
    candles_htf: pd.DataFrame,
    max_days: int = 22,
) -> BacktestResult:
    """Run a full backtest simulation.

    Args:
        candles_5min: 5-minute candles with session tags.
        candles_htf: Higher-timeframe candles (15min) for structure.
        max_days: Maximum trading days to simulate.

    Returns:
        BacktestResult with all trades and evaluation status.
    """
    result = BacktestResult()
    tracker = result.tracker

    # ── Pre-compute structure on entry timeframe ──
    print("[BACKTEST] Detecting market structure...")
    swings = detect_swing_points(candles_5min, lookback=PARAMS.swing_lookback)

    # HTF swings for trend confirmation
    swings_htf_raw = detect_swing_points(candles_htf, lookback=PARAMS.swing_lookback)

    # Build HTF trend map: for each entry candle, what's the HTF trend?
    # Map entry timestamps to nearest HTF candle index
    htf_trend_map = _build_htf_trend_map(candles_5min, candles_htf, swings_htf_raw)

    structure_events = detect_structure_breaks(candles_5min, swings)
    order_blocks = detect_order_blocks(candles_5min, structure_events, PARAMS.ob_max_age_candles)
    fvgs = detect_fvgs(candles_5min, PARAMS.fvg_min_size)
    sweeps = detect_liquidity_sweeps(
        candles_5min, swings,
        tolerance=PARAMS.equal_level_tolerance,
        lookback=PARAMS.sweep_lookback_candles,
    )

    print(f"[BACKTEST] Found {len(swings)} swings, {len(structure_events)} structure events, "
          f"{len(order_blocks)} OBs, {len(fvgs)} FVGs, {len(sweeps)} sweeps")

    # ── Compute Asian ranges ──
    asian_ranges = compute_asian_range(candles_5min)
    asian_dict = {}
    for _, row in asian_ranges.iterrows():
        asian_dict[row["date"]] = (row["asian_high"], row["asian_low"])

    # ── Generate signals ──
    print("[BACKTEST] Generating signals...")

    signals = generate_signals(
        candles_entry=candles_5min,
        swings_htf=swings,
        structure_events=structure_events,
        order_blocks=order_blocks,
        fvgs=fvgs,
        sweeps=sweeps,
        htf_trend_map=htf_trend_map,
    )
    print(f"[BACKTEST] Generated {len(signals)} raw signals")

    # ── Simulate trading ──
    print("[BACKTEST] Simulating trades...")

    closes = candles_5min["close"].values
    highs = candles_5min["high"].values
    lows = candles_5min["low"].values
    timestamps = candles_5min.index

    # Get trading dates
    if candles_5min.index.tz is None:
        ct_idx = candles_5min.index.tz_localize("UTC").tz_convert("US/Central")
    else:
        ct_idx = candles_5min.index.tz_convert("US/Central")
    trading_dates = ct_idx.date

    current_date = None
    open_position = None
    signal_queue = list(signals)
    signal_idx = 0

    for i in range(len(candles_5min)):
        candle_date = trading_dates[i]

        # New trading day
        if candle_date != current_date:
            current_date = candle_date
            tracker.start_new_day()

            if tracker.status != EvaluationStatus.ACTIVE:
                break
            if tracker.trading_days >= max_days:
                break

        # ── Manage open position ──
        if open_position is not None:
            signal, contracts, entry_day = open_position
            is_long = signal.direction == SignalDirection.LONG

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

            # When both SL and TP hit on same candle, use OHLC to determine order
            if hit_sl and hit_tp:
                o = candles_5min["open"].iloc[i]
                if is_long:
                    if o <= signal.entry_price:
                        hit_tp = False  # opened low -> went down first -> SL
                    else:
                        hit_sl = False  # opened high -> went up first -> TP
                else:
                    if o >= signal.entry_price:
                        hit_tp = False  # opened high -> went up first -> SL
                    else:
                        hit_sl = False  # opened low -> went down first -> TP

            if hit_sl:
                pnl = calculate_trade_pnl(
                    signal.entry_price, signal.stop_loss, contracts, is_long,
                )
                result.trades.append(TradeRecord(
                    entry_time=signal.timestamp, exit_time=timestamps[i],
                    direction=signal.direction.value,
                    entry_price=signal.entry_price, exit_price=signal.stop_loss,
                    stop_loss=signal.stop_loss, take_profit=signal.take_profit,
                    contracts=contracts, pnl=pnl, exit_reason="stop_loss",
                    confluence_score=signal.confluence_score,
                    confluence_details=signal.confluence_details,
                    trading_day=tracker.trading_days,
                ))
                can_continue = tracker.record_trade(pnl)
                open_position = None
                if not can_continue:
                    break

            elif hit_tp:
                pnl = calculate_trade_pnl(
                    signal.entry_price, signal.take_profit, contracts, is_long,
                )
                result.trades.append(TradeRecord(
                    entry_time=signal.timestamp, exit_time=timestamps[i],
                    direction=signal.direction.value,
                    entry_price=signal.entry_price, exit_price=signal.take_profit,
                    stop_loss=signal.stop_loss, take_profit=signal.take_profit,
                    contracts=contracts, pnl=pnl, exit_reason="take_profit",
                    confluence_score=signal.confluence_score,
                    confluence_details=signal.confluence_details,
                    trading_day=tracker.trading_days,
                ))
                can_continue = tracker.record_trade(pnl)
                open_position = None
                if not can_continue:
                    break

            # Skip signal processing if position is open
            if open_position is not None:
                result.equity_curve.append(tracker.balance)
                continue

        # ── Check for new signals ──
        if open_position is None and tracker.can_trade():
            # Find next signal at this candle
            while signal_idx < len(signal_queue) and signal_queue[signal_idx].index <= i:
                if signal_queue[signal_idx].index == i:
                    signal = signal_queue[signal_idx]
                    # Adaptive risk: scale risk based on progress
                    adaptive_risk = calculate_adaptive_risk(
                        tracker.cumulative_pnl, tracker.trading_days,
                    )
                    contracts = calculate_position_size(signal.sl_distance, adaptive_risk)

                    if contracts > 0:
                        # Get Asian range for today
                        ah, al = asian_dict.get(candle_date, (None, None))
                        open_position = (signal, contracts, tracker.trading_days)
                        signal_idx += 1
                        break
                signal_idx += 1

        result.equity_curve.append(tracker.balance)

    tracker.end_evaluation()
    result.tracker = tracker

    print(f"[BACKTEST] Complete: {len(result.trades)} trades, "
          f"Status: {tracker.status}, P&L: ${tracker.cumulative_pnl:.2f}")

    return result


def run_walkforward_backtest(
    candles_5min: pd.DataFrame,
    candles_htf: pd.DataFrame,
    window_days: int = 22,
    step_days: int = 5,
) -> list[BacktestResult]:
    """Run multiple backtests over rolling windows to simulate multiple evaluation attempts."""
    results = []

    if candles_5min.index.tz is None:
        ct_idx = candles_5min.index.tz_localize("UTC").tz_convert("US/Central")
    else:
        ct_idx = candles_5min.index.tz_convert("US/Central")

    all_dates = sorted(set(ct_idx.date))

    for start_idx in range(0, len(all_dates) - window_days, step_days):
        start_date = all_dates[start_idx]
        end_date = all_dates[min(start_idx + window_days, len(all_dates) - 1)]

        # Filter candles to this window
        mask_5 = (ct_idx.date >= start_date) & (ct_idx.date <= end_date)
        window_5min = candles_5min[mask_5]

        if candles_htf.index.tz is None:
            htf_ct = candles_htf.index.tz_localize("UTC").tz_convert("US/Central")
        else:
            htf_ct = candles_htf.index.tz_convert("US/Central")
        mask_htf = (htf_ct.date >= start_date) & (htf_ct.date <= end_date)
        window_htf = candles_htf[mask_htf]

        if len(window_5min) < 100 or len(window_htf) < 20:
            continue

        print(f"\n[WALKFORWARD] Window: {start_date} to {end_date}")
        result = run_backtest(window_5min, window_htf, max_days=window_days)
        results.append(result)

    return results
