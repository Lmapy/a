"""Signal generation: Structure-based entries with session and sweep confluence."""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
from bisect import bisect_right

from strategy.market_structure import (
    SwingPoint, SwingType, StructureEvent, StructureBreak,
    Trend, get_current_trend_at, get_last_swing,
)
from strategy.order_blocks import OrderBlock, OBType
from strategy.fvg import FairValueGap, FVGType
from strategy.liquidity import LiquiditySweep, SweepType
from strategy.sessions import is_tradeable_session
from config import PARAMS


class SignalDirection(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class TradeSignal:
    direction: SignalDirection
    index: int
    timestamp: object
    entry_price: float
    stop_loss: float
    take_profit: float
    confluence_score: int
    confluence_details: dict
    sl_distance: float
    tp_distance: float


def generate_signals(
    candles_entry: pd.DataFrame,
    swings_htf: list[SwingPoint],
    structure_events: list[StructureEvent],
    order_blocks: list[OrderBlock],
    fvgs: list[FairValueGap],
    sweeps: list[LiquiditySweep],
    asian_high: float | None = None,
    asian_low: float | None = None,
    htf_candle_map: dict | None = None,
) -> list[TradeSignal]:
    """Generate trade signals based on structure + confluence.

    Core logic:
    1. Wait for a structure break (BOS/CHoCH) to establish direction
    2. Wait for price to pull back (retrace) after the break
    3. Enter on a momentum candle in the direction of the break
    4. SL below/above the swing that was just swept or the pullback low/high
    5. TP at 2:1 R:R targeting the next swing level
    """
    signals = []
    closes = candles_entry["close"].values
    opens = candles_entry["open"].values
    highs = candles_entry["high"].values
    lows = candles_entry["low"].values
    sessions = candles_entry["session"].values if "session" in candles_entry.columns else None
    timestamps = candles_entry.index
    n = len(candles_entry)

    # Pre-index structure events
    event_map = {}  # index -> event
    for e in structure_events:
        event_map[e.index] = e

    # Pre-index sweeps
    sweep_map = {}
    for s in sweeps:
        sweep_map.setdefault(s.index, []).append(s)

    # Pre-index OBs and FVGs by index for proximity checks
    ob_by_idx = {}
    for ob in order_blocks:
        ob_by_idx.setdefault(ob.index, []).append(ob)

    fvg_by_idx = {}
    for f in fvgs:
        fvg_by_idx.setdefault(f.index, []).append(f)

    # Swing highs and lows for SL/TP references
    swing_highs = [s for s in swings_htf if s.swing_type == SwingType.HIGH]
    swing_lows = [s for s in swings_htf if s.swing_type == SwingType.LOW]

    print(f"[SIGNALS] Processing {n:,} candles...")

    last_signal_index = -20
    last_structure_event = None
    last_structure_index = -1000
    pullback_started = False
    pullback_extreme = 0.0

    for i in range(10, n):
        # Track the most recent structure event
        if i in event_map:
            last_structure_event = event_map[i]
            last_structure_index = i
            pullback_started = False

        # Skip if no recent structure event
        if last_structure_event is None or (i - last_structure_index) > 60:
            continue

        # Skip if too close to last signal
        if i - last_signal_index < 8:
            continue

        # Session filter
        if sessions is not None and not is_tradeable_session(sessions[i]):
            continue

        price = closes[i]

        # ── BULLISH SETUP: After bullish BOS/CHoCH ──
        if last_structure_event.break_type in (
            StructureBreak.BOS_BULLISH, StructureBreak.CHOCH_BULLISH
        ):
            # Wait for pullback: price retraces below the break level
            if not pullback_started:
                if price < last_structure_event.level_broken:
                    pullback_started = True
                    pullback_extreme = lows[i]
                continue

            # Track pullback low
            if lows[i] < pullback_extreme:
                pullback_extreme = lows[i]

            # Entry: momentum candle after pullback
            direction = _check_momentum(opens, closes, highs, lows, i)
            if direction != "bullish":
                continue

            # Build confluence score
            score = 1  # structure break is already 1 point
            details = {"structure": last_structure_event.break_type.value}

            # Check if pullback went into an OB zone
            ob_hit = _check_ob_proximity(ob_by_idx, pullback_extreme, i,
                                         OBType.BULLISH, max_age=40)
            if ob_hit:
                score += 1
                details["ob"] = f"{ob_hit.low:.1f}-{ob_hit.high:.1f}"

            # Check if pullback filled into an FVG
            fvg_hit = _check_fvg_proximity(fvg_by_idx, pullback_extreme, i,
                                           FVGType.BULLISH, max_age=30)
            if fvg_hit:
                score += 1
                details["fvg"] = f"{fvg_hit.bottom:.1f}-{fvg_hit.top:.1f}"

            # Check for sweep during pullback
            has_sweep = _check_sweep_in_range(sweep_map, SweepType.BULLISH,
                                              last_structure_index, i)
            if has_sweep:
                score += 1
                details["sweep"] = "bullish"

            # Session bonus
            if sessions is not None and sessions[i] == "ny":
                score += 1
                details["session"] = "ny"
            elif sessions is not None and sessions[i] == "london":
                score += 1
                details["session"] = "london"

            if score >= PARAMS.min_confluence_score:
                # SL: below pullback low (or OB/FVG bottom)
                sl_ref = pullback_extreme
                if ob_hit:
                    sl_ref = min(sl_ref, ob_hit.low)
                if fvg_hit:
                    sl_ref = min(sl_ref, fvg_hit.bottom)

                sl_price = sl_ref - PARAMS.sl_buffer
                sl_distance = price - sl_price

                if 1.0 < sl_distance <= 20.0:
                    tp_distance = sl_distance * PARAMS.reward_risk_ratio

                    # TP target: look for next swing high above entry
                    tp_price = price + tp_distance
                    for sh in swing_highs:
                        if sh.index > last_structure_index and sh.price > price + sl_distance:
                            tp_price = min(tp_price, sh.price)
                            tp_distance = tp_price - price
                            break

                    if tp_distance >= sl_distance * 1.5:  # at least 1.5:1 R:R
                        signals.append(TradeSignal(
                            direction=SignalDirection.LONG, index=i,
                            timestamp=timestamps[i], entry_price=price,
                            stop_loss=sl_price, take_profit=tp_price,
                            confluence_score=score, confluence_details=details,
                            sl_distance=sl_distance, tp_distance=tp_distance,
                        ))
                        last_signal_index = i
                        last_structure_event = None  # consumed

        # ── BEARISH SETUP: After bearish BOS/CHoCH ──
        elif last_structure_event.break_type in (
            StructureBreak.BOS_BEARISH, StructureBreak.CHOCH_BEARISH
        ):
            # Wait for pullback: price retraces above the break level
            if not pullback_started:
                if price > last_structure_event.level_broken:
                    pullback_started = True
                    pullback_extreme = highs[i]
                continue

            # Track pullback high
            if highs[i] > pullback_extreme:
                pullback_extreme = highs[i]

            # Entry: momentum candle after pullback
            direction = _check_momentum(opens, closes, highs, lows, i)
            if direction != "bearish":
                continue

            score = 1
            details = {"structure": last_structure_event.break_type.value}

            ob_hit = _check_ob_proximity(ob_by_idx, pullback_extreme, i,
                                         OBType.BEARISH, max_age=40)
            if ob_hit:
                score += 1
                details["ob"] = f"{ob_hit.low:.1f}-{ob_hit.high:.1f}"

            fvg_hit = _check_fvg_proximity(fvg_by_idx, pullback_extreme, i,
                                           FVGType.BEARISH, max_age=30)
            if fvg_hit:
                score += 1
                details["fvg"] = f"{fvg_hit.bottom:.1f}-{fvg_hit.top:.1f}"

            has_sweep = _check_sweep_in_range(sweep_map, SweepType.BEARISH,
                                              last_structure_index, i)
            if has_sweep:
                score += 1
                details["sweep"] = "bearish"

            if sessions is not None and sessions[i] in ("ny", "london"):
                score += 1
                details["session"] = sessions[i]

            if score >= PARAMS.min_confluence_score:
                sl_ref = pullback_extreme
                if ob_hit:
                    sl_ref = max(sl_ref, ob_hit.high)
                if fvg_hit:
                    sl_ref = max(sl_ref, fvg_hit.top)

                sl_price = sl_ref + PARAMS.sl_buffer
                sl_distance = sl_price - price

                if 1.0 < sl_distance <= 20.0:
                    tp_distance = sl_distance * PARAMS.reward_risk_ratio

                    tp_price = price - tp_distance
                    for sl in swing_lows:
                        if sl.index > last_structure_index and sl.price < price - sl_distance:
                            tp_price = max(tp_price, sl.price)
                            tp_distance = price - tp_price
                            break

                    if tp_distance >= sl_distance * 1.5:
                        signals.append(TradeSignal(
                            direction=SignalDirection.SHORT, index=i,
                            timestamp=timestamps[i], entry_price=price,
                            stop_loss=sl_price, take_profit=tp_price,
                            confluence_score=score, confluence_details=details,
                            sl_distance=sl_distance, tp_distance=tp_distance,
                        ))
                        last_signal_index = i
                        last_structure_event = None

    return signals


def _check_momentum(opens, closes, highs, lows, i: int) -> str | None:
    """Fast check for momentum/engulfing candle."""
    if i < 1:
        return None

    o, c, h, l = opens[i], closes[i], highs[i], lows[i]
    rng = h - l
    if rng < 0.30:
        return None

    body = abs(c - o)
    body_ratio = body / rng

    o_prev, c_prev = opens[i - 1], closes[i - 1]

    # Engulfing
    if c_prev < o_prev and c > o:
        if c > o_prev and o <= c_prev and body > abs(c_prev - o_prev):
            return "bullish"
    if c_prev > o_prev and c < o:
        if c < o_prev and o >= c_prev and body > abs(c_prev - o_prev):
            return "bearish"

    # Strong momentum
    if body_ratio >= 0.60:
        if c > o and (h - c) / rng < 0.25:
            return "bullish"
        if c < o and (c - l) / rng < 0.25:
            return "bearish"

    return None


def _check_ob_proximity(ob_by_idx: dict, extreme_price: float,
                        current_index: int, ob_type: OBType,
                        max_age: int = 40) -> OrderBlock | None:
    """Check if the pullback extreme touched an order block zone."""
    for idx in range(current_index - 1, max(current_index - max_age, 0), -1):
        if idx not in ob_by_idx:
            continue
        for ob in ob_by_idx[idx]:
            if ob.ob_type != ob_type or not ob.is_valid:
                continue
            if ob_type == OBType.BULLISH:
                # Bullish OB: pullback low should touch the OB zone
                if extreme_price <= ob.high and extreme_price >= ob.low * 0.998:
                    return ob
            else:
                # Bearish OB: pullback high should touch the OB zone
                if extreme_price >= ob.low and extreme_price <= ob.high * 1.002:
                    return ob
    return None


def _check_fvg_proximity(fvg_by_idx: dict, extreme_price: float,
                         current_index: int, fvg_type: FVGType,
                         max_age: int = 30) -> FairValueGap | None:
    """Check if the pullback filled into an FVG zone."""
    for idx in range(current_index - 1, max(current_index - max_age, 0), -1):
        if idx not in fvg_by_idx:
            continue
        for fvg in fvg_by_idx[idx]:
            if fvg.fvg_type != fvg_type or fvg.is_filled:
                continue
            if fvg_type == FVGType.BULLISH:
                if extreme_price <= fvg.top and extreme_price >= fvg.bottom * 0.998:
                    return fvg
            else:
                if extreme_price >= fvg.bottom and extreme_price <= fvg.top * 1.002:
                    return fvg
    return None


def _check_sweep_in_range(sweep_map: dict, sweep_type: SweepType,
                          start_idx: int, end_idx: int) -> bool:
    """Check if there was a sweep of the given type in the index range."""
    for idx in range(start_idx, end_idx):
        if idx in sweep_map:
            for s in sweep_map[idx]:
                if s.sweep_type == sweep_type:
                    return True
    return False
