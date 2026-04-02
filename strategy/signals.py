"""Signal generation: Structure-based entries with ATR, session, and sweep confluence.

Optimized for prop firm pass rate with:
- ATR-based dynamic SL sizing
- Session/time-of-day filtering (NY open + London are best)
- HTF trend alignment
- Pullback entries after structure breaks
"""

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
from strategy.regime import classify_regime
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


def compute_atr(highs, lows, closes, period=14):
    """Compute ATR array aligned with input arrays."""
    n = len(highs)
    atr = np.full(n, np.nan)
    if n < period + 1:
        return atr

    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))

    # EMA-style ATR
    atr[period] = np.mean(tr[1:period + 1])
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


def generate_signals(
    candles_entry: pd.DataFrame,
    swings_htf: list[SwingPoint],
    structure_events: list[StructureEvent],
    order_blocks: list[OrderBlock],
    fvgs: list[FairValueGap],
    sweeps: list[LiquiditySweep],
    asian_high: float | None = None,
    asian_low: float | None = None,
    htf_trend_map: dict | None = None,
) -> list[TradeSignal]:
    """Generate trade signals with ATR-based risk management."""
    signals = []
    closes = candles_entry["close"].values
    opens = candles_entry["open"].values
    highs = candles_entry["high"].values
    lows = candles_entry["low"].values
    sessions = candles_entry["session"].values if "session" in candles_entry.columns else None
    timestamps = candles_entry.index
    n = len(candles_entry)

    # Compute ATR for dynamic SL sizing
    atr = compute_atr(highs, lows, closes, period=14)

    # Compute market regime
    regimes, adx_arr, ema_arr = classify_regime(highs, lows, closes)

    # Get CT hours for time-of-day filter
    if candles_entry.index.tz is None:
        ct_idx = candles_entry.index.tz_localize("UTC").tz_convert("US/Central")
    else:
        ct_idx = candles_entry.index.tz_convert("US/Central")
    ct_hours = ct_idx.hour

    # Pre-index structure events and sweeps
    event_map = {}
    for e in structure_events:
        event_map[e.index] = e

    sweep_map = {}
    for s in sweeps:
        sweep_map.setdefault(s.index, []).append(s)

    ob_by_idx = {}
    for ob in order_blocks:
        ob_by_idx.setdefault(ob.index, []).append(ob)

    fvg_by_idx = {}
    for f in fvgs:
        fvg_by_idx.setdefault(f.index, []).append(f)

    swing_highs = [s for s in swings_htf if s.swing_type == SwingType.HIGH]
    swing_lows = [s for s in swings_htf if s.swing_type == SwingType.LOW]

    print(f"[SIGNALS] Processing {n:,} candles with ATR-based sizing...")

    last_signal_index = -20
    last_structure_event = None
    last_structure_index = -1000
    pullback_started = False
    pullback_extreme = 0.0

    for i in range(20, n):
        # Track structure events
        if i in event_map:
            last_structure_event = event_map[i]
            last_structure_index = i
            pullback_started = False

        # Skip if no recent structure event
        if last_structure_event is None or (i - last_structure_index) > 120:
            continue

        # Skip if too close to last signal
        if i - last_signal_index < 4:
            continue

        # ── TIME-OF-DAY FILTER ──
        # Trade during London + NY session hours (highest probability)
        # London: 2-8 AM CT, NY: 7-14 PM CT
        hour = ct_hours[i]
        if not (2 <= hour <= 14):
            continue

        # ── SESSION FILTER ──
        if sessions is not None and sessions[i] == "asian":
            continue

        # ── ATR FILTER ──
        current_atr = atr[i]
        if np.isnan(current_atr) or current_atr < 0.30 or current_atr > 4.0:
            continue

        # ── REGIME FILTER (optional) ──
        # Available but currently disabled for maximum trade frequency
        # regime = regimes[i]
        # if regime == 'ranging':
        #     continue

        price = closes[i]

        # ── BULLISH SETUP ──
        if last_structure_event.break_type in (
            StructureBreak.BOS_BULLISH, StructureBreak.CHOCH_BULLISH
        ):
            if not pullback_started:
                if price < last_structure_event.level_broken:
                    pullback_started = True
                    pullback_extreme = lows[i]
                continue

            if lows[i] < pullback_extreme:
                pullback_extreme = lows[i]

            direction = _check_momentum(opens, closes, highs, lows, i)
            if direction != "bullish":
                continue

            score, details, sl_ref = _evaluate_long(
                i, price, pullback_extreme, last_structure_index,
                swings_htf, ob_by_idx, fvg_by_idx, sweep_map,
                sessions, hour,
            )

            if score >= PARAMS.min_confluence_score:
                # ATR-based SL: use max(zone-based, 2x ATR)
                zone_sl = sl_ref - PARAMS.sl_buffer
                atr_sl = price - (current_atr * 2.5)
                sl_price = min(zone_sl, atr_sl)  # wider of the two
                sl_distance = price - sl_price

                if 0.8 < sl_distance <= 15.0:
                    tp_distance = sl_distance * PARAMS.reward_risk_ratio
                    tp_price = price + tp_distance

                    # Look for structural TP target
                    for sh in swing_highs:
                        if sh.index > last_structure_index and sh.price > price + sl_distance:
                            if sh.price < tp_price:
                                tp_price = sh.price
                                tp_distance = tp_price - price
                            break

                    if tp_distance >= sl_distance * 1.5:
                        signals.append(TradeSignal(
                            direction=SignalDirection.LONG, index=i,
                            timestamp=timestamps[i], entry_price=price,
                            stop_loss=sl_price, take_profit=tp_price,
                            confluence_score=score, confluence_details=details,
                            sl_distance=sl_distance, tp_distance=tp_distance,
                        ))
                        last_signal_index = i
                        last_structure_event = None

        # ── BEARISH SETUP ──
        elif last_structure_event.break_type in (
            StructureBreak.BOS_BEARISH, StructureBreak.CHOCH_BEARISH
        ):
            if not pullback_started:
                if price > last_structure_event.level_broken:
                    pullback_started = True
                    pullback_extreme = highs[i]
                continue

            if highs[i] > pullback_extreme:
                pullback_extreme = highs[i]

            direction = _check_momentum(opens, closes, highs, lows, i)
            if direction != "bearish":
                continue

            score, details, sl_ref = _evaluate_short(
                i, price, pullback_extreme, last_structure_index,
                swings_htf, ob_by_idx, fvg_by_idx, sweep_map,
                sessions, hour,
            )

            if score >= PARAMS.min_confluence_score:
                zone_sl = sl_ref + PARAMS.sl_buffer
                atr_sl = price + (current_atr * 2.5)
                sl_price = max(zone_sl, atr_sl)
                sl_distance = sl_price - price

                if 0.8 < sl_distance <= 15.0:
                    tp_distance = sl_distance * PARAMS.reward_risk_ratio
                    tp_price = price - tp_distance

                    for sl in swing_lows:
                        if sl.index > last_structure_index and sl.price < price - sl_distance:
                            if sl.price > tp_price:
                                tp_price = sl.price
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
    """Check for momentum/engulfing candle."""
    if i < 1:
        return None

    o, c, h, l = opens[i], closes[i], highs[i], lows[i]
    rng = h - l
    if rng < 0.20:
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

    # Strong momentum (body > 55% of range, small opposing wick)
    if body_ratio >= 0.55:
        if c > o and (h - c) / rng < 0.25:
            return "bullish"
        if c < o and (c - l) / rng < 0.25:
            return "bearish"

    return None


def _evaluate_long(i, price, pullback_low, struct_idx,
                   swings, ob_by_idx, fvg_by_idx, sweep_map,
                   sessions, hour):
    """Evaluate confluence for a long entry."""
    score = 1  # structure break already counts
    details = {"structure": "bullish_break"}
    sl_ref = pullback_low

    # Trend alignment from swings
    trend = get_current_trend_at(i, swings)
    if trend == Trend.BULLISH:
        score += 1
        details["trend"] = "bullish"

    # OB proximity
    ob = _check_ob_proximity(ob_by_idx, pullback_low, i, OBType.BULLISH)
    if ob:
        score += 1
        details["ob"] = f"{ob.low:.1f}-{ob.high:.1f}"
        sl_ref = min(sl_ref, ob.low)

    # FVG proximity
    fvg = _check_fvg_proximity(fvg_by_idx, pullback_low, i, FVGType.BULLISH)
    if fvg:
        score += 1
        details["fvg"] = f"{fvg.bottom:.1f}-{fvg.top:.1f}"
        sl_ref = min(sl_ref, fvg.bottom)

    # Sweep
    if _check_sweep_in_range(sweep_map, SweepType.BULLISH, struct_idx, i):
        score += 1
        details["sweep"] = "bullish"

    # Time bonus: NY open (7-10 AM CT) gets extra point
    if 7 <= hour <= 10:
        score += 1
        details["time"] = f"{hour}:00CT_NYopen"

    return score, details, sl_ref


def _evaluate_short(i, price, pullback_high, struct_idx,
                    swings, ob_by_idx, fvg_by_idx, sweep_map,
                    sessions, hour):
    """Evaluate confluence for a short entry."""
    score = 1
    details = {"structure": "bearish_break"}
    sl_ref = pullback_high

    trend = get_current_trend_at(i, swings)
    if trend == Trend.BEARISH:
        score += 1
        details["trend"] = "bearish"

    ob = _check_ob_proximity(ob_by_idx, pullback_high, i, OBType.BEARISH)
    if ob:
        score += 1
        details["ob"] = f"{ob.low:.1f}-{ob.high:.1f}"
        sl_ref = max(sl_ref, ob.high)

    fvg = _check_fvg_proximity(fvg_by_idx, pullback_high, i, FVGType.BEARISH)
    if fvg:
        score += 1
        details["fvg"] = f"{fvg.bottom:.1f}-{fvg.top:.1f}"
        sl_ref = max(sl_ref, fvg.top)

    if _check_sweep_in_range(sweep_map, SweepType.BEARISH, struct_idx, i):
        score += 1
        details["sweep"] = "bearish"

    if 7 <= hour <= 10:
        score += 1
        details["time"] = f"{hour}:00CT_NYopen"

    return score, details, sl_ref


def _check_ob_proximity(ob_by_idx, extreme_price, current_index,
                        ob_type, max_age=40):
    """Check if pullback extreme touched an OB zone."""
    for idx in range(current_index - 1, max(current_index - max_age, 0), -1):
        if idx not in ob_by_idx:
            continue
        for ob in ob_by_idx[idx]:
            if ob.ob_type != ob_type or not ob.is_valid:
                continue
            if ob_type == OBType.BULLISH:
                if extreme_price <= ob.high and extreme_price >= ob.low * 0.998:
                    return ob
            else:
                if extreme_price >= ob.low and extreme_price <= ob.high * 1.002:
                    return ob
    return None


def _check_fvg_proximity(fvg_by_idx, extreme_price, current_index,
                         fvg_type, max_age=30):
    """Check if pullback filled into an FVG zone."""
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


def _check_sweep_in_range(sweep_map, sweep_type, start_idx, end_idx):
    """Check if there was a sweep of the given type in the range."""
    for idx in range(start_idx, end_idx):
        if idx in sweep_map:
            for s in sweep_map[idx]:
                if s.sweep_type == sweep_type:
                    return True
    return False
