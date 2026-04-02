"""TomTrades CBR / AMD Type 3 Midline Strategy for Gold.

Corrected implementation based on deep research:
- FX Replay CBR model breakdown
- Type 3 AMD Midline Strategy (Scribd)
- Asia Session Reversal Strategy (TradingView)

Core Logic (AMD Type 3):
1. ACCUMULATION: Mark the Asian session range (high/low/midline)
2. MANIPULATION: Wait for price to sweep Asian high or low
3. After sweep, confirm Break of Structure (BOS)
4. ENTRY: Limit order at 50% of ASIAN RANGE (the midline)
5. CONFIRMATION: M/W pattern at the midline (optional, improves WR)
6. SL below the sweep low / above the sweep high
7. TP at opposite side of Asian range, or 1.5-3R

Time Window: 01:00-02:00 UTC (second hour of Asia / SGE open)
Extended: London open (08:00 UTC) can also sweep Asian range
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum


class CBRDirection(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class CBRSignal:
    direction: CBRDirection
    index: int
    timestamp: object
    entry_price: float
    stop_loss: float
    take_profit: float
    expansion_size: float
    sweep_level: float
    details: dict


def compute_asian_range_for_cbr(candles: pd.DataFrame) -> dict:
    """Compute Asian session range for each trading day.

    Asian session: 22:00 - 08:00 UTC (Tokyo + Sydney + SGE)
    Returns dict mapping trading_date -> (asian_high, asian_low, midline)
    """
    if candles.index.tz is None:
        utc_idx = candles.index.tz_localize("UTC")
    else:
        utc_idx = candles.index.tz_convert("UTC")

    highs = candles["high"].values
    lows = candles["low"].values
    hours_arr = np.array(utc_idx.hour)
    dates_arr = np.array(utc_idx.date)

    # Asian session: 22:00-08:00 UTC
    asian_mask = (hours_arr >= 22) | (hours_arr < 8)

    # Shift evening dates forward one day
    td_vals = dates_arr.copy()
    for i in range(len(td_vals)):
        if hours_arr[i] >= 22:
            td_vals[i] = (pd.Timestamp(td_vals[i]) + pd.Timedelta(days=1)).date()

    ranges = {}
    current_date = None
    day_high = None
    day_low = None

    for i in range(len(candles)):
        if not asian_mask[i]:
            continue

        d = td_vals[i]
        if d != current_date:
            if current_date is not None and day_high is not None:
                midline = (day_high + day_low) / 2
                ranges[current_date] = {
                    "high": day_high, "low": day_low, "midline": midline,
                    "range_size": day_high - day_low,
                }
            current_date = d
            day_high = highs[i]
            day_low = lows[i]
        else:
            day_high = max(day_high, highs[i])
            day_low = min(day_low, lows[i])

    # Last day
    if current_date is not None and day_high is not None:
        midline = (day_high + day_low) / 2
        ranges[current_date] = {
            "high": day_high, "low": day_low, "midline": midline,
            "range_size": day_high - day_low,
        }

    return ranges


def detect_sweep_of_level(highs, lows, closes, i, level, direction, tolerance=0.10, lookback=10):
    """Detect if price swept a level and then closed back inside.

    Multi-bar sweep: checks if any bar in the last `lookback` bars
    wicked beyond the level, AND the current bar closes back inside.
    This handles 1-min bars where the sweep and close-back happen
    on different candles.
    """
    if direction == "sweep_low":
        # Was the level swept recently? (any recent bar wicked below)
        swept = False
        for j in range(max(0, i - lookback), i + 1):
            if lows[j] < level - tolerance:
                swept = True
                break
        # And current bar closes back above the level
        return swept and closes[i] > level

    elif direction == "sweep_high":
        swept = False
        for j in range(max(0, i - lookback), i + 1):
            if highs[j] > level + tolerance:
                swept = True
                break
        return swept and closes[i] < level

    return False


def detect_bos(closes, highs, lows, i, direction, lookback=5):
    """Detect Break of Structure after a sweep.

    After bullish sweep (swept lows): look for close above recent swing high
    After bearish sweep (swept highs): look for close below recent swing low
    """
    if i < lookback + 1:
        return None

    if direction == "bullish_bos":
        recent_high = max(highs[i - lookback:i])
        if closes[i] > recent_high:
            return {"level": recent_high, "extreme": min(lows[i - lookback:i])}
    elif direction == "bearish_bos":
        recent_low = min(lows[i - lookback:i])
        if closes[i] < recent_low:
            return {"level": recent_low, "extreme": max(highs[i - lookback:i])}
    return None


def generate_cbr_signals(
    candles: pd.DataFrame,
    candles_1h: pd.DataFrame,
    min_range_size: float = 2.00,   # min Asian range to trade ($2.00)
    rr_ratio: float = 2.0,         # R:R ratio
    bos_lookback: int = 20,        # bars to look back for BOS (20 min on 1-min)
) -> list[CBRSignal]:
    """Generate AMD Type 3 signals.

    Full sequence:
    1. Mark Asian range (accumulation)
    2. Wait for sweep of Asian high or low (manipulation)
    3. Confirm BOS after sweep
    4. Place limit order at Asian midline (50% of range)
    5. SL beyond sweep extreme
    6. TP at opposite side of range or R:R ratio
    """
    signals = []
    opens = candles["open"].values
    closes = candles["close"].values
    highs = candles["high"].values
    lows = candles["low"].values
    timestamps = candles.index
    n = len(candles)

    # Get UTC hours for session detection
    if candles.index.tz is None:
        utc_idx = candles.index.tz_localize("UTC")
    else:
        utc_idx = candles.index.tz_convert("UTC")
    utc_hours = utc_idx.hour
    utc_dates = utc_idx.date

    # Compute Asian ranges
    asian_ranges = compute_asian_range_for_cbr(candles)

    print(f"[CBR] {len(asian_ranges)} Asian ranges computed")
    print(f"[CBR] Processing {n:,} candles for AMD Type 3 signals...")

    last_signal_idx = -200
    sweep_state = None  # tracks detected sweep waiting for BOS + midline entry

    for i in range(10, n):
        hour = utc_hours[i]
        date = utc_dates[i]

        # Determine which Asian range to use:
        # - Hours 7-10 (London): use TODAY's completed Asian range
        # - Hours 1-2 (Asia 2nd hour): use YESTERDAY's completed range
        #   (today's range is still forming)
        if hour >= 7:
            range_date = date
        else:
            # Use previous day's range
            range_date = (pd.Timestamp(date) - pd.Timedelta(days=1)).date()

        ar = asian_ranges.get(range_date)
        if ar is None:
            continue

        # Skip if Asian range is too small (no clear accumulation)
        if ar["range_size"] < min_range_size:
            continue

        asian_high = ar["high"]
        asian_low = ar["low"]
        midline = ar["midline"]

        # ── KILL ZONE ──
        # Window 1: Asia 2nd hour + London open (00:00-12:00 UTC)
        # Extended to allow sweep+BOS sequence to complete
        in_kill_zone = (0 <= hour <= 12)
        if not in_kill_zone:
            sweep_state = None
            continue

        if i - last_signal_idx < 120:  # 2 hour cooldown
            continue

        price = closes[i]

        # ── STATE: Looking for BOS after sweep ──
        if sweep_state is not None:
            sw = sweep_state
            bars_since_sweep = i - sw["sweep_idx"]

            # Timeout: 60 bars (60 min on 1-min, 300 min on 5-min)
            if bars_since_sweep > 60:
                sweep_state = None
                continue

            if sw["direction"] == "bullish":
                # After sweeping Asian low, look for bullish BOS
                bos = detect_bos(closes, highs, lows, i, "bullish_bos", lookback=bos_lookback)
                if bos is not None:
                    # Entry at Asian midline (limit order)
                    entry_price = midline
                    sl_price = sw["sweep_extreme"] - 0.50  # below the sweep low
                    sl_distance = entry_price - sl_price

                    if sl_distance <= 0 or sl_distance > 20:
                        sweep_state = None
                        continue

                    # TP: opposite side of range, or R:R
                    tp_by_range = asian_high  # target opposite side
                    tp_by_rr = entry_price + sl_distance * rr_ratio
                    tp_price = min(tp_by_range, tp_by_rr)  # use closer target
                    tp_distance = tp_price - entry_price

                    if tp_distance >= sl_distance * 1.2:  # at least 1.2R
                        signals.append(CBRSignal(
                            direction=CBRDirection.LONG,
                            index=i, timestamp=timestamps[i],
                            entry_price=entry_price,
                            stop_loss=sl_price,
                            take_profit=tp_price,
                            expansion_size=ar["range_size"],
                            sweep_level=sw["sweep_level"],
                            details={
                                "asian_high": asian_high, "asian_low": asian_low,
                                "midline": midline, "sweep": "low_swept",
                                "bos": bos, "entry_type": "midline_limit",
                            },
                        ))
                        last_signal_idx = i
                        sweep_state = None
                        continue

            elif sw["direction"] == "bearish":
                # After sweeping Asian high, look for bearish BOS
                bos = detect_bos(closes, highs, lows, i, "bearish_bos", lookback=bos_lookback)
                if bos is not None:
                    entry_price = midline
                    sl_price = sw["sweep_extreme"] + 0.50
                    sl_distance = sl_price - entry_price

                    if sl_distance <= 0 or sl_distance > 20:
                        sweep_state = None
                        continue

                    tp_by_range = asian_low
                    tp_by_rr = entry_price - sl_distance * rr_ratio
                    tp_price = max(tp_by_range, tp_by_rr)
                    tp_distance = entry_price - tp_price

                    if tp_distance >= sl_distance * 1.2:
                        signals.append(CBRSignal(
                            direction=CBRDirection.SHORT,
                            index=i, timestamp=timestamps[i],
                            entry_price=entry_price,
                            stop_loss=sl_price,
                            take_profit=tp_price,
                            expansion_size=ar["range_size"],
                            sweep_level=sw["sweep_level"],
                            details={
                                "asian_high": asian_high, "asian_low": asian_low,
                                "midline": midline, "sweep": "high_swept",
                                "bos": bos, "entry_type": "midline_limit",
                            },
                        ))
                        last_signal_idx = i
                        sweep_state = None
                        continue

            continue

        # ── DETECT SWEEP OF ASIAN RANGE ──
        # Bullish setup: price sweeps Asian LOW then reverses up
        swept_low = detect_sweep_of_level(highs, lows, closes, i, asian_low, "sweep_low")
        swept_high = detect_sweep_of_level(highs, lows, closes, i, asian_high, "sweep_high")

        if swept_low:
            sweep_state = {
                "direction": "bullish",
                "sweep_level": asian_low,
                "sweep_extreme": lows[i],  # the actual wick low
                "sweep_idx": i,
            }
            continue

        # Bearish setup: price sweeps Asian HIGH then reverses down
        if swept_high:
            sweep_state = {
                "direction": "bearish",
                "sweep_level": asian_high,
                "sweep_extreme": highs[i],
                "sweep_idx": i,
            }
            continue

    print(f"[CBR] Generated {len(signals)} AMD Type 3 signals")
    return signals
