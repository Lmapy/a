"""TomTrades CBR (Candle Behavior Reversal) Model for Gold Scalping.

Based on tomtrades' methodology (@itstomtrades):
- 78% win rate, 2.12 R:R, $1.5M+ P&L
- Mean-reversion scalp during Asian-to-London transition
- "One candle, one zone, one pullback"

Core logic:
1. Identify 20+ min one-sided price expansion on 1-min chart
2. Wait for 1H candle high/low sweep or range rebalance
3. Detect market structure shift (Type 3 MSB)
4. Enter at 50% retracement of the MSB move
5. SL beyond the sweep high/low
6. TP at 1.5R (or equilibrium)
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
    expansion_size: float   # how far price expanded before reversal
    sweep_level: float      # the level that was swept
    details: dict


def detect_expansion(closes, highs, lows, start_idx, min_bars=20, min_move=1.5):
    """Detect one-sided price expansion.

    Works on any timeframe. Looks at the net move over the last min_bars bars.
    An expansion is when price moves significantly in one direction, with
    the net move being at least 60% of the total range (directional).

    Returns (direction, expansion_high, expansion_low, end_idx) or None.
    """
    if start_idx < min_bars:
        return None

    lookback = min(min_bars, start_idx)
    start = start_idx - lookback

    window_high = max(highs[start:start_idx + 1])
    window_low = min(lows[start:start_idx + 1])
    total_range = window_high - window_low

    if total_range < min_move:
        return None

    net_move = closes[start_idx] - closes[start]
    directionality = abs(net_move) / total_range if total_range > 0 else 0

    # Require at least 50% directionality (net move is majority of range)
    if directionality < 0.50:
        return None

    if net_move > 0 and net_move >= min_move:
        return ("bullish_expansion", window_high, window_low, start)
    elif net_move < 0 and abs(net_move) >= min_move:
        return ("bearish_expansion", window_high, window_low, start)

    return None


def detect_sweep(highs, lows, closes, i, level, direction, tolerance=0.30):
    """Detect if price swept a level and closed back.

    Bullish sweep (of lows): wick below level, close above
    Bearish sweep (of highs): wick above level, close below
    """
    if direction == "sweep_lows":
        # Price wicked below the level but closed above
        if lows[i] < level - tolerance and closes[i] > level:
            return True
    elif direction == "sweep_highs":
        # Price wicked above the level but closed below
        if highs[i] > level + tolerance and closes[i] < level:
            return True
    return False


def detect_msb(opens, closes, highs, lows, i, direction, lookback=6):
    """Detect Type 3 Market Structure Break.

    After a sweep, look for a structure shift:
    - Bullish MSB: after bearish expansion + low sweep, price breaks above recent swing high
    - Bearish MSB: after bullish expansion + high sweep, price breaks below recent swing low
    """
    if i < lookback + 1:
        return None

    if direction == "bullish_msb":
        # Find recent swing high in the last lookback bars
        recent_high = max(highs[i - lookback:i])
        if closes[i] > recent_high:
            # MSB confirmed - find the swing low for retracement calc
            swing_low = min(lows[i - lookback:i + 1])
            return {"msb_level": recent_high, "swing_extreme": swing_low}

    elif direction == "bearish_msb":
        recent_low = min(lows[i - lookback:i])
        if closes[i] < recent_low:
            swing_high = max(highs[i - lookback:i + 1])
            return {"msb_level": recent_low, "swing_extreme": swing_high}

    return None


def generate_cbr_signals(
    candles_1min_or_5min: pd.DataFrame,
    candles_1h: pd.DataFrame,
    asian_session_start_hour: int = 18,  # 6PM CT = start of Asian
    asian_session_end_hour: int = 2,     # 2AM CT = end of Asian
    london_open_hour: int = 2,           # 2AM CT
    rr_ratio: float = 1.5,
    min_expansion_bars: int = 4,          # 20 min on 5-min chart
    min_expansion_move: float = 1.00,    # minimum $1.00 expansion
) -> list[CBRSignal]:
    """Generate CBR signals based on TomTrades' methodology.

    The strategy trades the rebalance after one-sided expansions,
    primarily during Asian-to-London session transition.
    """
    signals = []
    opens = candles_1min_or_5min["open"].values
    closes = candles_1min_or_5min["close"].values
    highs = candles_1min_or_5min["high"].values
    lows = candles_1min_or_5min["low"].values
    timestamps = candles_1min_or_5min.index
    n = len(candles_1min_or_5min)

    # Get CT hours
    if candles_1min_or_5min.index.tz is None:
        ct_idx = candles_1min_or_5min.index.tz_localize("UTC").tz_convert("US/Central")
    else:
        ct_idx = candles_1min_or_5min.index.tz_convert("US/Central")
    ct_hours = ct_idx.hour

    # Build 1H bias: bullish if close > open on last completed 1H candle
    h1_closes = candles_1h["close"].values
    h1_opens = candles_1h["open"].values
    h1_highs = candles_1h["high"].values
    h1_lows = candles_1h["low"].values
    h1_times = candles_1h.index

    # Map entry candle index to nearest 1H candle
    h1_ptr = 0

    # Track Asian session range per day
    asian_high = None
    asian_low = None
    current_asian_date = None
    in_asian = False

    last_signal_idx = -20
    expansion_state = None
    sweep_state = None  # track detected sweep waiting for MSB

    print(f"[CBR] Processing {n:,} candles for CBR signals...")

    for i in range(30, n):
        hour = ct_hours[i]

        # ── Track Asian session range ──
        is_asian = (hour >= asian_session_start_hour) or (hour < asian_session_end_hour)
        date_key = ct_idx[i].date()

        if is_asian:
            if not in_asian or date_key != current_asian_date:
                asian_high = highs[i]
                asian_low = lows[i]
                current_asian_date = date_key
                in_asian = True
                expansion_state = None
                sweep_state = None
            else:
                asian_high = max(asian_high, highs[i])
                asian_low = min(asian_low, lows[i])
        else:
            in_asian = False

        # ── Trade during expanded CBR window ──
        trade_window = (18 <= hour <= 23) or (0 <= hour <= 13)
        if not trade_window:
            expansion_state = None
            sweep_state = None
            continue

        if i - last_signal_idx < 6:
            continue

        # ── Step 3: If we have a sweep, look for MSB in next candles ──
        if sweep_state is not None:
            sw = sweep_state
            bars_since_sweep = i - sw["sweep_idx"]

            if bars_since_sweep > 12:
                # Timeout - no MSB found
                sweep_state = None
                expansion_state = None
            else:
                msb_dir = "bearish_msb" if sw["exp_dir"] == "bullish_expansion" else "bullish_msb"
                msb = detect_msb(opens, closes, highs, lows, i, msb_dir, lookback=8)

                if msb is not None:
                    # ── Step 4: Entry at 50% retracement ──
                    if msb_dir == "bearish_msb":
                        msb_range = msb["swing_extreme"] - msb["msb_level"]
                        entry_price = msb["msb_level"] + msb_range * 0.5
                        sl_price = msb["swing_extreme"] + 0.50
                        sl_distance = sl_price - entry_price
                        tp_price = entry_price - (sl_distance * rr_ratio)
                        direction = CBRDirection.SHORT
                    else:
                        msb_range = msb["msb_level"] - msb["swing_extreme"]
                        entry_price = msb["msb_level"] - msb_range * 0.5
                        sl_price = msb["swing_extreme"] - 0.50
                        sl_distance = entry_price - sl_price
                        tp_price = entry_price + (sl_distance * rr_ratio)
                        direction = CBRDirection.LONG

                    if 0.5 < sl_distance < 15.0:
                        signals.append(CBRSignal(
                            direction=direction,
                            index=i, timestamp=timestamps[i],
                            entry_price=entry_price,
                            stop_loss=sl_price,
                            take_profit=tp_price,
                            expansion_size=sw["exp_high"] - sw["exp_low"],
                            sweep_level=sw["sweep_level"],
                            details={"expansion": sw["exp_dir"], "msb": msb,
                                     "asian_high": asian_high, "asian_low": asian_low},
                        ))
                        last_signal_idx = i
                        sweep_state = None
                        expansion_state = None
                        continue
            continue

        # ── Step 1: Detect one-sided expansion ──
        if expansion_state is None:
            exp = detect_expansion(closes, highs, lows, i,
                                   min_bars=min_expansion_bars,
                                   min_move=min_expansion_move)
            if exp is not None:
                exp_dir, exp_high, exp_low, exp_start = exp
                expansion_state = {
                    "direction": exp_dir,
                    "high": exp_high,
                    "low": exp_low,
                    "start_idx": exp_start,
                    "detected_at": i,
                }
            continue

        # ── Step 2: After expansion, look for sweep ──
        exp = expansion_state

        if exp["direction"] == "bullish_expansion":
            sweep_level = exp["high"]
            if asian_high is not None:
                sweep_level = max(sweep_level, asian_high)

            if detect_sweep(highs, lows, closes, i, sweep_level, "sweep_highs"):
                sweep_state = {
                    "exp_dir": exp["direction"],
                    "exp_high": exp["high"],
                    "exp_low": exp["low"],
                    "sweep_level": sweep_level,
                    "sweep_idx": i,
                }
                # Also check MSB on same candle
                continue

        elif exp["direction"] == "bearish_expansion":
            sweep_level = exp["low"]
            if asian_low is not None:
                sweep_level = min(sweep_level, asian_low)

            if detect_sweep(highs, lows, closes, i, sweep_level, "sweep_lows"):
                sweep_state = {
                    "exp_dir": exp["direction"],
                    "exp_high": exp["high"],
                    "exp_low": exp["low"],
                    "sweep_level": sweep_level,
                    "sweep_idx": i,
                }
                continue

        # Expire expansion after 30 bars
        if i - exp["detected_at"] > 30:
            expansion_state = None

    print(f"[CBR] Generated {len(signals)} CBR signals")
    return signals
