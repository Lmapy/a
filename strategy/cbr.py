"""TomTrades CBR / AMD Type 3 Strategy for Gold - Improved Implementation.

Key teachings applied:
- "One candle. One zone. One pullback. That's all I trade."
- "Trading got easy when I stopped trying to guess where the market is going
   and started reacting to where it is."
- 1H bias + 1-min execution
- 20+ min one-sided expansion → sweep → MSB → 50% retracement entry
- Gold sweeps the Asian range 60-70% of the time at London open

Sources:
- FX Replay: fxreplay.com/strategies/tomtrades-cbr-model
- Type 3 AMD: Scribd document 907250211
- Asia Session Reversal: TradingView/CSROlPWp
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


def get_1h_bias(candles_1h, utc_time):
    """Get the 1H candle bias at a given time.

    Returns 'bullish', 'bearish', or 'neutral'.
    Uses the LAST COMPLETED 1H candle (not current).
    """
    if candles_1h is None or len(candles_1h) == 0:
        return "neutral"

    # Find the last 1H candle that closed before this time
    mask = candles_1h.index < utc_time
    if not mask.any():
        return "neutral"

    last_1h = candles_1h[mask].iloc[-1]
    body = last_1h["close"] - last_1h["open"]
    candle_range = last_1h["high"] - last_1h["low"]

    if candle_range == 0:
        return "neutral"

    body_ratio = abs(body) / candle_range

    # Need a decisive candle (body > 40% of range)
    if body_ratio < 0.40:
        return "neutral"

    return "bullish" if body > 0 else "bearish"


def detect_expansion(closes, highs, lows, i, min_bars=20, min_move=1.5):
    """Detect 20+ min one-sided price expansion.

    Measures net directional move over last min_bars bars.
    Requires at least 50% directionality (net move / total range).
    """
    if i < min_bars:
        return None

    start = i - min_bars
    window_high = max(highs[start:i + 1])
    window_low = min(lows[start:i + 1])
    total_range = window_high - window_low

    if total_range < min_move:
        return None

    net_move = closes[i] - closes[start]
    directionality = abs(net_move) / total_range if total_range > 0 else 0

    if directionality < 0.50:
        return None

    if net_move > 0 and net_move >= min_move:
        return ("bullish_expansion", window_high, window_low, start)
    elif net_move < 0 and abs(net_move) >= min_move:
        return ("bearish_expansion", window_high, window_low, start)

    return None


def detect_sweep(highs, lows, closes, i, level, direction, lookback=15):
    """Multi-bar sweep detection.

    Checks if price wicked beyond the level in last `lookback` bars
    AND current bar closes back inside.
    """
    if direction == "sweep_low":
        swept = any(lows[j] < level - 0.10 for j in range(max(0, i - lookback), i + 1))
        return swept and closes[i] > level
    elif direction == "sweep_high":
        swept = any(highs[j] > level + 0.10 for j in range(max(0, i - lookback), i + 1))
        return swept and closes[i] < level
    return False


def detect_msb(closes, highs, lows, i, direction, lookback=15):
    """Detect Market Structure Break - close beyond recent structure.

    After bullish sweep: close above recent swing high = bullish MSB
    After bearish sweep: close below recent swing low = bearish MSB
    Returns the MSB range for 50% retracement calculation.
    """
    if i < lookback + 1:
        return None

    if direction == "bullish":
        recent_high = max(highs[i - lookback:i])
        if closes[i] > recent_high:
            swing_low = min(lows[i - lookback:i])
            msb_range = closes[i] - swing_low
            return {
                "msb_high": closes[i],
                "swing_low": swing_low,
                "range": msb_range,
                "fifty_pct": swing_low + msb_range * 0.5,
            }

    elif direction == "bearish":
        recent_low = min(lows[i - lookback:i])
        if closes[i] < recent_low:
            swing_high = max(highs[i - lookback:i])
            msb_range = swing_high - closes[i]
            return {
                "msb_low": closes[i],
                "swing_high": swing_high,
                "range": msb_range,
                "fifty_pct": swing_high - msb_range * 0.5,
            }

    return None


def generate_cbr_signals(
    candles: pd.DataFrame,
    candles_1h: pd.DataFrame,
    min_expansion_bars: int = 20,
    min_expansion_move: float = 1.50,
    rr_ratio: float = 2.0,
    bos_lookback: int = 15,
) -> list[CBRSignal]:
    """Generate CBR signals with all TomTrades teachings applied.

    Flow:
    1. Check 1H bias (only trade in direction of 1H candle)
    2. Detect 20+ min expansion
    3. Wait for sweep of expansion extreme
    4. Confirm MSB (structure shift)
    5. Entry at 50% retracement of MSB move (limit order)
    6. SL beyond the sweep extreme
    7. TP at 1.5-2R
    """
    signals = []
    opens = candles["open"].values
    closes = candles["close"].values
    highs = candles["high"].values
    lows = candles["low"].values
    timestamps = candles.index
    n = len(candles)

    if candles.index.tz is None:
        utc_idx = candles.index.tz_localize("UTC")
    else:
        utc_idx = candles.index.tz_convert("UTC")
    utc_hours = np.array(utc_idx.hour)

    print(f"[CBR] Processing {n:,} candles...")

    last_signal_idx = -200
    state = "looking"  # looking -> expansion_found -> sweep_found -> waiting_msb
    exp_data = None
    sweep_data = None

    for i in range(30, n):
        hour = utc_hours[i]

        # ── KILL ZONE: 00:00-12:00 UTC (Asia + London + early NY) ──
        if not (0 <= hour <= 12):
            state = "looking"
            exp_data = None
            sweep_data = None
            continue

        if i - last_signal_idx < 120:
            continue

        price = closes[i]

        # ── STATE: WAITING FOR MSB AFTER SWEEP ──
        if state == "waiting_msb" and sweep_data is not None:
            bars_since = i - sweep_data["idx"]
            if bars_since > 60:
                state = "looking"
                sweep_data = None
                continue

            msb_dir = "bullish" if sweep_data["reversal"] == "long" else "bearish"
            msb = detect_msb(closes, highs, lows, i, msb_dir, lookback=bos_lookback)

            if msb is not None:
                # ── Check 1H bias alignment ──
                # STRICT: only trade when 1H bias is STRONG and aligned
                # This is the key filter TomTrades uses
                bias = get_1h_bias(candles_1h, utc_idx[i])
                if msb_dir == "bullish" and bias != "bullish":
                    state = "looking"
                    sweep_data = None
                    continue
                if msb_dir == "bearish" and bias != "bearish":
                    state = "looking"
                    sweep_data = None
                    continue

                # ── ENTRY at 50% of MSB move ──
                entry_price = msb["fifty_pct"]
                sl_buffer = 0.50

                if msb_dir == "bullish":
                    sl_price = msb["swing_low"] - sl_buffer
                    sl_dist = entry_price - sl_price
                    tp_price = entry_price + sl_dist * rr_ratio
                    direction = CBRDirection.LONG
                else:
                    sl_price = msb["swing_high"] + sl_buffer
                    sl_dist = sl_price - entry_price
                    tp_price = entry_price - sl_dist * rr_ratio
                    direction = CBRDirection.SHORT

                if 0.5 < sl_dist < 20.0 and sl_dist * rr_ratio > 0.5:
                    signals.append(CBRSignal(
                        direction=direction,
                        index=i, timestamp=timestamps[i],
                        entry_price=entry_price,
                        stop_loss=sl_price,
                        take_profit=tp_price,
                        expansion_size=exp_data["size"] if exp_data else 0,
                        sweep_level=sweep_data["level"],
                        details={
                            "msb": msb, "bias": bias,
                            "sweep_dir": sweep_data["reversal"],
                            "entry_type": "50pct_msb_limit",
                        },
                    ))
                    last_signal_idx = i

                state = "looking"
                sweep_data = None
                exp_data = None
            continue

        # ── STATE: EXPANSION FOUND, LOOKING FOR SWEEP ──
        if state == "expansion_found" and exp_data is not None:
            bars_since = i - exp_data["idx"]
            if bars_since > 60:
                state = "looking"
                exp_data = None
                continue

            # Look for sweep of expansion extreme
            if exp_data["dir"] == "bullish_expansion":
                # After bullish expansion, sweep HIGH then reverse down
                if detect_sweep(highs, lows, closes, i, exp_data["high"], "sweep_high"):
                    sweep_data = {
                        "level": exp_data["high"],
                        "extreme": max(highs[max(0, i-15):i+1]),
                        "idx": i,
                        "reversal": "short",
                    }
                    state = "waiting_msb"
            elif exp_data["dir"] == "bearish_expansion":
                # After bearish expansion, sweep LOW then reverse up
                if detect_sweep(highs, lows, closes, i, exp_data["low"], "sweep_low"):
                    sweep_data = {
                        "level": exp_data["low"],
                        "extreme": min(lows[max(0, i-15):i+1]),
                        "idx": i,
                        "reversal": "long",
                    }
                    state = "waiting_msb"
            continue

        # ── STATE: LOOKING FOR EXPANSION ──
        exp = detect_expansion(closes, highs, lows, i,
                               min_bars=min_expansion_bars,
                               min_move=min_expansion_move)
        if exp is not None:
            exp_dir, exp_high, exp_low, exp_start = exp
            exp_data = {
                "dir": exp_dir,
                "high": exp_high,
                "low": exp_low,
                "size": exp_high - exp_low,
                "idx": i,
            }
            state = "expansion_found"

    print(f"[CBR] Generated {len(signals)} signals")
    return signals
