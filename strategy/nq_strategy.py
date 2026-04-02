"""NQ Mean-Reversion + Morning Momentum Strategy.

Inspired by Phoenix (Aeromir) and Two Hour Trader (Power Trading Group).
Two setups per day on NQ/MNQ:

1. MORNING MOMENTUM (9:30-11:30 ET / 13:30-15:30 UTC):
   - Wait for first 15-min direction after open
   - Enter pullback in the direction of the initial move
   - SL below pullback low, TP at 2R
   - Works because NQ's opening drive is the strongest move of the day

2. MEAN REVERSION (any session):
   - When RSI(14) is oversold (<30) and price is below 50-EMA, look for reversal
   - When RSI(14) is overbought (>70) and price is above 50-EMA, look for reversal
   - Enter on reversal candle, SL beyond the extreme, TP at EMA
   - Works because NQ mean-reverts 60-70% of the time intraday

Both run on 5-min NQ data.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum


class NQDirection(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class NQSignal:
    direction: NQDirection
    index: int
    timestamp: object
    entry_price: float
    stop_loss: float
    take_profit: float
    sl_distance: float
    details: dict


def compute_rsi(closes, period=14):
    """Compute RSI."""
    n = len(closes)
    rsi = np.full(n, 50.0)  # default neutral

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)

    if len(gains) < period:
        return rsi

    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + losses[i-1]) / period

    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))

    return rsi


def compute_ema(data, period):
    """Compute EMA."""
    n = len(data)
    ema = np.full(n, np.nan)
    if n < period:
        return ema
    ema[period-1] = np.mean(data[:period])
    mult = 2.0 / (period + 1)
    for i in range(period, n):
        ema[i] = (data[i] - ema[i-1]) * mult + ema[i-1]
    return ema


def generate_nq_signals(
    candles: pd.DataFrame,
    rr_ratio: float = 2.0,
    rsi_period: int = 14,
    ema_period: int = 50,
    rsi_oversold: float = 30,
    rsi_overbought: float = 70,
    morning_start_hour: int = 13,   # 9:30 ET = 13:30 UTC, use 13
    morning_end_hour: int = 16,     # 11:30 ET = 15:30 UTC, use 16
    min_pullback: float = 5.0,      # minimum pullback in NQ points
) -> list[NQSignal]:
    """Generate NQ signals combining morning momentum + mean reversion."""
    signals = []
    opens = candles["open"].values
    closes = candles["close"].values
    highs = candles["high"].values
    lows = candles["low"].values
    timestamps = candles.index
    n = len(candles)

    rsi = compute_rsi(closes, rsi_period)
    ema = compute_ema(closes, ema_period)

    if candles.index.tz is None:
        utc = candles.index.tz_localize("UTC")
    else:
        utc = candles.index.tz_convert("UTC")
    utc_hours = np.array(utc.hour)
    utc_dates = utc.date

    print(f"[NQ] Processing {n:,} candles...")

    last_signal_idx = -10
    morning_traded_date = None

    # Track opening range for morning momentum
    open_range_high = None
    open_range_low = None
    open_range_date = None
    open_direction = None  # "up" or "down" based on first 15 min

    for i in range(ema_period + 5, n):
        hour = utc_hours[i]
        date = utc_dates[i]

        if i - last_signal_idx < 6:
            continue

        # ══════════════════════════════════════════════════
        # SETUP 1: MORNING MOMENTUM (13:00-16:00 UTC / 9-12 ET)
        # ══════════════════════════════════════════════════
        if morning_start_hour <= hour < morning_end_hour:

            # Build opening range from first 3 bars (15 min on 5-min chart)
            if date != open_range_date:
                if hour == morning_start_hour:
                    # Start tracking
                    open_range_high = highs[i]
                    open_range_low = lows[i]
                    open_range_date = date
                    open_direction = None
                    morning_traded_date = None
                continue

            if open_range_date == date and open_direction is None:
                # Still building range (first 3 bars = 15 min)
                open_range_high = max(open_range_high, highs[i])
                open_range_low = min(open_range_low, lows[i])

                # After 3 bars, determine direction
                bars_since_open = i
                # Count how many bars since we started tracking
                for j in range(i, max(0, i-10), -1):
                    if utc_dates[j] != date or utc_hours[j] < morning_start_hour:
                        bars_since_open = i - j - 1
                        break

                if bars_since_open >= 3:
                    if closes[i] > open_range_high - (open_range_high - open_range_low) * 0.3:
                        open_direction = "up"
                    elif closes[i] < open_range_low + (open_range_high - open_range_low) * 0.3:
                        open_direction = "down"
                    else:
                        open_direction = "neutral"
                continue

            # After range formed, look for pullback entry in the opening direction
            if open_direction and morning_traded_date != date and open_direction != "neutral":

                if open_direction == "up":
                    # Long setup: price pulls back then makes bullish candle
                    pullback_depth = open_range_high - lows[i]
                    if (pullback_depth >= min_pullback and
                            closes[i] > opens[i] and
                            closes[i] > closes[i-1]):
                        entry = closes[i]
                        sl = lows[i] - 2.0  # 2 points below pullback low
                        sl_dist = entry - sl
                        tp = entry + sl_dist * rr_ratio

                        if 5 < sl_dist < 50:
                            signals.append(NQSignal(
                                direction=NQDirection.LONG, index=i,
                                timestamp=timestamps[i], entry_price=entry,
                                stop_loss=sl, take_profit=tp, sl_distance=sl_dist,
                                details={"strategy": "NQ_MORNING", "direction": "up",
                                         "range_high": open_range_high,
                                         "range_low": open_range_low},
                            ))
                            last_signal_idx = i
                            morning_traded_date = date
                            continue

                elif open_direction == "down":
                    pullback_depth = highs[i] - open_range_low
                    if (pullback_depth >= min_pullback and
                            closes[i] < opens[i] and
                            closes[i] < closes[i-1]):
                        entry = closes[i]
                        sl = highs[i] + 2.0
                        sl_dist = sl - entry
                        tp = entry - sl_dist * rr_ratio

                        if 5 < sl_dist < 50:
                            signals.append(NQSignal(
                                direction=NQDirection.SHORT, index=i,
                                timestamp=timestamps[i], entry_price=entry,
                                stop_loss=sl, take_profit=tp, sl_distance=sl_dist,
                                details={"strategy": "NQ_MORNING", "direction": "down",
                                         "range_high": open_range_high,
                                         "range_low": open_range_low},
                            ))
                            last_signal_idx = i
                            morning_traded_date = date
                            continue

        # ══════════════════════════════════════════════════
        # SETUP 2: MEAN REVERSION (RSI + EMA, any session)
        # ══════════════════════════════════════════════════
        # Only trade during active hours (13:00-20:00 UTC / 9AM-4PM ET)
        if not (13 <= hour <= 20):
            continue

        if np.isnan(ema[i]):
            continue

        # Oversold bounce (long)
        if (rsi[i-1] < rsi_oversold and rsi[i] > rsi_oversold and
                closes[i] > opens[i] and closes[i] < ema[i]):
            entry = closes[i]
            sl = min(lows[i], lows[i-1]) - 2.0
            sl_dist = entry - sl
            tp = min(entry + sl_dist * rr_ratio, ema[i])  # target EMA or R:R
            tp_dist = tp - entry

            if 5 < sl_dist < 50 and tp_dist >= sl_dist * 1.0:
                signals.append(NQSignal(
                    direction=NQDirection.LONG, index=i,
                    timestamp=timestamps[i], entry_price=entry,
                    stop_loss=sl, take_profit=tp, sl_distance=sl_dist,
                    details={"strategy": "NQ_MR", "rsi": rsi[i],
                             "ema": ema[i], "type": "oversold_bounce"},
                ))
                last_signal_idx = i
                continue

        # Overbought fade (short)
        if (rsi[i-1] > rsi_overbought and rsi[i] < rsi_overbought and
                closes[i] < opens[i] and closes[i] > ema[i]):
            entry = closes[i]
            sl = max(highs[i], highs[i-1]) + 2.0
            sl_dist = sl - entry
            tp = max(entry - sl_dist * rr_ratio, ema[i])
            tp_dist = entry - tp

            if 5 < sl_dist < 50 and tp_dist >= sl_dist * 1.0:
                signals.append(NQSignal(
                    direction=NQDirection.SHORT, index=i,
                    timestamp=timestamps[i], entry_price=entry,
                    stop_loss=sl, take_profit=tp, sl_distance=sl_dist,
                    details={"strategy": "NQ_MR", "rsi": rsi[i],
                             "ema": ema[i], "type": "overbought_fade"},
                ))
                last_signal_idx = i

    print(f"[NQ] Generated {len(signals)} signals "
          f"(Morning: {sum(1 for s in signals if 'MORNING' in s.details.get('strategy',''))}, "
          f"MR: {sum(1 for s in signals if 'MR' in s.details.get('strategy',''))})")
    return signals
