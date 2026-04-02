"""Opening Range Breakout (ORB) Strategy for Gold Futures.

Completely different approach from MS/CBR:
- No market structure analysis
- No ICT concepts
- Pure statistical edge from opening range dynamics

The first 15-30 minutes after a session opens sets the range.
60-70% of the time, gold breaks one side and runs.
We trade the breakout with SL at the other side.

Two sessions per day = two opportunities:
1. London ORB (08:00 UTC): captures London momentum
2. NY ORB (13:30 UTC): captures US market momentum
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum


class ORBDirection(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class ORBSignal:
    direction: ORBDirection
    index: int
    timestamp: object
    entry_price: float
    stop_loss: float
    take_profit: float
    sl_distance: float
    details: dict


def generate_orb_signals(
    candles: pd.DataFrame,
    london_open_hour: int = 8,      # UTC
    ny_open_hour: int = 13,         # UTC (9:30 ET ~ 13:30 UTC, use 13)
    range_minutes: int = 30,        # minutes to build the opening range
    rr_ratio: float = 1.5,         # risk-reward ratio
    min_range: float = 1.0,         # minimum range size in dollars
    max_range: float = 12.0,        # maximum range (too wide = no edge)
    breakout_buffer: float = 0.10,  # buffer beyond range for entry
) -> list[ORBSignal]:
    """Generate Opening Range Breakout signals.

    For each session (London, NY):
    1. Mark the high and low of the first `range_minutes` minutes
    2. Wait for a candle to CLOSE beyond the range (confirmed breakout)
    3. Enter in the breakout direction
    4. SL at the opposite side of the range
    5. TP at RR ratio * range size from entry
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
    utc_minutes = np.array(utc_idx.minute)
    utc_dates = utc_idx.date

    # Detect candle interval
    if n > 1:
        interval_min = int((timestamps[1] - timestamps[0]).total_seconds() / 60)
    else:
        interval_min = 5
    range_bars = max(1, range_minutes // interval_min)

    print(f"[ORB] Processing {n:,} candles ({interval_min}-min), "
          f"range={range_bars} bars, R:R={rr_ratio}")

    # Track which sessions we've already traded today
    last_london_date = None
    last_ny_date = None

    session_configs = [
        {"name": "London", "hour": london_open_hour, "last_date_attr": "london"},
        {"name": "NY", "hour": ny_open_hour, "last_date_attr": "ny"},
    ]

    for session in session_configs:
        open_hour = session["hour"]

        i = 0
        while i < n:
            hour = utc_hours[i]
            minute = utc_minutes[i]
            date = utc_dates[i]

            # Find session open candle
            if hour != open_hour or minute > interval_min:
                i += 1
                continue

            # Check if we already traded this session today
            if session["name"] == "London" and date == last_london_date:
                i += 1
                continue
            if session["name"] == "NY" and date == last_ny_date:
                i += 1
                continue

            # ── Build the opening range ──
            range_start = i
            range_end = min(i + range_bars, n)

            if range_end >= n:
                break

            range_high = max(highs[range_start:range_end])
            range_low = min(lows[range_start:range_end])
            range_size = range_high - range_low

            if range_size < min_range or range_size > max_range:
                if session["name"] == "London":
                    last_london_date = date
                else:
                    last_ny_date = date
                i = range_end
                continue

            # ── Wait for breakout (close beyond range) ──
            # Look for breakout in the next 2 hours after range forms
            breakout_window = min(range_end + (120 // interval_min), n)
            found_signal = False

            for j in range(range_end, breakout_window):
                # Bullish breakout: close above range high
                if closes[j] > range_high + breakout_buffer:
                    entry = closes[j]
                    sl = range_low - breakout_buffer
                    sl_dist = entry - sl
                    tp = entry + sl_dist * rr_ratio

                    if sl_dist > 0:
                        signals.append(ORBSignal(
                            direction=ORBDirection.LONG,
                            index=j, timestamp=timestamps[j],
                            entry_price=entry, stop_loss=sl, take_profit=tp,
                            sl_distance=sl_dist,
                            details={
                                "session": session["name"],
                                "range_high": range_high, "range_low": range_low,
                                "range_size": range_size, "strategy": "ORB",
                            },
                        ))
                        found_signal = True
                        break

                # Bearish breakout: close below range low
                if closes[j] < range_low - breakout_buffer:
                    entry = closes[j]
                    sl = range_high + breakout_buffer
                    sl_dist = sl - entry
                    tp = entry - sl_dist * rr_ratio

                    if sl_dist > 0:
                        signals.append(ORBSignal(
                            direction=ORBDirection.SHORT,
                            index=j, timestamp=timestamps[j],
                            entry_price=entry, stop_loss=sl, take_profit=tp,
                            sl_distance=sl_dist,
                            details={
                                "session": session["name"],
                                "range_high": range_high, "range_low": range_low,
                                "range_size": range_size, "strategy": "ORB",
                            },
                        ))
                        found_signal = True
                        break

            if session["name"] == "London":
                last_london_date = date
            else:
                last_ny_date = date

            i = breakout_window if not found_signal else j + 1

    signals.sort(key=lambda x: x.timestamp)
    print(f"[ORB] Generated {len(signals)} ORB signals")
    return signals
