"""Mean Reversion + Supply/Demand Zone Strategy for Gold.

Designed to trade RANGING markets where trend strategies fail.
Complements MS (trend continuation) and CBR (session reversal).

Core logic:
1. Detect supply/demand zones from prior strong rejections
2. Wait for price to return to a zone
3. Enter on rejection candle at the zone boundary
4. SL beyond the zone, TP at opposite zone or range midpoint
5. Only active when ADX < 25 (ranging market)

This strategy specifically targets the Sep-Dec 2019 type periods
where gold ranges between support and resistance.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from strategy.regime import compute_adx, compute_ema


class MRDirection(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class MRSignal:
    direction: MRDirection
    index: int
    timestamp: object
    entry_price: float
    stop_loss: float
    take_profit: float
    sl_distance: float
    details: dict


@dataclass
class Zone:
    zone_type: str   # "demand" or "supply"
    top: float
    bottom: float
    strength: int     # number of times price rejected from this zone
    index: int        # when it was created
    is_valid: bool


def find_zones(highs, lows, closes, opens, lookback=100, min_rejection_size=0.50):
    """Find supply and demand zones from strong rejection candles.

    Demand zone: strong bullish rejection candle (long lower wick, close near high)
    Supply zone: strong bearish rejection candle (long upper wick, close near low)
    """
    n = len(highs)
    zones = []

    for i in range(20, n):
        candle_range = highs[i] - lows[i]
        if candle_range < min_rejection_size:
            continue

        body = abs(closes[i] - opens[i])
        upper_wick = highs[i] - max(opens[i], closes[i])
        lower_wick = min(opens[i], closes[i]) - lows[i]

        # Demand zone: strong bullish candle with long lower wick
        # (buyers stepped in aggressively at this level)
        if lower_wick > candle_range * 0.50 and closes[i] > opens[i]:
            zones.append(Zone(
                zone_type="demand",
                top=min(opens[i], closes[i]),  # body bottom
                bottom=lows[i],
                strength=1,
                index=i,
                is_valid=True,
            ))

        # Supply zone: strong bearish candle with long upper wick
        # (sellers stepped in at this level)
        if upper_wick > candle_range * 0.50 and closes[i] < opens[i]:
            zones.append(Zone(
                zone_type="supply",
                top=highs[i],
                bottom=max(opens[i], closes[i]),  # body top
                strength=1,
                index=i,
                is_valid=True,
            ))

        # Also detect engulfing candles as strong zones
        if i > 0:
            prev_body = abs(closes[i-1] - opens[i-1])
            curr_body = body

            # Bullish engulfing = demand zone at the low
            if (closes[i-1] < opens[i-1] and closes[i] > opens[i] and
                    curr_body > prev_body * 1.2 and curr_body > min_rejection_size):
                zones.append(Zone(
                    zone_type="demand",
                    top=opens[i],
                    bottom=min(lows[i], lows[i-1]),
                    strength=2,  # engulfing is stronger
                    index=i,
                    is_valid=True,
                ))

            # Bearish engulfing = supply zone at the high
            if (closes[i-1] > opens[i-1] and closes[i] < opens[i] and
                    curr_body > prev_body * 1.2 and curr_body > min_rejection_size):
                zones.append(Zone(
                    zone_type="supply",
                    top=max(highs[i], highs[i-1]),
                    bottom=opens[i],
                    strength=2,
                    index=i,
                    is_valid=True,
                ))

    return zones


def update_zones(zones, highs, lows, closes, current_idx, max_age=200):
    """Update zone validity. A zone is invalidated when price closes through it."""
    for z in zones:
        if not z.is_valid:
            continue
        if current_idx - z.index > max_age:
            z.is_valid = False
            continue
        if z.zone_type == "demand":
            # Demand invalidated if price closes below zone bottom
            if closes[current_idx] < z.bottom:
                z.is_valid = False
        elif z.zone_type == "supply":
            # Supply invalidated if price closes above zone top
            if closes[current_idx] > z.top:
                z.is_valid = False


def generate_mr_signals(
    candles: pd.DataFrame,
    rr_ratio: float = 1.5,
    min_zone_size: float = 0.50,
    max_zone_size: float = 8.0,
    adx_threshold: float = 25,
) -> list[MRSignal]:
    """Generate mean reversion signals at supply/demand zones.

    Only trades when market is RANGING (ADX < threshold).
    """
    signals = []
    opens = candles["open"].values
    closes = candles["close"].values
    highs = candles["high"].values
    lows = candles["low"].values
    timestamps = candles.index
    n = len(candles)

    # Compute ADX for regime detection
    adx = compute_adx(highs, lows, closes, period=14)

    # Get UTC hours for session filter
    if candles.index.tz is None:
        utc_idx = candles.index.tz_localize("UTC")
    else:
        utc_idx = candles.index.tz_convert("UTC")
    utc_hours = np.array(utc_idx.hour)

    # Find zones
    zones = find_zones(highs, lows, closes, opens, min_rejection_size=min_zone_size)
    print(f"[MR] Found {len(zones)} supply/demand zones")

    last_signal_idx = -20

    for i in range(50, n):
        # Only trade in ranging markets
        if np.isnan(adx[i]) or adx[i] >= adx_threshold:
            continue

        # Session filter: London + NY (avoid illiquid Asian for MR)
        hour = utc_hours[i]
        if not (7 <= hour <= 16):
            continue

        if i - last_signal_idx < 10:
            continue

        # Update zone validity
        update_zones(zones, highs, lows, closes, i)

        price = closes[i]

        # Look for demand zone touch (long entry)
        for z in reversed(zones):  # most recent first
            if not z.is_valid or z.zone_type != "demand":
                continue
            if i - z.index < 10:  # zone must be at least 10 bars old
                continue

            zone_size = z.top - z.bottom
            if zone_size < min_zone_size or zone_size > max_zone_size:
                continue

            # Price must be at/in the zone and showing rejection
            if lows[i] <= z.top and closes[i] > z.top:
                # Rejection: wicked into zone but closed above it
                # Need a bullish candle as confirmation
                if closes[i] > opens[i]:
                    entry = closes[i]
                    sl = z.bottom - 0.50
                    sl_dist = entry - sl
                    tp = entry + sl_dist * rr_ratio

                    if 0.5 < sl_dist < 15.0:
                        signals.append(MRSignal(
                            direction=MRDirection.LONG,
                            index=i, timestamp=timestamps[i],
                            entry_price=entry, stop_loss=sl, take_profit=tp,
                            sl_distance=sl_dist,
                            details={"zone": "demand", "zone_top": z.top,
                                     "zone_bottom": z.bottom, "adx": adx[i],
                                     "strategy": "MR"},
                        ))
                        last_signal_idx = i
                        break

        # Look for supply zone touch (short entry)
        if i - last_signal_idx < 10:
            continue

        for z in reversed(zones):
            if not z.is_valid or z.zone_type != "supply":
                continue
            if i - z.index < 10:
                continue

            zone_size = z.top - z.bottom
            if zone_size < min_zone_size or zone_size > max_zone_size:
                continue

            if highs[i] >= z.bottom and closes[i] < z.bottom:
                if closes[i] < opens[i]:
                    entry = closes[i]
                    sl = z.top + 0.50
                    sl_dist = sl - entry
                    tp = entry - sl_dist * rr_ratio

                    if 0.5 < sl_dist < 15.0:
                        signals.append(MRSignal(
                            direction=MRDirection.SHORT,
                            index=i, timestamp=timestamps[i],
                            entry_price=entry, stop_loss=sl, take_profit=tp,
                            sl_distance=sl_dist,
                            details={"zone": "supply", "zone_top": z.top,
                                     "zone_bottom": z.bottom, "adx": adx[i],
                                     "strategy": "MR"},
                        ))
                        last_signal_idx = i
                        break

    print(f"[MR] Generated {len(signals)} mean reversion signals")
    return signals
