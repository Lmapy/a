"""Liquidity sweep detection: equal highs/lows, stop hunts."""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum

from strategy.market_structure import SwingPoint, SwingType


class SweepType(Enum):
    BULLISH = "bullish"   # sweeps lows then reverses up
    BEARISH = "bearish"   # sweeps highs then reverses down


@dataclass
class LiquiditySweep:
    sweep_type: SweepType
    index: int            # candle that performed the sweep
    timestamp: object
    level_swept: float    # the liquidity level that was swept
    wick_extreme: float   # how far the wick went past the level
    close_price: float    # where the candle closed


def find_equal_levels(
    swings: list[SwingPoint],
    tolerance: float = 0.50,
    swing_type: SwingType = SwingType.HIGH,
) -> list[tuple[SwingPoint, SwingPoint]]:
    """Find pairs of swing points at approximately equal prices (liquidity pools)."""
    filtered = [s for s in swings if s.swing_type == swing_type]
    pairs = []

    for i in range(len(filtered)):
        for j in range(i + 1, len(filtered)):
            if abs(filtered[i].price - filtered[j].price) <= tolerance:
                pairs.append((filtered[i], filtered[j]))

    return pairs


def detect_liquidity_sweeps(
    candles: pd.DataFrame,
    swings: list[SwingPoint],
    tolerance: float = 0.50,
    lookback: int = 50,
) -> list[LiquiditySweep]:
    """Detect liquidity sweeps: wicks beyond swing levels that close back inside.

    Bullish sweep: candle wicks below a swing low but closes above it.
    Bearish sweep: candle wicks above a swing high but closes below it.
    """
    highs = candles["high"].values
    lows = candles["low"].values
    closes = candles["close"].values
    opens = candles["open"].values
    timestamps = candles.index
    n = len(candles)

    sweeps = []

    # Pre-sort swings by type for efficient lookup
    swing_lows = [s for s in swings if s.swing_type == SwingType.LOW]
    swing_highs = [s for s in swings if s.swing_type == SwingType.HIGH]

    sl_ptr = 0
    sh_ptr = 0

    for i in range(1, n):
        # Advance pointers to find relevant swings
        while sl_ptr < len(swing_lows) and swing_lows[sl_ptr].index < i - lookback:
            sl_ptr += 1
        while sh_ptr < len(swing_highs) and swing_highs[sh_ptr].index < i - lookback:
            sh_ptr += 1

        # Check bullish sweep against recent swing lows
        for j in range(sl_ptr, len(swing_lows)):
            sl = swing_lows[j]
            if sl.index >= i:
                break
            if lows[i] < sl.price and closes[i] > sl.price:
                sweeps.append(LiquiditySweep(
                    sweep_type=SweepType.BULLISH,
                    index=i, timestamp=timestamps[i],
                    level_swept=sl.price, wick_extreme=lows[i],
                    close_price=closes[i],
                ))
                break

        # Check bearish sweep against recent swing highs
        for j in range(sh_ptr, len(swing_highs)):
            sh = swing_highs[j]
            if sh.index >= i:
                break
            if highs[i] > sh.price and closes[i] < sh.price:
                sweeps.append(LiquiditySweep(
                    sweep_type=SweepType.BEARISH,
                    index=i, timestamp=timestamps[i],
                    level_swept=sh.price, wick_extreme=highs[i],
                    close_price=closes[i],
                ))
                break

    return sweeps


def check_recent_sweep(
    sweeps: list[LiquiditySweep],
    sweep_type: SweepType,
    current_index: int,
    recency: int = 10,
) -> LiquiditySweep | None:
    """Check if there was a recent liquidity sweep of the given type."""
    candidates = [
        s for s in sweeps
        if s.sweep_type == sweep_type
        and s.index < current_index
        and (current_index - s.index) <= recency
    ]
    return candidates[-1] if candidates else None


def check_session_sweep(
    candles: pd.DataFrame,
    current_index: int,
    asian_high: float | None,
    asian_low: float | None,
) -> SweepType | None:
    """Check if current candle sweeps Asian session range."""
    if asian_high is None or asian_low is None:
        return None

    highs = candles["high"].values
    lows = candles["low"].values
    closes = candles["close"].values

    # Sweep above Asian high then close back below
    if highs[current_index] > asian_high and closes[current_index] < asian_high:
        return SweepType.BEARISH

    # Sweep below Asian low then close back above
    if lows[current_index] < asian_low and closes[current_index] > asian_low:
        return SweepType.BULLISH

    return None
