"""Fair Value Gap (FVG / Imbalance) detection and tracking."""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum


class FVGType(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"


@dataclass
class FairValueGap:
    fvg_type: FVGType
    index: int          # index of the middle candle (candle that created the gap)
    timestamp: object
    top: float          # upper boundary of the gap
    bottom: float       # lower boundary of the gap
    size: float         # gap size in dollars
    is_filled: bool = False
    fill_pct: float = 0.0


def detect_fvgs(
    candles: pd.DataFrame,
    min_size: float = 0.50,
) -> list[FairValueGap]:
    """Detect Fair Value Gaps (3-candle imbalances).

    Bullish FVG: candle[i-1].high < candle[i+1].low
        Gap zone = [candle[i-1].high, candle[i+1].low]

    Bearish FVG: candle[i-1].low > candle[i+1].high
        Gap zone = [candle[i+1].high, candle[i-1].low]
    """
    highs = candles["high"].values
    lows = candles["low"].values
    timestamps = candles.index
    n = len(candles)
    fvgs = []

    for i in range(1, n - 1):
        # Bullish FVG: gap between candle[i-1] high and candle[i+1] low
        if highs[i - 1] < lows[i + 1]:
            gap_bottom = highs[i - 1]
            gap_top = lows[i + 1]
            size = gap_top - gap_bottom
            if size >= min_size:
                fvgs.append(FairValueGap(
                    fvg_type=FVGType.BULLISH,
                    index=i,
                    timestamp=timestamps[i],
                    top=gap_top,
                    bottom=gap_bottom,
                    size=size,
                ))

        # Bearish FVG: gap between candle[i+1] high and candle[i-1] low
        if lows[i - 1] > highs[i + 1]:
            gap_top = lows[i - 1]
            gap_bottom = highs[i + 1]
            size = gap_top - gap_bottom
            if size >= min_size:
                fvgs.append(FairValueGap(
                    fvg_type=FVGType.BEARISH,
                    index=i,
                    timestamp=timestamps[i],
                    top=gap_top,
                    bottom=gap_bottom,
                    size=size,
                ))

    return fvgs


def update_fvgs(
    fvgs: list[FairValueGap],
    candles: pd.DataFrame,
    current_index: int,
) -> list[FairValueGap]:
    """Update FVG fill status at a given candle index."""
    highs = candles["high"].values
    lows = candles["low"].values

    for fvg in fvgs:
        if fvg.is_filled or fvg.index >= current_index:
            continue

        if fvg.fvg_type == FVGType.BULLISH:
            # Bullish FVG gets filled when price comes back down into the gap
            if lows[current_index] <= fvg.top:
                penetration = fvg.top - max(lows[current_index], fvg.bottom)
                fvg.fill_pct = min(1.0, penetration / fvg.size)
                if lows[current_index] <= fvg.bottom:
                    fvg.is_filled = True
                    fvg.fill_pct = 1.0

        elif fvg.fvg_type == FVGType.BEARISH:
            # Bearish FVG gets filled when price comes back up into the gap
            if highs[current_index] >= fvg.bottom:
                penetration = min(highs[current_index], fvg.top) - fvg.bottom
                fvg.fill_pct = min(1.0, penetration / fvg.size)
                if highs[current_index] >= fvg.top:
                    fvg.is_filled = True
                    fvg.fill_pct = 1.0

    return fvgs


def find_active_fvg(
    fvgs: list[FairValueGap],
    fvg_type: FVGType,
    current_index: int,
    current_price: float,
    max_age: int = 50,
) -> FairValueGap | None:
    """Find the most recent unfilled FVG of given type near current price."""
    candidates = [
        fvg for fvg in fvgs
        if fvg.fvg_type == fvg_type
        and not fvg.is_filled
        and fvg.index < current_index
        and (current_index - fvg.index) <= max_age
    ]
    if not candidates:
        return None

    if fvg_type == FVGType.BULLISH:
        # Price should be near or in the gap (retracing down)
        candidates = [f for f in candidates if current_price <= f.top * 1.002]
    else:
        # Price should be near or in the gap (retracing up)
        candidates = [f for f in candidates if current_price >= f.bottom * 0.998]

    if candidates:
        return max(candidates, key=lambda f: f.index)
    return None
