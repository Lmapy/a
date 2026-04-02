"""Order block identification and tracking."""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum

from strategy.market_structure import (
    SwingPoint, SwingType, StructureEvent, StructureBreak,
)


class OBType(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"


@dataclass
class OrderBlock:
    ob_type: OBType
    index: int          # index of the OB candle
    timestamp: object
    high: float         # top of OB zone
    low: float          # bottom of OB zone
    is_valid: bool = True
    is_mitigated: bool = False


def detect_order_blocks(
    candles: pd.DataFrame,
    structure_events: list[StructureEvent],
    max_age: int = 100,
) -> list[OrderBlock]:
    """Detect order blocks based on structure breaks.

    Bullish OB: The last bearish candle before a bullish BOS/CHoCH.
    Bearish OB: The last bullish candle before a bearish BOS/CHoCH.
    """
    opens = candles["open"].values
    closes = candles["close"].values
    highs = candles["high"].values
    lows = candles["low"].values
    timestamps = candles.index

    order_blocks = []

    for event in structure_events:
        idx = event.index

        if event.break_type in (StructureBreak.BOS_BULLISH, StructureBreak.CHOCH_BULLISH):
            # Find last bearish candle before this break
            for j in range(idx - 1, max(idx - 20, 0), -1):
                if closes[j] < opens[j]:  # bearish candle
                    order_blocks.append(OrderBlock(
                        ob_type=OBType.BULLISH,
                        index=j,
                        timestamp=timestamps[j],
                        high=highs[j],
                        low=lows[j],
                    ))
                    break

        elif event.break_type in (StructureBreak.BOS_BEARISH, StructureBreak.CHOCH_BEARISH):
            # Find last bullish candle before this break
            for j in range(idx - 1, max(idx - 20, 0), -1):
                if closes[j] > opens[j]:  # bullish candle
                    order_blocks.append(OrderBlock(
                        ob_type=OBType.BEARISH,
                        index=j,
                        timestamp=timestamps[j],
                        high=highs[j],
                        low=lows[j],
                    ))
                    break

    return order_blocks


def update_order_blocks(
    order_blocks: list[OrderBlock],
    candles: pd.DataFrame,
    current_index: int,
    max_age: int = 100,
) -> list[OrderBlock]:
    """Update OB validity and mitigation status at a given candle index."""
    closes = candles["close"].values
    highs = candles["high"].values
    lows = candles["low"].values

    for ob in order_blocks:
        if not ob.is_valid:
            continue

        # Expire old OBs
        if current_index - ob.index > max_age:
            ob.is_valid = False
            continue

        if current_index <= ob.index:
            continue

        if ob.ob_type == OBType.BULLISH:
            # Invalidated if price closes below OB low
            if closes[current_index] < ob.low:
                ob.is_valid = False
            # Mitigated if price enters the zone (comes down to OB)
            elif lows[current_index] <= ob.high and not ob.is_mitigated:
                ob.is_mitigated = True

        elif ob.ob_type == OBType.BEARISH:
            # Invalidated if price closes above OB high
            if closes[current_index] > ob.high:
                ob.is_valid = False
            # Mitigated if price enters the zone (comes up to OB)
            elif highs[current_index] >= ob.low and not ob.is_mitigated:
                ob.is_mitigated = True

    return order_blocks


def find_active_ob(
    order_blocks: list[OrderBlock],
    ob_type: OBType,
    current_index: int,
    current_price: float,
) -> OrderBlock | None:
    """Find the most recent valid, unmitigated OB of given type near current price."""
    candidates = [
        ob for ob in order_blocks
        if ob.ob_type == ob_type
        and ob.is_valid
        and not ob.is_mitigated
        and ob.index < current_index
    ]
    if not candidates:
        return None

    # For bullish OB, we want price to retrace DOWN to the OB (price near or in the zone)
    if ob_type == OBType.BULLISH:
        candidates = [ob for ob in candidates if current_price <= ob.high * 1.002]
        if candidates:
            return max(candidates, key=lambda ob: ob.index)  # most recent

    # For bearish OB, we want price to retrace UP to the OB
    elif ob_type == OBType.BEARISH:
        candidates = [ob for ob in candidates if current_price >= ob.low * 0.998]
        if candidates:
            return max(candidates, key=lambda ob: ob.index)

    return None
