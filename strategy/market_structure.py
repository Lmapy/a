"""Market structure detection: swing points, BOS, CHoCH, trend identification."""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum


class SwingType(Enum):
    HIGH = "high"
    LOW = "low"


class Trend(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class StructureBreak(Enum):
    BOS_BULLISH = "bos_bullish"       # trend continuation upward
    BOS_BEARISH = "bos_bearish"       # trend continuation downward
    CHOCH_BULLISH = "choch_bullish"   # reversal to bullish
    CHOCH_BEARISH = "choch_bearish"   # reversal to bearish
    NONE = "none"


@dataclass
class SwingPoint:
    index: int          # position in the candles array
    timestamp: object   # datetime
    price: float
    swing_type: SwingType


@dataclass
class StructureEvent:
    index: int
    timestamp: object
    break_type: StructureBreak
    level_broken: float
    trend_before: Trend
    trend_after: Trend


def detect_swing_points(candles: pd.DataFrame, lookback: int = 5) -> list[SwingPoint]:
    """Detect swing highs and swing lows using a rolling window.

    A swing high at index i requires:
        high[i] > high[j] for all j in [i-lookback, i+lookback], j != i

    A swing low at index i requires:
        low[i] < low[j] for all j in [i-lookback, i+lookback], j != i
    """
    highs = candles["high"].values
    lows = candles["low"].values
    timestamps = candles.index if isinstance(candles.index, pd.DatetimeIndex) else candles["time"]
    n = len(candles)
    swings = []

    for i in range(lookback, n - lookback):
        # Check swing high
        window_highs = highs[i - lookback: i + lookback + 1]
        if highs[i] == window_highs.max() and np.sum(window_highs == highs[i]) == 1:
            swings.append(SwingPoint(
                index=i, timestamp=timestamps[i],
                price=highs[i], swing_type=SwingType.HIGH
            ))

        # Check swing low
        window_lows = lows[i - lookback: i + lookback + 1]
        if lows[i] == window_lows.min() and np.sum(window_lows == lows[i]) == 1:
            swings.append(SwingPoint(
                index=i, timestamp=timestamps[i],
                price=lows[i], swing_type=SwingType.LOW
            ))

    # Sort by index to maintain chronological order
    swings.sort(key=lambda s: s.index)
    return swings


def identify_trend(swing_highs: list[SwingPoint], swing_lows: list[SwingPoint]) -> Trend:
    """Determine trend from the last two swing highs and swing lows.

    Bullish: HH + HL (higher high, higher low)
    Bearish: LH + LL (lower high, lower low)
    """
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return Trend.NEUTRAL

    hh = swing_highs[-1].price > swing_highs[-2].price  # higher high
    hl = swing_lows[-1].price > swing_lows[-2].price     # higher low
    lh = swing_highs[-1].price < swing_highs[-2].price   # lower high
    ll = swing_lows[-1].price < swing_lows[-2].price     # lower low

    if hh and hl:
        return Trend.BULLISH
    elif lh and ll:
        return Trend.BEARISH
    return Trend.NEUTRAL


def detect_structure_breaks(
    candles: pd.DataFrame,
    swings: list[SwingPoint],
) -> list[StructureEvent]:
    """Detect Break of Structure (BOS) and Change of Character (CHoCH).

    Optimized: only checks candles between consecutive swing points rather
    than iterating over every single candle.
    """
    events = []
    closes = candles["close"].values
    timestamps = candles.index

    sh_list = [s for s in swings if s.swing_type == SwingType.HIGH]
    sl_list = [s for s in swings if s.swing_type == SwingType.LOW]

    if len(sh_list) < 2 or len(sl_list) < 2:
        return events

    current_trend = Trend.NEUTRAL
    sh_ptr = 0
    sl_ptr = 0
    n = len(candles)

    # Process swing-by-swing: check candles between each swing point
    for swing_idx in range(len(swings)):
        sw = swings[swing_idx]

        # Update pointers
        if sw.swing_type == SwingType.HIGH:
            sh_ptr = sh_list.index(sw) + 1
        else:
            sl_ptr = sl_list.index(sw) + 1

        if sh_ptr < 2 or sl_ptr < 2:
            continue

        trend = identify_trend(sh_list[:sh_ptr], sl_list[:sl_ptr])
        last_sh = sh_list[sh_ptr - 1]
        last_sl = sl_list[sl_ptr - 1]

        # Determine range to check: from this swing to next swing (or end)
        start_i = sw.index + 1
        end_i = swings[swing_idx + 1].index if swing_idx + 1 < len(swings) else min(start_i + 50, n)

        for i in range(start_i, min(end_i + 1, n)):
            if current_trend == Trend.BEARISH or (current_trend == Trend.NEUTRAL and trend == Trend.BEARISH):
                if last_sh.index < i and closes[i] > last_sh.price:
                    events.append(StructureEvent(
                        index=i, timestamp=timestamps[i],
                        break_type=StructureBreak.CHOCH_BULLISH,
                        level_broken=last_sh.price,
                        trend_before=Trend.BEARISH, trend_after=Trend.BULLISH,
                    ))
                    current_trend = Trend.BULLISH
                    break

            if current_trend == Trend.BULLISH or (current_trend == Trend.NEUTRAL and trend == Trend.BULLISH):
                if last_sl.index < i and closes[i] < last_sl.price:
                    events.append(StructureEvent(
                        index=i, timestamp=timestamps[i],
                        break_type=StructureBreak.CHOCH_BEARISH,
                        level_broken=last_sl.price,
                        trend_before=Trend.BULLISH, trend_after=Trend.BEARISH,
                    ))
                    current_trend = Trend.BEARISH
                    break

            # BOS
            if trend == Trend.BULLISH and current_trend == Trend.BULLISH:
                if sh_ptr >= 2:
                    prev_sh = sh_list[sh_ptr - 2]
                    if prev_sh.index < i and closes[i] > prev_sh.price and prev_sh.price < last_sh.price:
                        events.append(StructureEvent(
                            index=i, timestamp=timestamps[i],
                            break_type=StructureBreak.BOS_BULLISH,
                            level_broken=prev_sh.price,
                            trend_before=Trend.BULLISH, trend_after=Trend.BULLISH,
                        ))
                        break

            elif trend == Trend.BEARISH and current_trend == Trend.BEARISH:
                if sl_ptr >= 2:
                    prev_sl = sl_list[sl_ptr - 2]
                    if prev_sl.index < i and closes[i] < prev_sl.price and prev_sl.price > last_sl.price:
                        events.append(StructureEvent(
                            index=i, timestamp=timestamps[i],
                            break_type=StructureBreak.BOS_BEARISH,
                            level_broken=prev_sl.price,
                            trend_before=Trend.BEARISH, trend_after=Trend.BEARISH,
                        ))
                        break

        if current_trend == Trend.NEUTRAL:
            current_trend = trend

    return events


def get_current_trend_at(
    candle_index: int,
    swings: list[SwingPoint],
) -> Trend:
    """Get the trend at a specific candle index based on confirmed swings."""
    sh = [s for s in swings if s.swing_type == SwingType.HIGH and s.index <= candle_index]
    sl = [s for s in swings if s.swing_type == SwingType.LOW and s.index <= candle_index]
    return identify_trend(sh, sl)


def get_last_swing(
    swing_type: SwingType,
    before_index: int,
    swings: list[SwingPoint],
) -> SwingPoint | None:
    """Get the most recent swing of a given type before a candle index."""
    candidates = [s for s in swings if s.swing_type == swing_type and s.index < before_index]
    return candidates[-1] if candidates else None
