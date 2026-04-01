"""Break of Structure (BoS) strategy.

Identifies swing highs and lows, detects structural breaks, and enters
on pullbacks targeting the next swing level.

Entry logic:
  1. Detect swing highs (SH) and swing lows (SL) using N-bar pivots.
  2. Bullish BoS: price closes above a recent SH → wait for pullback
     to the broken SH level → enter long.
  3. Bearish BoS: price closes below a recent SL → wait for pullback
     to the broken SL level → enter short.

Exit logic:
  - Take profit at the next SH above entry (longs) or next SL below (shorts).
  - Stop loss behind the most recent SL before the BoS (longs)
    or the most recent SH before the BoS (shorts).
  - If no structural target is found, use a fixed R:R ratio.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.data.models import Bar, INSTRUMENT_SPECS, MarketRegime, Position, Signal, SignalDirection
from src.strategy.base import BaseStrategy


@dataclass
class SwingPoint:
    """A swing high or swing low."""
    price: float
    bar_index: int  # absolute bar count when detected
    kind: str  # "high" or "low"
    timestamp: object = None


@dataclass
class BoSSetup:
    """A pending BoS entry after structure breaks."""
    direction: str  # "long" or "short"
    broken_level: float  # the SH/SL that was broken
    stop_level: float  # SL before BoS (longs) or SH before BoS (shorts)
    target_level: float | None  # next SH above (longs) or next SL below (shorts)
    created_bar: int  # bar count when setup was created
    triggered: bool = False


class BoSStrategy(BaseStrategy):
    """Trade Break of Structure setups with pullback entries.

    Pure price action — no volume indicators needed.
    """

    name = "bos"
    allowed_regimes = [
        "high_volatility", "strong_trend_up", "strong_trend_down",
        "weak_trend", "ranging", "low_volatility",
    ]
    blocked_regimes: list[str] = []

    def __init__(self, params: dict | None = None):
        p = params or {}
        self.swing_lookback: int = p.get("swing_lookback", 5)  # 5-min bars each side for pivot
        self.pullback_buffer_ticks: int = p.get("pullback_buffer_ticks", 4)
        self.stop_buffer_ticks: int = p.get("stop_buffer_ticks", 4)
        self.default_rr: float = p.get("default_rr", 3.0)
        self.max_setup_age_bars: int = p.get("max_setup_age_bars", 48)  # 4 hours
        self.max_history_bars: int = p.get("max_history_bars", 150)
        self.min_swing_distance_ticks: int = p.get("min_swing_distance_ticks", 8)
        self.max_trades_per_day: int = p.get("max_trades_per_day", 2)
        self.max_stop_ticks: int = p.get("max_stop_ticks", 60)  # cap stop distance
        self.entry_mode: str = p.get("entry_mode", "pullback")  # "pullback" or "immediate"
        self.min_bars_from_open: int = p.get("min_bars_from_open", 0)  # skip first N bars of day
        # Timeframe aggregation: how many 5-min bars per candle for swing detection
        # 1=5min (default), 3=15min, 6=30min, 12=1hr
        self.agg_period: int = p.get("agg_period", 1)

        # State per instrument
        self._bars: dict[str, list[Bar]] = {}
        self._agg_bars: dict[str, list[dict]] = {}  # aggregated candles for swing detection
        self._bar_count: dict[str, int] = {}
        self._swings: dict[str, list[SwingPoint]] = {}
        self._setups: dict[str, list[BoSSetup]] = {}
        self._last_bos_idx: dict[str, int] = {}
        self._traded_today: dict[str, int] = {}
        self._current_date: dict[str, object] = {}
        self._day_bar_count: dict[str, int] = {}

    def reset(self) -> None:
        self._bars.clear()
        self._agg_bars.clear()
        self._bar_count.clear()
        self._swings.clear()
        self._setups.clear()
        self._last_bos_idx.clear()
        self._traded_today.clear()
        self._current_date.clear()
        self._day_bar_count.clear()

    def reset_daily(self) -> None:
        self._traded_today.clear()
        self._day_bar_count.clear()

    def _update_state(self, bar: Bar) -> None:
        """Track bars, optionally aggregate to higher TF, detect swing points."""
        inst = bar.instrument

        # Daily reset
        bar_date = bar.timestamp.date() if hasattr(bar.timestamp, "date") else None
        if bar_date and bar_date != self._current_date.get(inst):
            if self._current_date.get(inst) is not None:
                self.reset_daily()
            self._current_date[inst] = bar_date

        # Init
        if inst not in self._bars:
            self._bars[inst] = []
            self._agg_bars[inst] = []
            self._bar_count[inst] = 0
            self._swings[inst] = []
            self._setups[inst] = []

        self._bars[inst].append(bar)
        self._bar_count[inst] += 1
        self._day_bar_count[inst] = self._day_bar_count.get(inst, 0) + 1

        # Trim raw bar history
        if len(self._bars[inst]) > self.max_history_bars * 2:
            self._bars[inst] = self._bars[inst][-self.max_history_bars:]

        # Aggregate to higher TF candles if needed
        if self.agg_period > 1:
            if self._bar_count[inst] % self.agg_period == 0:
                recent = self._bars[inst][-self.agg_period:]
                agg = {
                    "high": max(b.high for b in recent),
                    "low": min(b.low for b in recent),
                    "bar_count": self._bar_count[inst],
                }
                self._agg_bars[inst].append(agg)
                if len(self._agg_bars[inst]) > 80:
                    self._agg_bars[inst] = self._agg_bars[inst][-60:]
                self._detect_swings_on_candles(inst)
        else:
            # Use raw 5-min bars directly for swing detection
            self._detect_swings_on_raw(inst, bar)

        # Expire old setups
        bc_now = self._bar_count[inst]
        self._setups[inst] = [
            s for s in self._setups.get(inst, [])
            if bc_now - s.created_bar < self.max_setup_age_bars and not s.triggered
        ]

    def _detect_swings_on_raw(self, inst: str, bar: Bar) -> None:
        """Detect swing points on raw 5-min bars."""
        bars = self._bars[inst]
        n = self.swing_lookback
        if len(bars) < 2 * n + 1:
            return

        center = len(bars) - n - 1
        center_bar = bars[center]

        is_high = True
        is_low = True
        for offset in range(1, n + 1):
            left = bars[center - offset]
            right = bars[center + offset]
            if left.high >= center_bar.high or right.high >= center_bar.high:
                is_high = False
            if left.low <= center_bar.low or right.low <= center_bar.low:
                is_low = False

        bc = self._bar_count[inst] - n
        self._maybe_add_swing(inst, center_bar.high, center_bar.low,
                              is_high, is_low, bc, center_bar.timestamp)

    def _detect_swings_on_candles(self, inst: str) -> None:
        """Detect swing points on aggregated higher-TF candles."""
        candles = self._agg_bars[inst]
        n = self.swing_lookback
        if len(candles) < 2 * n + 1:
            return

        center_idx = len(candles) - n - 1
        center = candles[center_idx]

        is_high = True
        is_low = True
        for offset in range(1, n + 1):
            left = candles[center_idx - offset]
            right = candles[center_idx + offset]
            if left["high"] >= center["high"] or right["high"] >= center["high"]:
                is_high = False
            if left["low"] <= center["low"] or right["low"] <= center["low"]:
                is_low = False

        bc = center["bar_count"]
        self._maybe_add_swing(inst, center["high"], center["low"],
                              is_high, is_low, bc, None)

    def _maybe_add_swing(self, inst: str, high: float, low: float,
                         is_high: bool, is_low: bool, bc: int,
                         timestamp: object) -> None:
        """Add swing point if it meets minimum distance criteria."""
        spec = INSTRUMENT_SPECS.get(inst, {})
        tick_size = spec.get("tick_size", 0.25)
        min_dist = self.min_swing_distance_ticks * tick_size

        if is_high:
            last_sh = None
            for sp in reversed(self._swings[inst]):
                if sp.kind == "high":
                    last_sh = sp
                    break
            if last_sh is None or abs(high - last_sh.price) >= min_dist:
                self._swings[inst].append(SwingPoint(
                    price=high, bar_index=bc, kind="high", timestamp=timestamp,
                ))

        if is_low:
            last_sl = None
            for sp in reversed(self._swings[inst]):
                if sp.kind == "low":
                    last_sl = sp
                    break
            if last_sl is None or abs(low - last_sl.price) >= min_dist:
                self._swings[inst].append(SwingPoint(
                    price=low, bar_index=bc, kind="low", timestamp=timestamp,
                ))

        # Keep swings bounded
        if len(self._swings[inst]) > 30:
            self._swings[inst] = self._swings[inst][-20:]

    def _detect_bos_and_setups(self, bar: Bar) -> Signal | None:
        """Check if current bar breaks structure. Create setups or return immediate signal."""
        inst = bar.instrument
        swings = self._swings.get(inst, [])
        if len(swings) < 3:
            return None

        spec = INSTRUMENT_SPECS.get(inst, {})
        tick_size = spec.get("tick_size", 0.25)
        bc = self._bar_count[inst]

        # Get recent swing highs and lows (sorted by time)
        recent_highs = [s for s in swings if s.kind == "high"]
        recent_lows = [s for s in swings if s.kind == "low"]

        # Bullish BoS: bar closes above the most recent swing high
        if recent_highs and recent_lows:
            last_sh = recent_highs[-1]

            # Only detect BoS once per swing level
            if bar.close > last_sh.price and self._last_bos_idx.get((inst, "long")) != last_sh.bar_index:
                self._last_bos_idx[(inst, "long")] = last_sh.bar_index

                # Stop = most recent swing low before the broken SH
                stop_swing = None
                for sp in reversed(recent_lows):
                    if sp.bar_index < last_sh.bar_index:
                        stop_swing = sp
                        break
                if stop_swing is None and recent_lows:
                    stop_swing = recent_lows[-1]

                if stop_swing:
                    stop = stop_swing.price - self.stop_buffer_ticks * tick_size

                    # Target = next swing high above the broken one, or use default R:R
                    target = None
                    # Look for higher swing highs in history
                    for sp in reversed(recent_highs[:-1]):
                        if sp.price > last_sh.price:
                            target = sp.price
                            break

                    # If no higher swing, use default R:R
                    risk = last_sh.price - stop
                    max_stop = self.max_stop_ticks * tick_size
                    if risk > 0 and risk <= max_stop:
                        if target is None:
                            target = last_sh.price + risk * self.default_rr

                        if self.entry_mode == "immediate":
                            if self._traded_today.get(inst, 0) < self.max_trades_per_day:
                                # Recalculate risk from actual entry (bar.close)
                                actual_risk = bar.close - stop
                                if actual_risk > 0:
                                    actual_tp = target if target else bar.close + actual_risk * self.default_rr
                                    reward = actual_tp - bar.close
                                    if reward / actual_risk >= 1.5:
                                        self._traded_today[inst] = self._traded_today.get(inst, 0) + 1
                                        return Signal(
                                            direction=SignalDirection.LONG,
                                            instrument=inst,
                                            entry_price=bar.close,
                                            stop_loss=stop,
                                            take_profit=actual_tp,
                                            confidence=1.0,
                                            strategy_name=self.name,
                                            metadata={"broken_level": last_sh.price,
                                                      "setup_type": "bullish_bos",
                                                      "risk": actual_risk, "reward": reward},
                                        )
                        else:
                            self._setups.setdefault(inst, []).append(BoSSetup(
                                direction="long",
                                broken_level=last_sh.price,
                                stop_level=stop,
                                target_level=target,
                                created_bar=bc,
                            ))

        # Bearish BoS: bar closes below the most recent swing low
        if recent_lows and recent_highs:
            last_sl = recent_lows[-1]

            if bar.close < last_sl.price and self._last_bos_idx.get((inst, "short")) != last_sl.bar_index:
                self._last_bos_idx[(inst, "short")] = last_sl.bar_index

                # Stop = most recent swing high before the broken SL
                stop_swing = None
                for sp in reversed(recent_highs):
                    if sp.bar_index < last_sl.bar_index:
                        stop_swing = sp
                        break
                if stop_swing is None and recent_highs:
                    stop_swing = recent_highs[-1]

                if stop_swing:
                    stop = stop_swing.price + self.stop_buffer_ticks * tick_size

                    # Target = next swing low below the broken one
                    target = None
                    for sp in reversed(recent_lows[:-1]):
                        if sp.price < last_sl.price:
                            target = sp.price
                            break

                    risk = stop - last_sl.price
                    max_stop = self.max_stop_ticks * tick_size
                    if risk > 0 and risk <= max_stop:
                        if target is None:
                            target = last_sl.price - risk * self.default_rr

                        if self.entry_mode == "immediate":
                            if self._traded_today.get(inst, 0) < self.max_trades_per_day:
                                actual_risk = stop - bar.close
                                if actual_risk > 0:
                                    actual_tp = target if target else bar.close - actual_risk * self.default_rr
                                    reward = bar.close - actual_tp
                                    if reward / actual_risk >= 1.5:
                                        self._traded_today[inst] = self._traded_today.get(inst, 0) + 1
                                        return Signal(
                                            direction=SignalDirection.SHORT,
                                            instrument=inst,
                                            entry_price=bar.close,
                                            stop_loss=stop,
                                            take_profit=actual_tp,
                                            confidence=1.0,
                                            strategy_name=self.name,
                                            metadata={"broken_level": last_sl.price,
                                                      "setup_type": "bearish_bos",
                                                      "risk": actual_risk, "reward": reward},
                                        )
                        else:
                            self._setups.setdefault(inst, []).append(BoSSetup(
                                direction="short",
                                broken_level=last_sl.price,
                                stop_level=stop,
                                target_level=target,
                                created_bar=bc,
                            ))

        return None

    def _check_pullback_entry(self, bar: Bar) -> Signal | None:
        """Check if price has pulled back to a broken level for entry."""
        inst = bar.instrument
        spec = INSTRUMENT_SPECS.get(inst, {})
        tick_size = spec.get("tick_size", 0.25)
        buffer = self.pullback_buffer_ticks * tick_size

        if self._traded_today.get(inst, 0) >= self.max_trades_per_day:
            return None

        for setup in self._setups.get(inst, []):
            if setup.triggered:
                continue

            if setup.direction == "long":
                # Price must dip to broken level (pullback) and close above it
                level = setup.broken_level
                if bar.low <= level + buffer and bar.close > level:
                    # Verify minimum pullback: price came from above
                    risk = bar.close - setup.stop_level
                    if risk <= 0:
                        continue

                    # Ensure R:R is at least 1.5
                    reward = setup.target_level - bar.close if setup.target_level else risk * self.default_rr
                    if reward / risk < 1.5:
                        continue

                    tp = setup.target_level if setup.target_level else bar.close + risk * self.default_rr

                    setup.triggered = True
                    self._traded_today[inst] = self._traded_today.get(inst, 0) + 1

                    return Signal(
                        direction=SignalDirection.LONG,
                        instrument=inst,
                        entry_price=bar.close,
                        stop_loss=setup.stop_level,
                        take_profit=tp,
                        confidence=1.0,
                        strategy_name=self.name,
                        metadata={
                            "broken_level": level,
                            "setup_type": "bullish_bos",
                            "risk": risk,
                            "reward": reward,
                        },
                    )

            elif setup.direction == "short":
                level = setup.broken_level
                if bar.high >= level - buffer and bar.close < level:
                    risk = setup.stop_level - bar.close
                    if risk <= 0:
                        continue

                    reward = bar.close - setup.target_level if setup.target_level else risk * self.default_rr
                    if reward / risk < 1.5:
                        continue

                    tp = setup.target_level if setup.target_level else bar.close - risk * self.default_rr

                    setup.triggered = True
                    self._traded_today[inst] = self._traded_today.get(inst, 0) + 1

                    return Signal(
                        direction=SignalDirection.SHORT,
                        instrument=inst,
                        entry_price=bar.close,
                        stop_loss=setup.stop_level,
                        take_profit=tp,
                        confidence=1.0,
                        strategy_name=self.name,
                        metadata={
                            "broken_level": level,
                            "setup_type": "bearish_bos",
                            "risk": risk,
                            "reward": reward,
                        },
                    )

        return None

    def on_bar(self, bar: Bar, indicators: pd.Series, regime: MarketRegime) -> Signal | None:
        self._update_state(bar)

        if not self.is_regime_allowed(regime):
            return None

        # Always detect BoS setups regardless of time filter
        sig = self._detect_bos_and_setups(bar)

        # Skip signal generation in early session (let breakout strategy operate)
        inst = bar.instrument
        if self.min_bars_from_open > 0:
            day_count = self._day_bar_count.get(inst, 0)
            if day_count <= self.min_bars_from_open:
                return None

        if sig:
            return sig

        # Pullback mode checks pending setups
        return self._check_pullback_entry(bar)

    def track_bar(self, bar: Bar, indicators: pd.Series, regime: MarketRegime) -> None:
        self._update_state(bar)
        self._detect_bos_and_setups(bar)

    def should_exit(self, position: Position, bar: Bar, indicators: pd.Series) -> Signal | None:
        if position.stop_loss is not None:
            if position.direction == SignalDirection.LONG and bar.low <= position.stop_loss:
                return Signal(
                    direction=SignalDirection.SHORT,
                    instrument=bar.instrument,
                    entry_price=position.stop_loss,
                    stop_loss=0, take_profit=0, confidence=1.0,
                    strategy_name=self.name,
                    metadata={"exit_reason": "stop_loss"},
                )
            if position.direction == SignalDirection.SHORT and bar.high >= position.stop_loss:
                return Signal(
                    direction=SignalDirection.LONG,
                    instrument=bar.instrument,
                    entry_price=position.stop_loss,
                    stop_loss=0, take_profit=0, confidence=1.0,
                    strategy_name=self.name,
                    metadata={"exit_reason": "stop_loss"},
                )

        if position.take_profit is not None:
            if position.direction == SignalDirection.LONG and bar.high >= position.take_profit:
                return Signal(
                    direction=SignalDirection.SHORT,
                    instrument=bar.instrument,
                    entry_price=position.take_profit,
                    stop_loss=0, take_profit=0, confidence=1.0,
                    strategy_name=self.name,
                    metadata={"exit_reason": "take_profit"},
                )
            if position.direction == SignalDirection.SHORT and bar.low <= position.take_profit:
                return Signal(
                    direction=SignalDirection.LONG,
                    instrument=bar.instrument,
                    entry_price=position.take_profit,
                    stop_loss=0, take_profit=0, confidence=1.0,
                    strategy_name=self.name,
                    metadata={"exit_reason": "take_profit"},
                )

        return None
