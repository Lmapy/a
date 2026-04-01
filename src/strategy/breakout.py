from __future__ import annotations

import pandas as pd

from src.data.models import Bar, MarketRegime, Position, Signal, SignalDirection
from src.strategy.base import BaseStrategy


class BreakoutStrategy(BaseStrategy):
    """Trade opening range breakouts.

    Entry: price breaks above/below the opening range (first N minutes)
    Stop: half of OR range from entry (tighter than full OR).
    Take profit: 2x risk (measured from entry).
    """

    name = "breakout"
    allowed_regimes = ["high_volatility", "strong_trend_up", "strong_trend_down", "weak_trend", "ranging", "low_volatility"]
    blocked_regimes = []

    def __init__(self, params: dict | None = None):
        p = params or {}
        self.opening_range_minutes = p.get("opening_range_minutes", 60)
        self.stop_range_pct = p.get("stop_range_pct", 0.5)  # stop at 50% of OR range
        self.take_profit_rr = p.get("take_profit_rr", 2.0)  # R:R target
        self.min_range_points = p.get("min_range_points", 0.5)
        self.max_range_points = p.get("max_range_points", 20.0)
        self.max_bars_after_or = p.get("max_bars_after_or", 18)  # 90 min window
        self.min_tick_profit = p.get("min_tick_profit", 0)  # min ticks for TP (MFF: 4)

        # Per-instrument daily state
        self._or_high: dict[str, float] = {}
        self._or_low: dict[str, float] = {}
        self._or_established: dict[str, bool] = {}
        self._breakout_traded: dict[str, bool] = {}
        self._bar_count: dict[str, int] = {}
        self._bars_after_or: dict[str, int] = {}

    def reset_daily(self) -> None:
        """Reset opening range state at start of each day."""
        self._or_high.clear()
        self._or_low.clear()
        self._or_established.clear()
        self._breakout_traded.clear()
        self._bar_count.clear()
        self._bars_after_or.clear()

    def on_bar(self, bar: Bar, indicators: pd.Series, regime: MarketRegime) -> Signal | None:
        if not self.is_regime_allowed(regime):
            return None

        inst = bar.instrument

        # Track bar count for opening range calculation
        self._bar_count[inst] = self._bar_count.get(inst, 0) + 1

        # Build opening range
        if not self._or_established.get(inst, False):
            self._or_high[inst] = max(self._or_high.get(inst, bar.high), bar.high)
            self._or_low[inst] = min(self._or_low.get(inst, bar.low), bar.low)

            # Check if opening range period is complete (assuming 5-min bars)
            bars_needed = self.opening_range_minutes // 5
            if self._bar_count[inst] >= bars_needed:
                self._or_established[inst] = True
                self._bars_after_or[inst] = 0

            return None

        # Already traded a breakout today
        if self._breakout_traded.get(inst, False):
            return None

        # Only look for breakouts within time window after OR
        self._bars_after_or[inst] = self._bars_after_or.get(inst, 0) + 1
        if self._bars_after_or[inst] > self.max_bars_after_or:
            return None

        or_high = self._or_high[inst]
        or_low = self._or_low[inst]
        or_range = or_high - or_low

        # Range size filter
        if or_range < self.min_range_points or or_range > self.max_range_points:
            return None

        # Breakout above opening range high
        if bar.close > or_high:
            stop = bar.close - or_range * self.stop_range_pct
            risk = bar.close - stop
            tp = bar.close + risk * self.take_profit_rr

            # Enforce minimum tick profit (e.g., MFF 4-tick rule)
            if self.min_tick_profit > 0:
                from src.data.models import INSTRUMENT_SPECS
                spec = INSTRUMENT_SPECS.get(inst, {})
                tick_size = spec.get("tick_size", 0.25)
                min_tp = bar.close + self.min_tick_profit * tick_size
                tp = max(tp, min_tp)

            self._breakout_traded[inst] = True
            return Signal(
                direction=SignalDirection.LONG,
                instrument=inst,
                entry_price=bar.close,
                stop_loss=stop,
                take_profit=tp,
                confidence=1.0,
                strategy_name=self.name,
                metadata={"or_high": or_high, "or_low": or_low, "or_range": or_range},
            )

        # Breakout below opening range low
        if bar.close < or_low:
            stop = bar.close + or_range * self.stop_range_pct
            risk = stop - bar.close
            tp = bar.close - risk * self.take_profit_rr

            # Enforce minimum tick profit (e.g., MFF 4-tick rule)
            if self.min_tick_profit > 0:
                from src.data.models import INSTRUMENT_SPECS
                spec = INSTRUMENT_SPECS.get(inst, {})
                tick_size = spec.get("tick_size", 0.25)
                min_tp = bar.close - self.min_tick_profit * tick_size
                tp = min(tp, min_tp)

            self._breakout_traded[inst] = True
            return Signal(
                direction=SignalDirection.SHORT,
                instrument=inst,
                entry_price=bar.close,
                stop_loss=stop,
                take_profit=tp,
                confidence=1.0,
                strategy_name=self.name,
                metadata={"or_high": or_high, "or_low": or_low, "or_range": or_range},
            )

        return None

    def should_exit(self, position: Position, bar: Bar, indicators: pd.Series) -> Signal | None:
        # Check stop loss
        if position.stop_loss is not None:
            if position.direction == SignalDirection.LONG and bar.low <= position.stop_loss:
                return self._exit_signal(position, position.stop_loss, "stop_loss")
            if position.direction == SignalDirection.SHORT and bar.high >= position.stop_loss:
                return self._exit_signal(position, position.stop_loss, "stop_loss")

        # Check take profit
        if position.take_profit is not None:
            if position.direction == SignalDirection.LONG and bar.high >= position.take_profit:
                return self._exit_signal(position, position.take_profit, "take_profit")
            if position.direction == SignalDirection.SHORT and bar.low <= position.take_profit:
                return self._exit_signal(position, position.take_profit, "take_profit")

        return None

    @staticmethod
    def _exit_signal(position: Position, price: float, reason: str) -> Signal:
        exit_dir = SignalDirection.SHORT if position.direction == SignalDirection.LONG else SignalDirection.LONG
        return Signal(
            direction=exit_dir,
            instrument=position.instrument,
            entry_price=price,
            stop_loss=0,
            take_profit=0,
            confidence=1.0,
            strategy_name="breakout",
            metadata={"exit_reason": reason},
        )
