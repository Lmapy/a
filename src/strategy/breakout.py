from __future__ import annotations

import pandas as pd

from src.data.models import Bar, MarketRegime, Position, Signal, SignalDirection
from src.strategy.base import BaseStrategy


class BreakoutStrategy(BaseStrategy):
    """Trade opening range breakouts and key level breaks.

    Entry: price breaks above/below the opening range (first N minutes)
           with volume confirmation (volume > threshold * avg volume).
    Stop: opposite side of the range (clamped to stop_loss_atr * ATR).
    Take profit: measured move (range height projected from breakout).
    Filter: volatile/trending regime, ATR above median.
    """

    name = "breakout"
    allowed_regimes = ["high_volatility", "strong_trend_up", "strong_trend_down"]
    blocked_regimes = ["low_volatility"]

    def __init__(self, params: dict | None = None):
        p = params or {}
        self.opening_range_minutes = p.get("opening_range_minutes", 30)
        self.volume_spike_threshold = p.get("volume_spike_threshold", 1.5)
        self.stop_loss_atr = p.get("stop_loss_atr", 1.5)
        self.take_profit_multiplier = p.get("take_profit_multiplier", 1.5)
        self.min_range_atr = p.get("min_range_atr", 0.5)
        self.max_range_atr = p.get("max_range_atr", 3.0)

        # Per-instrument daily state
        self._or_high: dict[str, float] = {}
        self._or_low: dict[str, float] = {}
        self._or_established: dict[str, bool] = {}
        self._breakout_traded: dict[str, bool] = {}
        self._bar_count: dict[str, int] = {}

    def reset_daily(self) -> None:
        """Reset opening range state at start of each day."""
        self._or_high.clear()
        self._or_low.clear()
        self._or_established.clear()
        self._breakout_traded.clear()
        self._bar_count.clear()

    def on_bar(self, bar: Bar, indicators: pd.Series, regime: MarketRegime) -> Signal | None:
        if not self.is_regime_allowed(regime):
            return None

        inst = bar.instrument
        atr_val = indicators.get("atr_14")
        volume_sma = indicators.get("volume_sma")

        if any(v is None or (isinstance(v, float) and pd.isna(v))
               for v in [atr_val, volume_sma]):
            return None

        if atr_val <= 0:
            return None

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

            return None

        # Already traded a breakout today
        if self._breakout_traded.get(inst, False):
            return None

        or_high = self._or_high[inst]
        or_low = self._or_low[inst]
        or_range = or_high - or_low

        # Range size filter
        if or_range < self.min_range_atr * atr_val:
            return None  # range too narrow, likely a squeeze — wait
        if or_range > self.max_range_atr * atr_val:
            return None  # range too wide, stop would be too large

        # Volume confirmation
        has_volume = volume_sma > 0 and bar.volume > volume_sma * self.volume_spike_threshold

        # Breakout above opening range high
        if bar.close > or_high and has_volume:
            stop = max(or_low, bar.close - self.stop_loss_atr * atr_val)
            tp = bar.close + or_range * self.take_profit_multiplier
            confidence = min(bar.volume / (volume_sma * 2) if volume_sma > 0 else 0.5, 1.0)
            self._breakout_traded[inst] = True
            return Signal(
                direction=SignalDirection.LONG,
                instrument=inst,
                entry_price=bar.close,
                stop_loss=stop,
                take_profit=tp,
                confidence=confidence,
                strategy_name=self.name,
                metadata={"or_high": or_high, "or_low": or_low, "or_range": or_range},
            )

        # Breakout below opening range low
        if bar.close < or_low and has_volume:
            stop = min(or_high, bar.close + self.stop_loss_atr * atr_val)
            tp = bar.close - or_range * self.take_profit_multiplier
            confidence = min(bar.volume / (volume_sma * 2) if volume_sma > 0 else 0.5, 1.0)
            self._breakout_traded[inst] = True
            return Signal(
                direction=SignalDirection.SHORT,
                instrument=inst,
                entry_price=bar.close,
                stop_loss=stop,
                take_profit=tp,
                confidence=confidence,
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
