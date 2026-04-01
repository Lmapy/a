from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from src.data.models import Bar, INSTRUMENT_SPECS, MarketRegime, Position, Signal, SignalDirection


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    name: str = ""
    allowed_regimes: list[str] = []
    blocked_regimes: list[str] = []

    @abstractmethod
    def on_bar(self, bar: Bar, indicators: pd.Series, regime: MarketRegime) -> Signal | None:
        """Evaluate current bar and return a signal or None."""

    def track_bar(self, bar: Bar, indicators: pd.Series, regime: MarketRegime) -> None:
        """Update internal state without generating signals.

        Called for every bar so strategies can maintain bar history,
        zones, and levels even when another strategy fires first.
        Default implementation calls on_bar and discards the result.
        Override for better performance.
        """
        self.on_bar(bar, indicators, regime)

    @abstractmethod
    def should_exit(self, position: Position, bar: Bar, indicators: pd.Series) -> Signal | None:
        """Check if an open position should be exited."""

    def reset(self) -> None:
        """Reset all strategy state for a new backtest run."""
        pass

    def is_regime_allowed(self, regime: MarketRegime) -> bool:
        if self.blocked_regimes and regime.value in self.blocked_regimes:
            return False
        if self.allowed_regimes and regime.value not in self.allowed_regimes:
            return False
        return True

    @staticmethod
    def _move_stop_to_breakeven(position: Position, bar: Bar, trigger_r: float = 2.0) -> None:
        """Move stop to breakeven + 1 tick once price reaches trigger_r * risk.

        Args:
            trigger_r: Multiple of risk at which to move stop to breakeven.
                       Default 2.0 means the stop moves when price reaches 2R profit.
        """
        if position.stop_loss is None:
            return

        spec = INSTRUMENT_SPECS.get(position.instrument, INSTRUMENT_SPECS["ES"])
        tick_size = spec.get("tick_size", 0.25)
        entry = position.entry_price
        risk = abs(entry - position.stop_loss)

        if risk <= 0:
            return

        if position.direction == SignalDirection.LONG:
            target = entry + risk * trigger_r
            if bar.high >= target and position.stop_loss < entry + tick_size:
                position.stop_loss = entry + tick_size
        else:
            target = entry - risk * trigger_r
            if bar.low <= target and position.stop_loss > entry - tick_size:
                position.stop_loss = entry - tick_size
