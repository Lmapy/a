from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from src.data.models import Bar, MarketRegime, Position, Signal


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    name: str = ""
    allowed_regimes: list[str] = []
    blocked_regimes: list[str] = []

    @abstractmethod
    def on_bar(self, bar: Bar, indicators: pd.Series, regime: MarketRegime) -> Signal | None:
        """Evaluate current bar and return a signal or None."""

    @abstractmethod
    def should_exit(self, position: Position, bar: Bar, indicators: pd.Series) -> Signal | None:
        """Check if an open position should be exited."""

    def is_regime_allowed(self, regime: MarketRegime) -> bool:
        if self.blocked_regimes and regime.value in self.blocked_regimes:
            return False
        if self.allowed_regimes and regime.value not in self.allowed_regimes:
            return False
        return True
