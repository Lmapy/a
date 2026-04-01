from __future__ import annotations

from src.data.models import MarketRegime
from src.strategy.base import BaseStrategy


# Default regime -> strategy allocation mapping
DEFAULT_ALLOCATIONS: dict[MarketRegime, dict[str, float]] = {
    MarketRegime.STRONG_TREND_UP:   {"trend_following": 0.70, "breakout": 0.30},
    MarketRegime.STRONG_TREND_DOWN: {"trend_following": 0.70, "breakout": 0.30},
    MarketRegime.WEAK_TREND:        {"trend_following": 0.50, "mean_reversion": 0.50},
    MarketRegime.RANGING:           {"mean_reversion": 0.80, "breakout": 0.20},
    MarketRegime.HIGH_VOLATILITY:   {"breakout": 0.60, "mean_reversion": 0.40},
    MarketRegime.LOW_VOLATILITY:    {"mean_reversion": 1.00},
}


class StrategySelector:
    """Maps the current market regime to active strategies with risk allocations.

    The allocation percentages represent the fraction of the total risk budget
    each strategy can use — not position size directly.
    """

    def __init__(self, strategies: dict[str, BaseStrategy],
                 allocations: dict[MarketRegime, dict[str, float]] | None = None,
                 cooldown_bars: int = 5):
        self.strategies = strategies
        self.allocations = allocations or DEFAULT_ALLOCATIONS
        self.cooldown_bars = cooldown_bars

        self._current_regime: MarketRegime | None = None
        self._bars_since_switch = 0

    def select(self, regime: MarketRegime) -> dict[str, tuple[BaseStrategy, float]]:
        """Returns dict of strategy_name -> (strategy_instance, allocation_pct)."""
        # Track regime changes with cooldown
        if regime != self._current_regime:
            self._bars_since_switch = 0
            self._current_regime = regime
        else:
            self._bars_since_switch += 1

        alloc = self.allocations.get(regime, {"mean_reversion": 1.0})

        result = {}
        for strat_name, pct in alloc.items():
            if strat_name in self.strategies:
                strat = self.strategies[strat_name]
                if strat.is_regime_allowed(regime):
                    result[strat_name] = (strat, pct)

        # If no strategies available, fall back to mean reversion at minimal allocation
        if not result and "mean_reversion" in self.strategies:
            result["mean_reversion"] = (self.strategies["mean_reversion"], 0.5)

        return result

    @property
    def in_cooldown(self) -> bool:
        return self._bars_since_switch < self.cooldown_bars
