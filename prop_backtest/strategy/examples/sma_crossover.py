"""Simple SMA crossover strategy — reference implementation.

Goes long when the fast SMA crosses above the slow SMA,
goes short when it crosses below. Closes on the opposite cross.
Uses a volatility-based position sizing: stays at 1 contract by default.

This is a demonstration strategy, not a recommendation.
"""
from __future__ import annotations

from prop_backtest.account.state import AccountState
from prop_backtest.contracts.specs import ContractSpec
from prop_backtest.data.loader import BarData
from prop_backtest.engine.broker import Signal
from prop_backtest.firms.base import AccountTier
from prop_backtest.strategy.base import Strategy


class SMACrossover(Strategy):
    """SMA crossover strategy.

    Parameters:
        fast: Period for the fast moving average (default 9).
        slow: Period for the slow moving average (default 21).
        contracts: Number of contracts per trade (default 1).
        risk_aware: If True, reduce position size when near the drawdown floor.
    """

    def __init__(
        self,
        fast: int = 9,
        slow: int = 21,
        contracts: int = 1,
        risk_aware: bool = True,
    ) -> None:
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be less than slow ({slow})")
        self.fast = fast
        self.slow = slow
        self.contracts = contracts
        self.risk_aware = risk_aware

    def on_start(self, contract: ContractSpec, tier: AccountTier) -> None:
        self._contract = contract
        self._tier = tier

    def on_bar(
        self,
        bar: BarData,
        history: list[BarData],
        account: AccountState,
    ) -> Signal:
        if len(history) < self.slow:
            return Signal(action="hold")

        closes = [b.close for b in history[-self.slow :]]
        sma_fast = sum(closes[-self.fast :]) / self.fast
        sma_slow = sum(closes) / self.slow

        pos = account.position_contracts
        contracts = self._get_contracts(account)

        # ── Entry logic ────────────────────────────────────────────────
        if pos == 0:
            if sma_fast > sma_slow:
                return Signal(action="buy", contracts=contracts)
            elif sma_fast < sma_slow:
                return Signal(action="sell", contracts=contracts)

        # ── Exit / reversal logic ──────────────────────────────────────
        elif pos > 0 and sma_fast < sma_slow:
            return Signal(action="close")

        elif pos < 0 and sma_fast > sma_slow:
            return Signal(action="close")

        return Signal(action="hold")

    def _get_contracts(self, account: AccountState) -> int:
        """Reduce size when approaching the drawdown floor."""
        if not self.risk_aware:
            return self.contracts
        # Scale down linearly below 0.5× the safety buffer
        proximity = account.dd_floor_proximity
        if proximity >= 0.5:
            return self.contracts
        elif proximity >= 0.25:
            return max(1, self.contracts // 2)
        else:
            return 1
