"""Strategy interface — ABC and convenience wrappers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

from prop_backtest.account.state import AccountState
from prop_backtest.contracts.specs import ContractSpec
from prop_backtest.data.loader import BarData
from prop_backtest.engine.broker import Fill, Signal
from prop_backtest.firms.base import AccountTier


class Strategy(ABC):
    """Abstract base class for all trading strategies.

    Subclass this and implement ``on_bar`` to define your trading logic.

    The strategy interacts with the engine through four lifecycle hooks:

    on_start(contract, tier)
        Called once before the first bar. Use for one-time setup.

    on_bar(bar, history, account) -> Signal
        Called for every bar (after any pending fill has been processed).
        Must return a Signal. Return Signal("hold") to do nothing.

    on_fill(fill)
        Called whenever an order is filled. Optional; useful for logging.

    on_end()
        Called after the last bar. Optional; useful for cleanup.

    AccountState fields available in on_bar (read-only):
        position_contracts  — signed int (+long, -short, 0=flat)
        equity              — float, total account equity
        realized_balance    — float
        open_pnl            — float
        drawdown_floor      — float, current floor level
        dd_floor_proximity  — float, normalised safety margin [0..∞]
        current_day_pnl     — float, realized PnL today
        intraday_hwm        — float
    """

    def on_start(self, contract: ContractSpec, tier: AccountTier) -> None:
        """Called once before the first bar. Override for setup."""

    @abstractmethod
    def on_bar(
        self,
        bar: BarData,
        history: list[BarData],
        account: AccountState,
    ) -> Signal:
        """Called on every bar. Must return a Signal."""

    def on_fill(self, fill: Fill) -> None:
        """Called when an order is filled. Override for logging/state updates."""

    def on_end(self) -> None:
        """Called after the last bar. Override for cleanup."""


class FunctionStrategy(Strategy):
    """Wraps a plain callable as a Strategy.

    The callable must match the signature:
        fn(bar: BarData, history: list[BarData], account: AccountState) -> Signal
    """

    def __init__(
        self,
        fn: Callable[[BarData, list[BarData], AccountState], Signal],
        name: str = "FunctionStrategy",
    ) -> None:
        self._fn = fn
        self.__class__.__name__ = name

    def on_bar(
        self,
        bar: BarData,
        history: list[BarData],
        account: AccountState,
    ) -> Signal:
        return self._fn(bar, history, account)
