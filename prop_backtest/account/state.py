"""Account state — tracks all mutable financial state bar-by-bar."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from prop_backtest.contracts.specs import ContractSpec
from prop_backtest.firms.base import AccountTier, FirmRules


@dataclass
class AccountState:
    """All mutable state for a simulated prop firm account.

    This object is mutated in-place by RuleMonitor and SimulatedBroker.
    Strategies receive a snapshot (or the live object) for read-only use.
    """

    # ── configuration ──────────────────────────────────────────────────────
    tier: AccountTier
    firm_rules: FirmRules
    contract: ContractSpec

    # ── balance tracking ───────────────────────────────────────────────────
    realized_balance: float = field(init=False)
    # Cash balance: starting_balance + sum of all closed trade net PnL.

    open_pnl: float = 0.0
    # Mark-to-market on the currently open position.

    # ── drawdown tracking ──────────────────────────────────────────────────
    intraday_hwm: float = field(init=False)
    # Highest equity (realized + open) seen since account opened.
    # Used for TRAILING_INTRADAY drawdown firms.

    eod_hwm: float = field(init=False)
    # Highest realized balance at end of any trading day.
    # Used for TRAILING_EOD drawdown firms.

    drawdown_floor: float = field(init=False)
    # Current absolute dollar floor. Equity must stay above this value.

    # ── daily tracking ─────────────────────────────────────────────────────
    day_start_balance: float = field(init=False)
    # realized_balance at the open of the current calendar day.

    trading_days_active: set[date] = field(default_factory=set)
    # Set of calendar dates on which at least one fill occurred.

    current_date: Optional[date] = None
    # The date of the most recently processed bar.

    # ── open position ──────────────────────────────────────────────────────
    position_contracts: int = 0
    # Signed: positive = long, negative = short, 0 = flat.

    avg_entry_price: float = 0.0
    # Volume-weighted average entry price of the open position.

    # ── trade excursion tracking (for open position) ───────────────────────
    position_min_price: float = field(init=False)   # lowest price seen during trade
    position_max_price: float = field(init=False)   # highest price seen during trade

    # ── violation flags ────────────────────────────────────────────────────
    breached_trailing_dd: bool = False
    breached_daily_loss: bool = False
    hit_profit_target: bool = False
    is_terminated: bool = False

    def __post_init__(self) -> None:
        self.realized_balance = self.tier.starting_balance
        self.intraday_hwm = self.tier.starting_balance
        self.eod_hwm = self.tier.starting_balance
        self.day_start_balance = self.tier.starting_balance
        self.drawdown_floor = self.tier.starting_balance - self.tier.max_trailing_drawdown
        self.position_min_price = 0.0
        self.position_max_price = 0.0

    @property
    def starting_balance(self) -> float:
        return self.tier.starting_balance

    @property
    def equity(self) -> float:
        """Total account equity: realized balance + open position mark-to-market."""
        return self.realized_balance + self.open_pnl

    @property
    def current_day_pnl(self) -> float:
        """Net realized PnL for the current calendar day (does not include open PnL)."""
        return self.realized_balance - self.day_start_balance

    @property
    def total_net_pnl(self) -> float:
        """Total net PnL including open position."""
        return self.equity - self.tier.starting_balance

    @property
    def dd_floor_proximity(self) -> float:
        """Normalised distance from the drawdown floor [0..∞].
        0 means equity equals the floor (imminent breach).
        1 means equity is exactly max_trailing_drawdown above floor.
        """
        if self.tier.max_trailing_drawdown == 0:
            return 1.0
        return (self.equity - self.drawdown_floor) / self.tier.max_trailing_drawdown
