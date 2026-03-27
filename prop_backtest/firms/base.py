"""Base dataclasses for prop firm challenge rules."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DrawdownType(Enum):
    """How the trailing drawdown high-water mark is updated."""
    TRAILING_INTRADAY = "trailing_intraday"
    # HWM tracks the highest equity seen on any bar during the session.
    # Most common (TopStep Express, Lucid).

    TRAILING_EOD = "trailing_eod"
    # HWM only updates at end of each trading day based on the realized
    # (closed) balance. Open P&L is NOT counted toward the HWM.
    # Used by some MyFundedFutures account types.

    STATIC = "static"
    # Floor is fixed at starting_balance - max_trailing_drawdown.
    # Does not trail upward as account grows.


@dataclass(frozen=True)
class AccountTier:
    """Parameters for a single account size tier within a firm's challenge."""
    name: str                         # e.g. "50K", "100K"
    starting_balance: float           # initial account balance in USD
    profit_target: float              # USD above starting_balance required to pass
    max_trailing_drawdown: float      # maximum USD drawdown from HWM before account is breached
    daily_loss_limit: float           # maximum USD loss in a single calendar day
    min_trading_days: int             # minimum number of days on which at least one trade occurs
    max_contracts: Optional[int]      # position size cap (None = uncapped)
    consistency_rule: bool = False    # if True, no single day can account for >X% of total profit
    consistency_pct: float = 0.40     # max fraction of total profit allowed in one day


@dataclass(frozen=True)
class FirmRules:
    """All rules for a specific prop firm."""
    firm_name: str
    drawdown_type: DrawdownType
    tiers: list[AccountTier] = field(default_factory=list)

    def get_tier(self, name: str) -> AccountTier:
        """Retrieve a tier by name (case-insensitive)."""
        key = name.upper()
        for tier in self.tiers:
            if tier.name.upper() == key:
                return tier
        available = ", ".join(t.name for t in self.tiers)
        raise ValueError(
            f"Tier '{name}' not found for {self.firm_name}. Available: {available}"
        )
