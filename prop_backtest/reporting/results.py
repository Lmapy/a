"""Result dataclasses for backtest output."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal, Optional


@dataclass
class TradeRecord:
    """A completed round-turn trade."""
    trade_id: int
    entry_time: datetime
    exit_time: datetime
    direction: Literal["long", "short"]
    contracts: int
    entry_price: float
    exit_price: float
    gross_pnl: float       # before commission
    commission: float      # total commission (both legs)
    net_pnl: float         # gross_pnl - commission
    mae: float             # maximum adverse excursion in dollars
    mfe: float             # maximum favourable excursion in dollars

    @property
    def duration_seconds(self) -> float:
        return (self.exit_time - self.entry_time).total_seconds()


@dataclass
class RuleCheckResult:
    """Pass/fail result for a single prop firm rule."""
    rule_name: str         # e.g. "profit_target", "trailing_drawdown"
    passed: bool
    value: float           # actual value achieved
    threshold: float       # required threshold
    detail: str            # human-readable explanation


@dataclass
class BacktestResult:
    """Full result of a completed backtest."""

    # ── Metadata ──────────────────────────────────────────────────────────
    firm_name: str
    tier_name: str
    contract_symbol: str
    start_date: date
    end_date: date

    # ── Financial summary ─────────────────────────────────────────────────
    starting_balance: float
    final_realized_balance: float
    final_equity: float
    peak_equity: float
    max_drawdown_dollars: float   # largest equity drawdown in $

    # ── Trade log ─────────────────────────────────────────────────────────
    trades: list[TradeRecord] = field(default_factory=list)

    # ── Equity curve [(timestamp, equity), ...] ────────────────────────────
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)

    # ── Prop firm verdict ─────────────────────────────────────────────────
    rule_checks: list[RuleCheckResult] = field(default_factory=list)
    passed: bool = False
    failure_reason: Optional[str] = None

    # ── Statistics (populated by compute_statistics) ──────────────────────
    stats: dict[str, float] = field(default_factory=dict)
