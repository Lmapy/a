"""Drawdown engines.

A DrawdownState tracks one prop account through a sequence of trades
grouped by trading day. The engine fires `breach()` returning the first
breach reason (one of "max_loss", "daily_loss", "trailing_dd") or None.

Three modes:
  static                 fail if balance <= start - max_loss
  eod_trailing           after each completed trading day, peak_eod
                         shifts up; fail if eod_balance <= peak_eod -
                         trailing_drawdown
  intraday_trailing      same but peak updates on every trade tick
"""
from __future__ import annotations

from dataclasses import dataclass, field

from prop_challenge.accounts import AccountSpec


@dataclass
class DrawdownState:
    spec: AccountSpec
    balance: float
    starting_balance: float
    peak_eod: float
    peak_intraday: float
    day_starting_balance: float
    daily_pnl: float = 0.0
    last_breach: str | None = None
    breach_day: int | None = None
    locked_today: bool = False
    daily_pnls: list[float] = field(default_factory=list)
    days_traded: int = 0

    @classmethod
    def fresh(cls, spec: AccountSpec) -> "DrawdownState":
        return cls(
            spec=spec,
            balance=spec.starting_balance,
            starting_balance=spec.starting_balance,
            peak_eod=spec.starting_balance,
            peak_intraday=spec.starting_balance,
            day_starting_balance=spec.starting_balance,
        )

    # ---------- mutators ----------

    def apply_trade(self, dollar_pnl: float, day_index: int) -> None:
        if self.last_breach is not None:
            return
        self.balance += dollar_pnl
        self.daily_pnl += dollar_pnl
        self.peak_intraday = max(self.peak_intraday, self.balance)
        self._check_breach(day_index)

    def end_of_day(self, day_index: int) -> None:
        if self.last_breach is not None:
            return
        if abs(self.daily_pnl) > 0:
            self.daily_pnls.append(self.daily_pnl)
            self.days_traded += 1
        # EOD trailing: peak shifts after the day completes (capped at start + buffer? No -- standard prop EOD trailing caps once breakeven hit on funded; for a challenge it just rolls up).
        self.peak_eod = max(self.peak_eod, self.balance)
        self._check_breach(day_index)
        # reset day-scoped trackers
        self.daily_pnl = 0.0
        self.day_starting_balance = self.balance
        self.locked_today = False
        # intraday peak resets to current balance for fairness next day
        self.peak_intraday = self.balance

    # ---------- breach checks ----------

    def _check_breach(self, day_index: int) -> None:
        s = self.spec
        # 1. Daily loss limit -- per-day floor measured from day start.
        # Some firms (MFFU 2026 onwards) do not have a DLL; spec sets
        # daily_loss_limit=None and the check is skipped.
        if s.daily_loss_limit is not None and self.daily_pnl <= -s.daily_loss_limit:
            self.last_breach = "daily_loss"
            self.breach_day = day_index
            return
        # 2. Max loss / drawdown
        if s.drawdown_type == "static":
            if self.balance <= self.starting_balance - s.max_loss:
                self.last_breach = "max_loss_static"
                self.breach_day = day_index
                return
        elif s.drawdown_type == "eod_trailing":
            if self.balance <= self.peak_eod - s.trailing_drawdown:
                self.last_breach = "trailing_dd_eod"
                self.breach_day = day_index
                return
        elif s.drawdown_type == "intraday_trailing":
            if self.balance <= self.peak_intraday - s.trailing_drawdown:
                self.last_breach = "trailing_dd_intraday"
                self.breach_day = day_index
                return
        else:
            raise ValueError(f"unknown drawdown_type: {s.drawdown_type}")

    # ---------- queries ----------

    def alive(self) -> bool:
        return self.last_breach is None

    def reached_target(self) -> bool:
        return self.balance >= self.starting_balance + self.spec.profit_target

    def consistency_breached(self) -> bool:
        """Single day's PnL must not exceed `consistency_rule_percent` % of total profit."""
        if not self.daily_pnls:
            return False
        total_profit = sum(p for p in self.daily_pnls if p > 0)
        if total_profit <= 0:
            return False
        biggest = max(self.daily_pnls)
        if biggest <= 0:
            return False
        return (biggest / total_profit) * 100.0 > self.spec.consistency_rule_percent
