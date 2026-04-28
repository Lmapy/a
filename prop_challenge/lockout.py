"""Daily-rule (lockout) engine.

Each trading day the rules decide whether the next trade may be taken.
The engine returns True to STOP trading for the day after a trigger
event fires.

Triggers:
  max_trades_per_day        cap on trade count
  stop_after_n_wins         lock after this many winning trades
  stop_after_n_losses       lock after this many losing trades
  daily_profit_stop_dollar  lock when day PnL >= this number
  daily_loss_stop_dollar    lock when day PnL <= -this number
  session_only              only allow trades during one session
                            (asia | london | ny | london_open | ny_open)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from regime.filters import session_label   # already exists


@dataclass
class DailyRules:
    name: str
    max_trades_per_day: int | None = None
    stop_after_n_wins: int | None = None
    stop_after_n_losses: int | None = None
    daily_profit_stop_dollar: float | None = None
    daily_loss_stop_dollar: float | None = None
    session_only: str | None = None         # one of asia|london|ny|london_open|ny_open

    def label(self) -> str:
        bits = [self.name]
        if self.max_trades_per_day:    bits.append(f"max{self.max_trades_per_day}")
        if self.stop_after_n_wins:     bits.append(f"sw{self.stop_after_n_wins}")
        if self.stop_after_n_losses:   bits.append(f"sl{self.stop_after_n_losses}")
        if self.daily_profit_stop_dollar:  bits.append(f"dp{int(self.daily_profit_stop_dollar)}")
        if self.daily_loss_stop_dollar:    bits.append(f"dl{int(self.daily_loss_stop_dollar)}")
        if self.session_only: bits.append(self.session_only)
        return "_".join(bits)


@dataclass
class DayState:
    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0
    pnl: float = 0.0
    locked: bool = False


def session_in(session_only: str | None, ts) -> bool:
    if not session_only:
        return True
    if session_only == "london_open":
        return ts.hour in {7, 8}
    if session_only == "ny_open":
        return ts.hour in {12, 13}
    return session_label(ts) == session_only


def admit_trade(rules: DailyRules, day: DayState, ts) -> bool:
    """Return True if the trade is allowed under daily rules."""
    if day.locked:
        return False
    if rules.max_trades_per_day is not None and day.n_trades >= rules.max_trades_per_day:
        return False
    if not session_in(rules.session_only, ts):
        return False
    return True


def update_day(rules: DailyRules, day: DayState, dollar_pnl: float) -> None:
    day.n_trades += 1
    day.pnl += dollar_pnl
    if dollar_pnl > 0:
        day.n_wins += 1
    elif dollar_pnl < 0:
        day.n_losses += 1
    if rules.stop_after_n_wins is not None and day.n_wins >= rules.stop_after_n_wins:
        day.locked = True
    if rules.stop_after_n_losses is not None and day.n_losses >= rules.stop_after_n_losses:
        day.locked = True
    if rules.daily_profit_stop_dollar is not None and day.pnl >= rules.daily_profit_stop_dollar:
        day.locked = True
    if rules.daily_loss_stop_dollar is not None and day.pnl <= -rules.daily_loss_stop_dollar:
        day.locked = True


def all_rule_sets() -> list[DailyRules]:
    return [
        DailyRules(name="none"),
        DailyRules(name="max1",          max_trades_per_day=1),
        DailyRules(name="max2",          max_trades_per_day=2),
        DailyRules(name="max3",          max_trades_per_day=3),
        DailyRules(name="stop_w1",       stop_after_n_wins=1),
        DailyRules(name="stop_l1",       stop_after_n_losses=1),
        DailyRules(name="stop_l2",       stop_after_n_losses=2),
        DailyRules(name="dp250",         daily_profit_stop_dollar=250.0),
        DailyRules(name="dl250",         daily_loss_stop_dollar=250.0),
        DailyRules(name="dp500_dl300",   daily_profit_stop_dollar=500.0,
                                          daily_loss_stop_dollar=300.0),
        DailyRules(name="ny_only_max2",  max_trades_per_day=2, session_only="ny"),
        DailyRules(name="london_only_max2", max_trades_per_day=2, session_only="london"),
    ]
