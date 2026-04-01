from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import time

from src.core.config import RiskConfig, TradingHoursConfig

logger = logging.getLogger(__name__)


@dataclass
class LockoutState:
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    trade_count: int = 0
    locked: bool = False
    lock_reason: str = ""


class LockoutManager:
    """Determines when to stop trading for the day.

    Rules:
        1. Daily loss exceeds threshold % of remaining drawdown budget
        2. Too many consecutive losses (tilt prevention)
        3. Profitable day in afternoon: protect gains (EOD drawdown specific!)
        4. Too close to session end: flatten and done
    """

    def __init__(self, risk_config: RiskConfig, trading_hours: TradingHoursConfig):
        self.risk = risk_config
        self.trading_hours = trading_hours
        self.state = LockoutState()

        # Parse session end time
        parts = trading_hours.end.split(":")
        self.session_end = time(int(parts[0]), int(parts[1]))

    def reset(self) -> None:
        """Call at start of each trading day."""
        self.state = LockoutState()

    def on_trade_closed(self, pnl: float, remaining_budget: float) -> None:
        """Update state after a trade closes."""
        self.state.daily_pnl += pnl
        self.state.trade_count += 1

        if pnl < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

        self._check_lockout(remaining_budget)

    def check_time_lockout(self, current_time: time) -> bool:
        """Check if we're too close to session end."""
        if not self.trading_hours.flat_by_close:
            return False

        # Calculate minutes until session end
        end_mins = self.session_end.hour * 60 + self.session_end.minute
        cur_mins = current_time.hour * 60 + current_time.minute
        mins_left = end_mins - cur_mins

        if mins_left <= self.risk.end_of_day_flatten_minutes:
            self.state.locked = True
            self.state.lock_reason = f"session ending in {mins_left} min"
            return True
        return False

    def should_protect_gains(self, current_time: time) -> bool:
        """In the afternoon, if we're up nicely, stop trading to protect EOD balance.

        This is crucial for EOD drawdown: if we give back gains, the HWM
        still went up (or will go up) but our balance dropped back — a double penalty.
        """
        if self.state.daily_pnl < self.risk.profit_protection_threshold:
            return False

        # Only protect in the afternoon (after 12:00)
        if current_time.hour < 12:
            return False

        self.state.locked = True
        self.state.lock_reason = (
            f"protecting daily gains of ${self.state.daily_pnl:.0f} in afternoon"
        )
        logger.info(f"LOCKOUT: {self.state.lock_reason}")
        return True

    def is_locked(self) -> bool:
        return self.state.locked

    def _check_lockout(self, remaining_budget: float) -> None:
        if self.state.locked:
            return

        # Rule 1: Daily loss exceeds threshold
        max_daily_loss = remaining_budget * (self.risk.max_daily_loss_pct / 100.0)
        if self.state.daily_pnl < -max_daily_loss:
            self.state.locked = True
            self.state.lock_reason = (
                f"daily loss ${self.state.daily_pnl:.0f} exceeds "
                f"max ${-max_daily_loss:.0f}"
            )
            logger.warning(f"LOCKOUT: {self.state.lock_reason}")
            return

        # Rule 2: Consecutive losses (tilt prevention)
        if self.state.consecutive_losses >= self.risk.max_consecutive_losses:
            self.state.locked = True
            self.state.lock_reason = (
                f"{self.state.consecutive_losses} consecutive losses"
            )
            logger.warning(f"LOCKOUT: {self.state.lock_reason}")
            return
