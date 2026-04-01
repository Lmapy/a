from __future__ import annotations

import logging

from src.data.models import ChallengeStatus

logger = logging.getLogger(__name__)


class EODTrailingDrawdownTracker:
    """Tracks EOD trailing drawdown per prop firm rules.

    With EOD trailing drawdown, the high-water mark (HWM) only updates at
    end-of-day based on the closing balance. Intraday unrealized P&L does
    NOT affect the drawdown calculation. This is a key advantage we exploit:
    we can tolerate intraday heat as long as we close the day above the floor.

    The drawdown floor = HWM - drawdown_amount. If the closing balance
    hits or goes below the floor, the account is blown.
    """

    def __init__(self, starting_balance: float, drawdown_amount: float,
                 profit_target: float, trails_up: bool = True):
        self.starting_balance = starting_balance
        self.drawdown_amount = drawdown_amount
        self.profit_target = profit_target
        self.trails_up = trails_up

        self.high_water_mark = starting_balance
        self.drawdown_floor = starting_balance - drawdown_amount
        self.current_balance = starting_balance
        self.unrealized_pnl = 0.0
        self.day_count = 0
        self.status = ChallengeStatus.ACTIVE

        self._daily_open_balance = starting_balance

    @property
    def remaining_budget(self) -> float:
        """How much we can lose before hitting the drawdown floor (based on current balance)."""
        return self.current_balance - self.drawdown_floor

    @property
    def profit_so_far(self) -> float:
        return self.current_balance - self.starting_balance

    @property
    def profit_remaining(self) -> float:
        return self.profit_target - self.profit_so_far

    @property
    def daily_pnl(self) -> float:
        return self.current_balance - self._daily_open_balance

    def start_session(self) -> None:
        """Call at the start of each trading day."""
        self._daily_open_balance = self.current_balance
        self.unrealized_pnl = 0.0
        self.day_count += 1
        logger.info(
            f"Day {self.day_count}: balance={self.current_balance:.2f}, "
            f"HWM={self.high_water_mark:.2f}, floor={self.drawdown_floor:.2f}, "
            f"remaining_budget={self.remaining_budget:.2f}, "
            f"profit_so_far={self.profit_so_far:.2f}"
        )

    def on_realized_pnl(self, pnl: float) -> None:
        """Called when a trade is closed and P&L is realized."""
        self.current_balance += pnl
        logger.debug(f"Realized P&L: {pnl:+.2f}, balance: {self.current_balance:.2f}")

    def update_unrealized(self, unrealized: float) -> None:
        """Update unrealized P&L from open positions (informational only for EOD)."""
        self.unrealized_pnl = unrealized

    def on_session_close(self) -> ChallengeStatus:
        """Called at end of trading day. Updates HWM and checks status.

        All positions must be flat before calling this (unrealized = 0).
        """
        closing_balance = self.current_balance  # positions are flat

        # Update high-water mark if we closed higher
        if self.trails_up and closing_balance > self.high_water_mark:
            old_hwm = self.high_water_mark
            self.high_water_mark = closing_balance
            self.drawdown_floor = self.high_water_mark - self.drawdown_amount
            logger.info(
                f"HWM updated: {old_hwm:.2f} -> {self.high_water_mark:.2f}, "
                f"new floor: {self.drawdown_floor:.2f}"
            )

        # Check if blown
        if closing_balance <= self.drawdown_floor:
            self.status = ChallengeStatus.BLOWN
            logger.warning(
                f"ACCOUNT BLOWN on day {self.day_count}! "
                f"Balance: {closing_balance:.2f}, floor: {self.drawdown_floor:.2f}"
            )
            return self.status

        # Check if passed
        if self.profit_so_far >= self.profit_target:
            self.status = ChallengeStatus.PASSED
            logger.info(
                f"CHALLENGE PASSED on day {self.day_count}! "
                f"Balance: {closing_balance:.2f}, profit: {self.profit_so_far:.2f}"
            )
            return self.status

        self.status = ChallengeStatus.ACTIVE
        logger.info(
            f"Day {self.day_count} close: balance={closing_balance:.2f}, "
            f"daily P&L={self.daily_pnl:+.2f}, profit_so_far={self.profit_so_far:.2f}"
        )
        return self.status

    def get_state(self) -> dict:
        return {
            "day_count": self.day_count,
            "current_balance": self.current_balance,
            "high_water_mark": self.high_water_mark,
            "drawdown_floor": self.drawdown_floor,
            "remaining_budget": self.remaining_budget,
            "profit_so_far": self.profit_so_far,
            "profit_remaining": self.profit_remaining,
            "daily_pnl": self.daily_pnl,
            "status": self.status.value,
        }
