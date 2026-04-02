"""Prop firm rule tracking: drawdown, daily loss, consistency."""

from dataclasses import dataclass, field
from config import RULES, PARAMS


class EvaluationStatus:
    ACTIVE = "active"
    PASSED = "passed"
    FAILED_DRAWDOWN = "failed_drawdown"
    FAILED_TIME = "failed_time"


@dataclass
class PropFirmTracker:
    """Tracks prop firm evaluation rules in real-time."""
    starting_balance: float = RULES.starting_balance
    balance: float = RULES.starting_balance
    max_eod_balance: float = RULES.starting_balance
    trailing_dd_floor: float = RULES.starting_balance - RULES.max_trailing_drawdown
    daily_pnl: float = 0.0
    cumulative_pnl: float = 0.0
    best_day_pnl: float = 0.0
    trading_days: int = 0
    trades_today: int = 0
    status: str = EvaluationStatus.ACTIVE
    daily_pnl_history: list = field(default_factory=list)

    def start_new_day(self):
        """Called at the start of each trading day."""
        if self.daily_pnl != 0 or self.trades_today > 0:
            self.daily_pnl_history.append(self.daily_pnl)
            if self.daily_pnl > self.best_day_pnl:
                self.best_day_pnl = self.daily_pnl
            self.trading_days += 1

        self.daily_pnl = 0.0
        self.trades_today = 0

        # Update EOD trailing drawdown
        if self.balance > self.max_eod_balance:
            self.max_eod_balance = self.balance
            new_floor = self.max_eod_balance - RULES.max_trailing_drawdown
            # Floor only moves up, never down; locks at starting balance
            self.trailing_dd_floor = min(
                max(new_floor, self.trailing_dd_floor),
                self.starting_balance,
            )

    def record_trade(self, pnl: float) -> bool:
        """Record a trade and check if we can continue trading.

        Returns True if trading can continue, False if must stop.
        """
        self.daily_pnl += pnl
        self.cumulative_pnl += pnl
        self.balance += pnl
        self.trades_today += 1

        # Check drawdown
        if self.balance <= self.trailing_dd_floor:
            self.status = EvaluationStatus.FAILED_DRAWDOWN
            return False

        # Check if passed (must also meet consistency rule)
        if self.cumulative_pnl >= RULES.profit_target:
            if self._check_consistency():
                self.status = EvaluationStatus.PASSED
            return False

        return True

    def can_trade(self) -> bool:
        """Check if we're allowed to take another trade right now."""
        if self.status != EvaluationStatus.ACTIVE:
            return False

        # Max trading days check
        if self.trading_days >= RULES.max_trading_days:
            self.status = EvaluationStatus.FAILED_TIME
            return False

        # Max trades per day
        if self.trades_today >= PARAMS.max_trades_per_day:
            return False

        # Daily loss gate (conservative buffer)
        if RULES.daily_loss_limit is not None:
            if self.daily_pnl <= -PARAMS.daily_loss_gate:
                return False

        # Daily profit gate (consistency buffer)
        if self.daily_pnl >= PARAMS.daily_profit_gate:
            return False

        # Early stop: if we've lost 3+ trades in a row and are deep negative,
        # reduce trading to preserve capital
        if self.cumulative_pnl < -1500:
            return False  # stop trading to avoid bust

        return True

    def end_evaluation(self):
        """Finalize the evaluation."""
        # Record last day
        if self.daily_pnl != 0 or self.trades_today > 0:
            self.daily_pnl_history.append(self.daily_pnl)
            if self.daily_pnl > self.best_day_pnl:
                self.best_day_pnl = self.daily_pnl
            self.trading_days += 1

        if self.status == EvaluationStatus.ACTIVE:
            if self.cumulative_pnl >= RULES.profit_target:
                if self._check_consistency():
                    self.status = EvaluationStatus.PASSED
            elif self.trading_days >= RULES.max_trading_days:
                self.status = EvaluationStatus.FAILED_TIME

    def _check_consistency(self) -> bool:
        """Check the consistency rule: best day <= 50% of profit target.

        Returns True if consistency is met, False if violated.
        In real evals, you must keep trading until best day < 50% of total profit.
        """
        max_allowed = RULES.profit_target * RULES.consistency_pct
        if self.best_day_pnl > max_allowed:
            # Consistency violated - need to keep trading
            # Check if best day is < 50% of total accumulated profit
            if self.cumulative_pnl > 0:
                return self.best_day_pnl <= self.cumulative_pnl * 0.5
            return False
        return True

    def get_summary(self) -> dict:
        """Get evaluation summary."""
        return {
            "status": self.status,
            "cumulative_pnl": round(self.cumulative_pnl, 2),
            "balance": round(self.balance, 2),
            "trading_days": self.trading_days,
            "best_day_pnl": round(self.best_day_pnl, 2),
            "max_eod_balance": round(self.max_eod_balance, 2),
            "trailing_dd_floor": round(self.trailing_dd_floor, 2),
            "daily_pnl_history": [round(x, 2) for x in self.daily_pnl_history],
        }
