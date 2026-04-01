from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import time

from src.core.config import FirmConfig, RiskConfig
from src.data.models import (
    Bar,
    ChallengeStatus,
    Fill,
    INSTRUMENT_SPECS,
    Order,
    Position,
)
from src.risk.drawdown_tracker import EODTrailingDrawdownTracker
from src.risk.lockout import LockoutManager
from src.risk.position_sizer import PositionSizer

logger = logging.getLogger(__name__)


@dataclass
class ApprovalResult:
    approved: bool
    quantity: int = 0
    reason: str = ""


class RiskEngine:
    """Central risk authority. Every order request passes through here.

    The orchestrator MUST call approve_order before sending any order.
    The risk engine can reject, reduce size, or approve.
    """

    def __init__(self, firm_config: FirmConfig, risk_config: RiskConfig):
        self.firm = firm_config
        self.risk = risk_config

        self.tracker = EODTrailingDrawdownTracker(
            starting_balance=firm_config.account_size,
            drawdown_amount=firm_config.trailing_drawdown.initial_amount,
            profit_target=firm_config.profit_target,
            trails_up=firm_config.trailing_drawdown.trails_up,
        )
        self.sizer = PositionSizer(firm_config, risk_config)
        self.lockout = LockoutManager(risk_config, firm_config.trading_hours)

        self.positions: dict[str, Position] = {}
        self._session_active = False

    @property
    def status(self) -> ChallengeStatus:
        return self.tracker.status

    @property
    def remaining_budget(self) -> float:
        return self.tracker.remaining_budget

    def start_session(self) -> None:
        self.tracker.start_session()
        self.lockout.reset()
        self._session_active = True

    def get_position_size(self, instrument: str, stop_distance_ticks: float) -> int:
        if self.lockout.is_locked():
            return 0
        return self.sizer.calculate(
            instrument=instrument,
            stop_distance_ticks=stop_distance_ticks,
            remaining_budget=self.tracker.remaining_budget,
            profit_so_far=self.tracker.profit_so_far,
        )

    def approve_order(self, order: Order) -> ApprovalResult:
        """Gate every order through risk checks."""
        if not self._session_active:
            return ApprovalResult(False, 0, "no active session")

        if self.tracker.status != ChallengeStatus.ACTIVE:
            return ApprovalResult(False, 0, f"challenge {self.tracker.status.value}")

        if self.lockout.is_locked():
            return ApprovalResult(False, 0, f"locked out: {self.lockout.state.lock_reason}")

        if order.quantity <= 0:
            return ApprovalResult(False, 0, "zero quantity")

        # Check instrument is allowed
        if order.instrument not in self.firm.allowed_instruments:
            return ApprovalResult(False, 0, f"{order.instrument} not allowed")

        # Check max contracts for instrument
        max_ct = self.firm.max_contracts.get(order.instrument, 1)
        existing_qty = 0
        if order.instrument in self.positions:
            existing_qty = self.positions[order.instrument].quantity

        available = max_ct - existing_qty
        if available <= 0:
            return ApprovalResult(False, 0, f"max contracts reached for {order.instrument}")

        approved_qty = min(order.quantity, available)

        # Check remaining budget can support this trade
        if self.tracker.remaining_budget <= self.risk.min_drawdown_buffer:
            return ApprovalResult(False, 0, "below min drawdown buffer")

        return ApprovalResult(True, approved_qty, "approved")

    def on_fill(self, fill: Fill, strategy_name: str = "") -> float | None:
        """Process a fill. Returns realized P&L if a position was closed, else None."""
        instrument = fill.instrument

        if instrument in self.positions:
            pos = self.positions[instrument]
            # Closing trade (opposite direction or same direction reduction)
            if fill.direction != pos.direction:
                spec = INSTRUMENT_SPECS[instrument]
                ticks = (fill.fill_price - pos.entry_price) / spec["tick_size"]
                realized_pnl = ticks * spec["tick_value"] * pos.quantity * pos.direction.value
                realized_pnl -= fill.commission

                self.tracker.on_realized_pnl(realized_pnl)
                self.lockout.on_trade_closed(realized_pnl, self.tracker.remaining_budget)

                del self.positions[instrument]
                logger.info(
                    f"Closed {instrument} {pos.direction.name} x{pos.quantity} "
                    f"@ {fill.fill_price}, P&L: {realized_pnl:+.2f}"
                )
                return realized_pnl
        else:
            # Opening trade
            self.positions[instrument] = Position(
                instrument=instrument,
                direction=fill.direction,
                quantity=fill.quantity,
                entry_price=fill.fill_price,
                entry_time=fill.timestamp,
                strategy_name=strategy_name,
            )
            logger.info(
                f"Opened {instrument} {fill.direction.name} x{fill.quantity} "
                f"@ {fill.fill_price}"
            )
        return None

    def on_bar(self, bar: Bar) -> None:
        """Update unrealized P&L and check time-based lockout."""
        if bar.instrument in self.positions:
            pos = self.positions[bar.instrument]
            spec = INSTRUMENT_SPECS[bar.instrument]
            pos.update_pnl(bar.close, spec["tick_value"], spec["tick_size"])

        # Sum all unrealized
        total_unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        self.tracker.update_unrealized(total_unrealized)

        # Check time-based lockout
        bar_time = bar.timestamp.time() if hasattr(bar.timestamp, "time") else time(0, 0)
        self.lockout.check_time_lockout(bar_time)
        self.lockout.should_protect_gains(bar_time)

    def flatten_all(self, current_prices: dict[str, float]) -> float:
        """Close all positions at given prices. Returns total realized P&L."""
        total_pnl = 0.0
        for instrument, pos in list(self.positions.items()):
            price = current_prices.get(instrument, pos.entry_price)
            spec = INSTRUMENT_SPECS[instrument]
            ticks = (price - pos.entry_price) / spec["tick_size"]
            pnl = ticks * spec["tick_value"] * pos.quantity * pos.direction.value
            self.tracker.on_realized_pnl(pnl)
            total_pnl += pnl
            logger.info(f"Flattened {instrument} @ {price}, P&L: {pnl:+.2f}")
        self.positions.clear()
        return total_pnl

    def on_session_close(self, current_prices: dict[str, float] | None = None) -> ChallengeStatus:
        """End of day. Flatten positions and update drawdown."""
        if self.positions and current_prices:
            self.flatten_all(current_prices)
        self._session_active = False
        return self.tracker.on_session_close()

    def get_state(self) -> dict:
        state = self.tracker.get_state()
        state["positions"] = {
            k: {"direction": v.direction.name, "qty": v.quantity, "entry": v.entry_price,
                "unrealized": v.unrealized_pnl}
            for k, v in self.positions.items()
        }
        state["lockout"] = {
            "locked": self.lockout.is_locked(),
            "reason": self.lockout.state.lock_reason,
            "daily_pnl": self.lockout.state.daily_pnl,
            "consecutive_losses": self.lockout.state.consecutive_losses,
        }
        return state
