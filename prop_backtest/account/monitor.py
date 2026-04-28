"""Rule monitor — enforces prop firm rules bar-by-bar."""
from __future__ import annotations

from prop_backtest.data.loader import BarData
from prop_backtest.firms.base import DrawdownType
from .state import AccountState


class RuleMonitor:
    """Evaluates prop firm rules on each bar and at end-of-day.

    All mutation of AccountState passes through this class so rule
    logic is centralised in one place.
    """

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def update_open_pnl(self, state: AccountState, bar: BarData) -> None:
        """Recompute open_pnl from the bar's close price."""
        if state.position_contracts == 0:
            state.open_pnl = 0.0
            return

        is_short = state.position_contracts < 0
        state.open_pnl = state.contract.pnl(
            state.avg_entry_price,
            bar.close,
            state.position_contracts,
            is_short=is_short,
        )

        # Track price excursion (MAE/MFE)
        if state.position_min_price == 0.0:
            state.position_min_price = bar.low
            state.position_max_price = bar.high
        else:
            state.position_min_price = min(state.position_min_price, bar.low)
            state.position_max_price = max(state.position_max_price, bar.high)

    def update_intraday_hwm(self, state: AccountState) -> None:
        """Update intraday high-water mark and recompute drawdown floor.
        Called after open_pnl is updated. For TRAILING_INTRADAY firms only.
        """
        if state.firm_rules.drawdown_type != DrawdownType.TRAILING_INTRADAY:
            return
        if state.equity > state.intraday_hwm:
            state.intraday_hwm = state.equity
        state.drawdown_floor = self._compute_floor(state)

    def update_eod(self, state: AccountState) -> None:
        """End-of-day update: refresh EOD HWM (for TRAILING_EOD firms),
        reset daily balance tracker.
        Called when the engine detects a date boundary.
        """
        if state.firm_rules.drawdown_type == DrawdownType.TRAILING_EOD:
            # Only closed (realized) balance counts toward EOD HWM
            if state.realized_balance > state.eod_hwm:
                state.eod_hwm = state.realized_balance
            state.drawdown_floor = self._compute_floor(state)

        # Reset day tracking regardless of drawdown type
        state.day_start_balance = state.realized_balance

    def check_violations(self, state: AccountState) -> bool:
        """Check all rules and set violation flags. Returns True if terminated."""
        if state.is_terminated:
            return True

        # ── trailing drawdown breach ──────────────────────────────────────
        if state.equity <= state.drawdown_floor:
            state.breached_trailing_dd = True
            state.is_terminated = True
            return True

        # ── daily loss limit ──────────────────────────────────────────────
        # Include open PnL in the daily loss check for intraday firms;
        # for EOD firms, only check realized PnL.
        if state.firm_rules.drawdown_type == DrawdownType.TRAILING_INTRADAY:
            daily_pnl = (state.realized_balance - state.day_start_balance) + state.open_pnl
        else:
            daily_pnl = state.realized_balance - state.day_start_balance

        if daily_pnl <= -state.tier.daily_loss_limit:
            state.breached_daily_loss = True
            state.is_terminated = True
            return True

        # ── profit target ─────────────────────────────────────────────────
        # Only realized balance counts toward passing
        if state.realized_balance >= state.starting_balance + state.tier.profit_target:
            state.hit_profit_target = True
            state.is_terminated = True
            return True

        return False

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _compute_floor(self, state: AccountState) -> float:
        """Compute the current drawdown floor.

        Lock-to-breakeven rule: once the HWM has risen enough that the raw
        floor would exceed starting_balance, the floor is capped at
        starting_balance. This prevents the floor from ever going above the
        starting balance (i.e. a profit is never put at risk by the floor).
        """
        tier = state.tier
        dt = state.firm_rules.drawdown_type

        if dt == DrawdownType.TRAILING_INTRADAY:
            hwm = state.intraday_hwm
        elif dt == DrawdownType.TRAILING_EOD:
            hwm = state.eod_hwm
        else:  # STATIC
            return state.starting_balance - tier.max_trailing_drawdown

        raw_floor = hwm - tier.max_trailing_drawdown
        # Lock-to-breakeven: floor never exceeds starting_balance
        return min(raw_floor, state.starting_balance)
