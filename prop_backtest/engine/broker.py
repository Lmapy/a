"""Simulated broker — order execution, fills, and PnL calculation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from prop_backtest.account.state import AccountState
from prop_backtest.data.loader import BarData


@dataclass
class Signal:
    """Trading signal produced by a strategy on each bar."""
    action: Literal["buy", "sell", "close", "hold"]
    contracts: int = 1
    # 'buy'/'sell': open or add to position with this many contracts.
    # 'close': close the entire open position.
    # 'hold': do nothing.


@dataclass
class Fill:
    """Execution record for a filled order."""
    bar_index: int
    timestamp: object          # datetime
    action: str
    direction: str             # "long" or "short"
    contracts: int
    fill_price: float
    commission: float          # total commission for this fill ($ per side)
    realized_pnl: float        # gross PnL from closing contracts (0 if opening)
    net_pnl: float             # realized_pnl - commission
    slippage_ticks: int = 0


class SimulatedBroker:
    """Processes signals and simulates order execution.

    Fill model:
    - Orders generated on bar N are filled at bar N+1's open price
      (next_open). This avoids lookahead bias since the strategy sees
      bar N's close before deciding, but can only act at the next open.
    - Slippage is applied as additional ticks in the adverse direction.
    - Commission is charged per contract per side (half of round-turn on
      open, half on close).
    """

    def __init__(
        self,
        commission_per_rt: float = 4.50,
        slippage_ticks: int = 0,
    ) -> None:
        self.commission_per_rt = commission_per_rt
        self.slippage_ticks = slippage_ticks

        self._pending_signal: Optional[Signal] = None
        self._pending_bar_index: int = -1

        # MAE/MFE tracking for the open trade
        self._trade_entry_bar_index: int = -1
        self._trade_min_price: float = float("inf")
        self._trade_max_price: float = float("-inf")

    def submit(self, signal: Signal, bar_index: int) -> None:
        """Queue a signal. It will be executed on the next bar's open."""
        if signal.action == "hold":
            return
        self._pending_signal = signal
        self._pending_bar_index = bar_index

    def process_pending(
        self,
        bar: BarData,
        bar_index: int,
        state: AccountState,
    ) -> Optional[Fill]:
        """Execute the pending order at bar's open (call at start of each bar).

        Returns a Fill if an order was executed, else None.
        Mutates AccountState: realized_balance, position_contracts, avg_entry_price.
        """
        if self._pending_signal is None:
            return None

        signal = self._pending_signal
        self._pending_signal = None

        action = signal.action
        fill_price = bar.open

        # Apply slippage: adverse ticks
        if action == "buy" or (action == "close" and state.position_contracts < 0):
            fill_price += self.slippage_ticks * bar.contract.tick_size
        elif action == "sell" or (action == "close" and state.position_contracts > 0):
            fill_price -= self.slippage_ticks * bar.contract.tick_size

        # Validate / clip to max_contracts
        max_c = state.tier.max_contracts
        if action in ("buy", "sell"):
            contracts = signal.contracts
            if max_c is not None:
                # Respect total position limit
                if action == "buy":
                    allowed = max_c - max(0, state.position_contracts)
                else:
                    allowed = max_c - max(0, -state.position_contracts)
                contracts = min(contracts, max(0, allowed))
            if contracts <= 0:
                return None  # position limit hit, skip
        else:
            contracts = abs(state.position_contracts)
            if contracts == 0:
                return None  # nothing to close

        commission_this_side = (self.commission_per_rt / 2.0) * contracts
        realized_pnl = 0.0
        direction = "long" if action == "buy" else "short"

        if action == "close":
            direction = "long" if state.position_contracts > 0 else "short"
            realized_pnl = state.contract.pnl(
                state.avg_entry_price,
                fill_price,
                state.position_contracts,
                is_short=(state.position_contracts < 0),
            )
            # Also charge the open-side commission that was deferred to now
            close_commission = (self.commission_per_rt / 2.0) * contracts
            total_commission = close_commission + commission_this_side
            realized_pnl -= total_commission

            state.realized_balance += realized_pnl + total_commission - total_commission
            # Simplified: net the commission into realized_pnl already
            state.realized_balance += realized_pnl
            state.open_pnl = 0.0
            state.position_contracts = 0
            state.avg_entry_price = 0.0

            fill = Fill(
                bar_index=bar_index,
                timestamp=bar.timestamp,
                action=action,
                direction=direction,
                contracts=contracts,
                fill_price=fill_price,
                commission=total_commission,
                realized_pnl=realized_pnl + total_commission,
                net_pnl=realized_pnl,
                slippage_ticks=self.slippage_ticks,
            )
            self._reset_excursion()
            return fill

        # Opening or adding to position
        commission_open = commission_this_side
        state.realized_balance -= commission_open

        prev_contracts = state.position_contracts
        if action == "buy":
            new_contracts = prev_contracts + contracts
        else:
            new_contracts = prev_contracts - contracts

        # Update average entry price
        if new_contracts == 0:
            state.avg_entry_price = 0.0
        elif prev_contracts == 0:
            state.avg_entry_price = fill_price
        elif (prev_contracts > 0 and action == "buy") or (prev_contracts < 0 and action == "sell"):
            # Adding to position — volume-weighted average
            total_old = abs(prev_contracts)
            total_new = contracts
            state.avg_entry_price = (
                (state.avg_entry_price * total_old + fill_price * total_new)
                / (total_old + total_new)
            )
        else:
            # Partial close — entry price unchanged
            if abs(new_contracts) < abs(prev_contracts):
                partial_pnl = state.contract.pnl(
                    state.avg_entry_price,
                    fill_price,
                    contracts,
                    is_short=(prev_contracts < 0),
                )
                close_commission = commission_this_side  # already deducted above as open side
                partial_pnl -= close_commission
                state.realized_balance += partial_pnl
                realized_pnl = partial_pnl
            # else: reversal — handle as a close then open (simplified: just move forward)

        state.position_contracts = new_contracts
        if new_contracts == 0:
            state.open_pnl = 0.0
            self._reset_excursion()

        if self._trade_entry_bar_index < 0 and new_contracts != 0:
            self._trade_entry_bar_index = bar_index
            self._trade_min_price = bar.low
            self._trade_max_price = bar.high

        return Fill(
            bar_index=bar_index,
            timestamp=bar.timestamp,
            action=action,
            direction=direction,
            contracts=contracts,
            fill_price=fill_price,
            commission=commission_open,
            realized_pnl=realized_pnl,
            net_pnl=realized_pnl - commission_open,
            slippage_ticks=self.slippage_ticks,
        )

    def update_excursion(self, bar: BarData) -> None:
        """Update price excursion tracking for the open position."""
        if self._trade_entry_bar_index >= 0:
            self._trade_min_price = min(self._trade_min_price, bar.low)
            self._trade_max_price = max(self._trade_max_price, bar.high)

    def get_excursion(self, state: AccountState) -> tuple[float, float]:
        """Return (MAE, MFE) in dollars for the current open trade."""
        if state.position_contracts == 0 or self._trade_entry_bar_index < 0:
            return 0.0, 0.0
        entry = state.avg_entry_price
        is_long = state.position_contracts > 0
        c = abs(state.position_contracts)
        if is_long:
            mae = state.contract.pnl(entry, self._trade_min_price, c, is_short=False)
            mfe = state.contract.pnl(entry, self._trade_max_price, c, is_short=False)
        else:
            mae = state.contract.pnl(entry, self._trade_max_price, c, is_short=True)
            mfe = state.contract.pnl(entry, self._trade_min_price, c, is_short=True)
        return min(mae, 0.0), max(mfe, 0.0)

    def _reset_excursion(self) -> None:
        self._trade_entry_bar_index = -1
        self._trade_min_price = float("inf")
        self._trade_max_price = float("-inf")
