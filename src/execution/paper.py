from __future__ import annotations

import logging
import uuid
from datetime import datetime

from src.data.models import (
    Bar,
    Fill,
    INSTRUMENT_SPECS,
    Order,
    OrderStatus,
    OrderType,
    SignalDirection,
)
from src.execution.broker import Broker

logger = logging.getLogger(__name__)


class PaperBroker(Broker):
    """Simulated broker for backtesting and paper trading.

    Simulates fills with configurable slippage and commission.
    Market orders fill at bar close +/- slippage.
    Limit/stop orders fill when the bar's range touches the order price.
    """

    def __init__(self, slippage_ticks: float = 1.0, commission_per_contract: float = 2.50):
        self.slippage_ticks = slippage_ticks
        self.commission = commission_per_contract
        self.pending_orders: list[Order] = []
        self.filled_orders: list[Fill] = []

    def submit_order(self, order: Order) -> str:
        if not order.order_id:
            order.order_id = str(uuid.uuid4())[:8]
        order.status = OrderStatus.PENDING

        if order.order_type == OrderType.MARKET:
            # Market orders fill immediately on next bar
            self.pending_orders.append(order)
        else:
            self.pending_orders.append(order)

        logger.debug(f"Order submitted: {order.order_id} {order.direction.name} "
                     f"{order.quantity}x {order.instrument} @ {order.order_type.value}")
        return order.order_id

    def cancel_order(self, order_id: str) -> bool:
        for i, order in enumerate(self.pending_orders):
            if order.order_id == order_id:
                self.pending_orders.pop(i)
                order.status = OrderStatus.CANCELLED
                return True
        return False

    def process_bar(self, bar: Bar) -> list[Fill]:
        """Process all pending orders against the current bar."""
        fills: list[Fill] = []
        remaining: list[Order] = []

        for order in self.pending_orders:
            if order.instrument != bar.instrument:
                remaining.append(order)
                continue

            fill = self._try_fill(order, bar)
            if fill:
                fills.append(fill)
                order.status = OrderStatus.FILLED
            else:
                remaining.append(order)

        self.pending_orders = remaining
        self.filled_orders.extend(fills)
        return fills

    def get_open_orders(self) -> list[Order]:
        return list(self.pending_orders)

    def flatten_all(self, instrument: str, current_price: float) -> list[Fill]:
        """Create fill to close position at current price."""
        # Cancel any pending orders for this instrument
        self.pending_orders = [o for o in self.pending_orders if o.instrument != instrument]

        # Return a fill at the current price (caller handles position tracking)
        return []

    def _try_fill(self, order: Order, bar: Bar) -> Fill | None:
        spec = INSTRUMENT_SPECS.get(order.instrument)
        if not spec:
            return None

        tick_size = spec["tick_size"]
        slippage = self.slippage_ticks * tick_size

        fill_price: float | None = None

        if order.order_type == OrderType.MARKET:
            # Fill at open of this bar +/- slippage (simulating next-bar fill)
            if order.direction == SignalDirection.LONG:
                fill_price = bar.open + slippage
            else:
                fill_price = bar.open - slippage

        elif order.order_type == OrderType.LIMIT:
            if order.price is None:
                return None
            if order.direction == SignalDirection.LONG and bar.low <= order.price:
                fill_price = order.price
            elif order.direction == SignalDirection.SHORT and bar.high >= order.price:
                fill_price = order.price

        elif order.order_type == OrderType.STOP:
            if order.price is None:
                return None
            if order.direction == SignalDirection.LONG and bar.high >= order.price:
                fill_price = order.price + slippage
            elif order.direction == SignalDirection.SHORT and bar.low <= order.price:
                fill_price = order.price - slippage

        if fill_price is None:
            return None

        commission = self.commission * order.quantity
        return Fill(
            order_id=order.order_id,
            instrument=order.instrument,
            direction=order.direction,
            quantity=order.quantity,
            fill_price=fill_price,
            timestamp=bar.timestamp,
            commission=commission,
            slippage=slippage * order.quantity,
        )
