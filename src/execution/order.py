from __future__ import annotations

import uuid
from datetime import datetime

from src.data.models import Order, OrderType, SignalDirection, Signal, INSTRUMENT_SPECS


def create_order_from_signal(signal: Signal, quantity: int,
                             order_type: OrderType = OrderType.MARKET) -> Order:
    """Create an Order from a trading Signal."""
    return Order(
        instrument=signal.instrument,
        direction=signal.direction,
        quantity=quantity,
        order_type=order_type,
        price=signal.entry_price if order_type != OrderType.MARKET else None,
        stop_loss=signal.stop_loss,
        take_profit=signal.take_profit,
        strategy_name=signal.strategy_name,
        timestamp=datetime.now(),
        order_id=str(uuid.uuid4())[:8],
    )


def calculate_stop_ticks(entry_price: float, stop_price: float,
                         instrument: str) -> float:
    """Calculate number of ticks between entry and stop loss."""
    spec = INSTRUMENT_SPECS.get(instrument)
    if not spec:
        return 0
    return abs(entry_price - stop_price) / spec["tick_size"]
