from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class SignalDirection(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class ChallengeStatus(Enum):
    ACTIVE = "active"
    PASSED = "passed"
    BLOWN = "blown"


class MarketRegime(Enum):
    STRONG_TREND_UP = "strong_trend_up"
    STRONG_TREND_DOWN = "strong_trend_down"
    WEAK_TREND = "weak_trend"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    instrument: str
    timeframe: str = "5min"


@dataclass
class Signal:
    direction: SignalDirection
    instrument: str
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    strategy_name: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Order:
    instrument: str
    direction: SignalDirection
    quantity: int
    order_type: OrderType
    price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    strategy_name: str = ""
    timestamp: datetime | None = None
    order_id: str = ""
    status: OrderStatus = OrderStatus.PENDING


@dataclass
class Fill:
    order_id: str
    instrument: str
    direction: SignalDirection
    quantity: int
    fill_price: float
    timestamp: datetime
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class Position:
    instrument: str
    direction: SignalDirection
    quantity: int
    entry_price: float
    entry_time: datetime
    stop_loss: float | None = None
    take_profit: float | None = None
    strategy_name: str = ""
    unrealized_pnl: float = 0.0

    def update_pnl(self, current_price: float, tick_value: float, tick_size: float) -> None:
        ticks = (current_price - self.entry_price) / tick_size
        self.unrealized_pnl = ticks * tick_value * self.quantity * self.direction.value


@dataclass
class TradeRecord:
    instrument: str
    direction: SignalDirection
    quantity: int
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    commission: float
    strategy_name: str
    metadata: dict = field(default_factory=dict)


INSTRUMENT_SPECS: dict[str, dict] = {
    "ES": {"tick_size": 0.25, "tick_value": 12.50, "point_value": 50.0, "margin": 500},
    "NQ": {"tick_size": 0.25, "tick_value": 5.00, "point_value": 20.0, "margin": 500},
    "MES": {"tick_size": 0.25, "tick_value": 1.25, "point_value": 5.0, "margin": 50},
    "MNQ": {"tick_size": 0.25, "tick_value": 0.50, "point_value": 2.0, "margin": 50},
    "CL": {"tick_size": 0.01, "tick_value": 10.00, "point_value": 1000.0, "margin": 500},
    "GC": {"tick_size": 0.10, "tick_value": 10.00, "point_value": 100.0, "margin": 500},
    "MGC": {"tick_size": 0.10, "tick_value": 1.00, "point_value": 10.0, "margin": 50},
}
