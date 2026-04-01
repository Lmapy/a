from __future__ import annotations

from abc import ABC, abstractmethod

from src.data.models import Bar, Fill, Order


class Broker(ABC):
    @abstractmethod
    def submit_order(self, order: Order) -> str:
        """Submit an order. Returns order_id."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""

    @abstractmethod
    def process_bar(self, bar: Bar) -> list[Fill]:
        """Process a bar and return any fills that occurred."""

    @abstractmethod
    def get_open_orders(self) -> list[Order]:
        """Get all pending orders."""

    @abstractmethod
    def flatten_all(self, instrument: str, current_price: float) -> list[Fill]:
        """Close all positions for an instrument at current price."""
