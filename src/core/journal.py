from __future__ import annotations

import csv
import logging
from pathlib import Path

from src.data.models import TradeRecord

logger = logging.getLogger(__name__)


class TradeJournal:
    """Writes trade records to a CSV file for analysis."""

    HEADERS = [
        "entry_time", "exit_time", "instrument", "direction", "quantity",
        "entry_price", "exit_price", "pnl", "commission", "strategy",
    ]

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.trades: list[TradeRecord] = []

    def add_trade(self, trade: TradeRecord) -> None:
        self.trades.append(trade)

    def add_trades(self, trades: list[TradeRecord]) -> None:
        self.trades.extend(trades)

    def save(self) -> None:
        with open(self.output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.HEADERS)
            for t in self.trades:
                writer.writerow([
                    t.entry_time.isoformat() if t.entry_time else "",
                    t.exit_time.isoformat() if t.exit_time else "",
                    t.instrument,
                    t.direction.name,
                    t.quantity,
                    f"{t.entry_price:.2f}",
                    f"{t.exit_price:.2f}",
                    f"{t.pnl:.2f}",
                    f"{t.commission:.2f}",
                    t.strategy_name,
                ])
        logger.info(f"Saved {len(self.trades)} trades to {self.output_path}")
