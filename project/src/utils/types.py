from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


ExchangeName = Literal["bybit", "hyperliquid", "binance"]


@dataclass(frozen=True)
class DownloadRequest:
    exchange: ExchangeName
    symbol: str
    start: datetime
    end: datetime
    timeframe: str = "1m"


CANONICAL_COLUMNS = [
    "timestamp",
    "exchange",
    "symbol",
    "timeframe",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "funding_rate",
    "open_interest",
    "mark_price",
    "index_price",
]
