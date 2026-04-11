from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd


class ExchangeAdapter(ABC):
    exchange_name: str

    @abstractmethod
    def fetch_ohlcv(self, symbol: str, start: datetime, end: datetime, timeframe: str = "1m") -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def fetch_funding(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def fetch_open_interest(self, symbol: str, start: datetime, end: datetime, timeframe: str = "1m") -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def fetch_mark_index(self, symbol: str, start: datetime, end: datetime, timeframe: str = "1m") -> pd.DataFrame:
        raise NotImplementedError
