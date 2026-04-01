from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DataFeed(ABC):
    @abstractmethod
    def get_bars(self, instrument: str, timeframe: str,
                 start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
        """Return OHLCV DataFrame with columns: open, high, low, close, volume.

        Index should be a DatetimeIndex.
        """

    def get_daily_bars(self, instrument: str,
                       start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
        return self.get_bars(instrument, "1d", start, end)


class CSVDataFeed(DataFeed):
    """Loads bar data from CSV files.

    Supports common export formats from NinjaTrader, Sierra Chart, TradingView.
    Expected columns (case-insensitive): date/time/datetime, open, high, low, close, volume.
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    def get_bars(self, instrument: str, timeframe: str,
                 start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
        # Try common file patterns
        patterns = [
            f"{instrument}_{timeframe}.csv",
            f"{instrument}_{timeframe}.parquet",
            f"{instrument}.csv",
            f"{instrument}.parquet",
        ]

        for pattern in patterns:
            path = self.data_dir / pattern
            if path.exists():
                return self._load_file(path, start, end)

        raise FileNotFoundError(
            f"No data file found for {instrument} {timeframe} in {self.data_dir}. "
            f"Tried: {', '.join(patterns)}"
        )

    def _load_file(self, path: Path,
                   start: datetime | None, end: datetime | None) -> pd.DataFrame:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        df = self._normalize_columns(df)
        df = self._parse_datetime_index(df)
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df["volume"] = df["volume"].fillna(0).astype(int)

        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]

        return df.sort_index()

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        col_map: dict[str, str] = {}
        for col in df.columns:
            lower = col.strip().lower()
            if lower in ("open", "o"):
                col_map[col] = "open"
            elif lower in ("high", "h"):
                col_map[col] = "high"
            elif lower in ("low", "l"):
                col_map[col] = "low"
            elif lower in ("close", "c", "last"):
                col_map[col] = "close"
            elif lower in ("volume", "vol", "v"):
                col_map[col] = "volume"
            elif lower in ("date", "time", "datetime", "timestamp", "date/time"):
                col_map[col] = "datetime"
        return df.rename(columns=col_map)

    @staticmethod
    def _parse_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
        if "datetime" in df.columns:
            df.index = pd.to_datetime(df["datetime"])
            df = df.drop(columns=["datetime"])
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df


class YFinanceDataFeed(DataFeed):
    """Downloads futures data from Yahoo Finance via yfinance."""

    SYMBOL_MAP = {
        "ES": "ES=F",
        "NQ": "NQ=F",
        "MES": "MES=F",
        "MNQ": "MNQ=F",
        "CL": "CL=F",
        "GC": "GC=F",
    }

    INTERVAL_MAP = {
        "1min": "1m",
        "2min": "2m",
        "5min": "5m",
        "15min": "15m",
        "30min": "30m",
        "1h": "1h",
        "1d": "1d",
    }

    def get_bars(self, instrument: str, timeframe: str,
                 start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
        import yfinance as yf

        symbol = self.SYMBOL_MAP.get(instrument, instrument)
        interval = self.INTERVAL_MAP.get(timeframe, timeframe)

        # yfinance limits: 1m data = 7 days, 5m = 60 days, etc.
        if interval in ("1m", "2m"):
            period = "7d"
        elif interval in ("5m", "15m", "30m"):
            period = "60d"
        elif interval == "1h":
            period = "730d"
        else:
            period = "max"

        kwargs: dict = {"interval": interval}
        if start and end:
            kwargs["start"] = start
            kwargs["end"] = end
        else:
            kwargs["period"] = period

        ticker = yf.Ticker(symbol)
        df = ticker.history(**kwargs)

        if df.empty:
            raise ValueError(f"No data returned for {symbol} ({interval})")

        df.columns = [c.lower() for c in df.columns]
        # yfinance returns: Open, High, Low, Close, Volume, Dividends, Stock Splits
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df["volume"] = df["volume"].fillna(0).astype(int)

        logger.info(f"Downloaded {len(df)} bars for {instrument} ({timeframe})")
        return df
