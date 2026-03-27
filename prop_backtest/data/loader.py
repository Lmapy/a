"""Market data loading — yfinance and local CSV."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import pandas as pd

from prop_backtest.contracts.specs import ContractSpec

# yfinance interval → approximate max history depth
_YFINANCE_INTERVAL_LIMITS: dict[str, str] = {
    "1m": "7 days",
    "2m": "60 days",
    "5m": "60 days",
    "15m": "60 days",
    "30m": "60 days",
    "60m": "730 days",
    "1h": "730 days",
    "1d": "unlimited",
    "1wk": "unlimited",
    "1mo": "unlimited",
}


@dataclass
class BarData:
    """A single OHLCV bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    contract: ContractSpec

    @property
    def date(self) -> date:
        return self.timestamp.date()


class DataLoader:
    """Load OHLCV bars from yfinance or a local CSV file.

    Notes on yfinance futures data:
    - Continuous front-month contract: ES=F, NQ=F, CL=F etc.
    - Daily bars: typically ~1 year of history available.
    - Hourly (1h/60m): up to ~730 days.
    - 15m and shorter: up to ~60 days only.
    - Data is not back-adjusted at roll dates; gaps may appear.
    - For production use, a premium data source (Norgate, CQG, Barchart) is recommended.
    """

    def __init__(self, contract: ContractSpec):
        self.contract = contract

    def load(
        self,
        start: str | date,
        end: str | date,
        interval: str = "1d",
        local_csv_path: Optional[str | Path] = None,
    ) -> list[BarData]:
        """Load bars for the given date range.

        Args:
            start: Start date, e.g. "2024-01-01" or datetime.date object.
            end: End date (inclusive).
            interval: yfinance interval string, e.g. "1d", "1h", "15m".
            local_csv_path: If provided, load from a CSV file instead of yfinance.
                            Expected columns: datetime, open, high, low, close, volume.

        Returns:
            List of BarData sorted by timestamp ascending.
        """
        if local_csv_path is not None:
            return self._load_csv(Path(local_csv_path))

        return self._load_yfinance(str(start), str(end), interval)

    def _load_yfinance(self, start: str, end: str, interval: str) -> list[BarData]:
        try:
            import yfinance as yf
        except ImportError as e:
            raise ImportError("yfinance is required. Install with: pip install yfinance") from e

        if interval not in _YFINANCE_INTERVAL_LIMITS:
            raise ValueError(
                f"Unsupported interval '{interval}'. "
                f"Supported: {', '.join(_YFINANCE_INTERVAL_LIMITS)}"
            )

        limit = _YFINANCE_INTERVAL_LIMITS[interval]
        if limit != "unlimited":
            warnings.warn(
                f"yfinance interval '{interval}' supports at most {limit} of history. "
                "Consider using a local CSV for longer backtests.",
                stacklevel=3,
            )

        ticker = self.contract.yfinance_ticker
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False, auto_adjust=True)

        if df.empty:
            raise ValueError(
                f"No data returned for {ticker} ({start} to {end}, interval={interval}). "
                "Check the date range and ticker symbol."
            )

        # yfinance may return a MultiIndex for single ticker
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [c.lower() for c in df.columns]
        df = df.dropna(subset=["open", "high", "low", "close"])

        bars: list[BarData] = []
        for ts, row in df.iterrows():
            # Ensure timezone-naive datetime
            if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                ts = ts.tz_localize(None)
            bars.append(BarData(
                timestamp=pd.Timestamp(ts).to_pydatetime(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row.get("volume", 0)),
                contract=self.contract,
            ))

        if not bars:
            raise ValueError(f"All bars were NaN after cleaning for {ticker}.")

        return bars

    def _load_csv(self, path: Path) -> list[BarData]:
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]

        # Find datetime column
        dt_col = None
        for candidate in ["datetime", "timestamp", "date", "time"]:
            if candidate in df.columns:
                dt_col = candidate
                break
        if dt_col is None:
            raise ValueError(
                f"CSV must have a datetime column named one of: datetime, timestamp, date, time. "
                f"Found columns: {list(df.columns)}"
            )

        df[dt_col] = pd.to_datetime(df[dt_col])
        df = df.sort_values(dt_col).reset_index(drop=True)
        df = df.dropna(subset=["open", "high", "low", "close"])

        bars: list[BarData] = []
        for _, row in df.iterrows():
            ts = row[dt_col]
            if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                ts = ts.tz_localize(None)
            bars.append(BarData(
                timestamp=ts.to_pydatetime(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row.get("volume", 0)) if "volume" in row else 0,
                contract=self.contract,
            ))

        return bars

    @staticmethod
    def save_csv(bars: list[BarData], path: str | Path) -> None:
        """Cache bars to a CSV file for re-use without re-downloading."""
        records = [
            {
                "datetime": b.timestamp.isoformat(),
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
            }
            for b in bars
        ]
        pd.DataFrame(records).to_csv(path, index=False)
