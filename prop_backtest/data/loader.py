"""Market data loading — yfinance, Barchart, and local CSV."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import pandas as pd

from prop_backtest.contracts.specs import ContractSpec

# Barchart symbol map: ContractSpec.symbol → Barchart root + continuous suffix
_BARCHART_SYMBOL_MAP: dict[str, str] = {
    "ES": "ESc1",   # S&P 500 E-mini continuous front month
    "NQ": "NQc1",   # Nasdaq E-mini
    "CL": "CLc1",   # Crude Oil WTI
    "GC": "GCc1",   # Gold
    "RTY": "RTYc1", # Russell 2000 E-mini
    "MES": "MESc1", # Micro S&P 500
    "MNQ": "MNQc1", # Micro Nasdaq
}

# Barchart interval string mapping
_BARCHART_INTERVAL_MAP: dict[str, tuple[str, str]] = {
    # DataLoader interval → (type, interval)
    "1m":  ("minutes", "1"),
    "5m":  ("minutes", "5"),
    "15m": ("minutes", "15"),
    "30m": ("minutes", "30"),
    "1h":  ("minutes", "60"),
    "60m": ("minutes", "60"),
    "1d":  ("daily",   "1"),
    "1wk": ("weekly",  "1"),
}

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
    """Load OHLCV bars from yfinance, Barchart, or a local CSV file.

    Notes on yfinance futures data:
    - Continuous front-month contract: ES=F, NQ=F, CL=F etc.
    - Daily bars: typically ~1 year of history available.
    - Hourly (1h/60m): up to ~730 days.
    - 15m and shorter: up to ~60 days only.
    - Data is not back-adjusted at roll dates; gaps may appear.

    Notes on Barchart data:
    - Uses the free cmdty.io / barchart getHistory endpoint (no API key needed).
    - Continuous front-month contracts: ESc1, NQc1, CLc1, etc.
    - Daily history goes back many years; intraday depth varies (~1 year for 1d).
    - Rate-limited; avoid hammering the endpoint in tight loops.
    - For production use, a premium source (Norgate, CQG, Databento) is recommended.
    """

    def __init__(self, contract: ContractSpec):
        self.contract = contract

    def load(
        self,
        start: str | date,
        end: str | date,
        interval: str = "1d",
        local_csv_path: Optional[str | Path] = None,
        source: str = "yfinance",
        barchart_api_key: Optional[str] = None,
    ) -> list[BarData]:
        """Load bars for the given date range.

        Args:
            start: Start date, e.g. "2024-01-01" or datetime.date object.
            end: End date (inclusive).
            interval: Bar interval, e.g. "1d", "1h", "15m", "5m".
            local_csv_path: If provided, load from a CSV file instead of a remote source.
                            Expected columns: datetime, open, high, low, close, volume.
            source: Remote data source when local_csv_path is None.
                    "yfinance" (default) or "barchart".
            barchart_api_key: Optional Barchart API key. When provided the official
                              /v2/history endpoint is used (higher rate limits + full
                              history). When omitted the free cmdtyview endpoint is used.

        Returns:
            List of BarData sorted by timestamp ascending.
        """
        if local_csv_path is not None:
            return self._load_csv(Path(local_csv_path))

        if source == "barchart":
            return self._load_barchart(str(start), str(end), interval, barchart_api_key)

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

    def _load_barchart(
        self,
        start: str,
        end: str,
        interval: str,
        api_key: Optional[str],
    ) -> list[BarData]:
        """Fetch OHLCV bars from Barchart.

        Two modes:
        - **No API key**: scrapes the free cmdtyview/getHistory endpoint used by
          barchart.com own charts. Works for daily bars and light intraday use.
          May break if Barchart changes their internal API.
        - **API key**: uses the official Barchart Market Data API v2 endpoint
          (marketdata.websol.barchart.com/getHistory.json).
          Requires a free or paid account at barchart.com/ondemand.
        """
        try:
            import requests
        except ImportError as e:
            raise ImportError(
                "requests is required for Barchart data. Install with: pip install requests"
            ) from e

        symbol = _BARCHART_SYMBOL_MAP.get(self.contract.symbol)
        if symbol is None:
            raise ValueError(
                f"No Barchart symbol mapping for contract '{self.contract.symbol}'. "
                f"Supported: {', '.join(_BARCHART_SYMBOL_MAP)}"
            )

        if interval not in _BARCHART_INTERVAL_MAP:
            raise ValueError(
                f"Unsupported interval '{interval}' for Barchart. "
                f"Supported: {', '.join(_BARCHART_INTERVAL_MAP)}"
            )
        bar_type, bar_interval = _BARCHART_INTERVAL_MAP[interval]

        if api_key:
            rows = self._barchart_api_v2(symbol, start, end, bar_type, bar_interval, api_key)
        else:
            rows = self._barchart_free(symbol, start, end, bar_type, bar_interval)

        if not rows:
            raise ValueError(
                f"No data returned from Barchart for {symbol} ({start} to {end}, "
                f"interval={interval})."
            )
        return rows

    def _barchart_free(
        self,
        symbol: str,
        start: str,
        end: str,
        bar_type: str,
        bar_interval: str,
    ) -> list[BarData]:
        """Use Barchart free (unauthenticated) getHistory endpoint."""
        import requests

        # This endpoint powers barchart.com's own chart widget — no key required.
        url = "https://www.barchart.com/proxies/timeseries/queryeod.ashx"
        params = {
            "data":      bar_type,
            "symbol":    symbol,
            "startDate": start.replace("-", ""),
            "endDate":   end.replace("-", ""),
            "volume":    "total",
            "order":     "asc",
            "dividends": "false",
            "backadjust": "false",
            "daystoexpiration": "1",
            "contractroll": "combined",
        }
        if bar_type == "minutes":
            params["interval"] = bar_interval

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.barchart.com/",
        }

        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()

        # Response is plain CSV text
        from io import StringIO
        raw = resp.text.strip()
        if not raw or raw.startswith("Error") or raw.startswith("<!"):
            raise ValueError(
                f"Barchart free endpoint returned unexpected response. "
                "Try using an API key or a local CSV file instead."
            )

        df = pd.read_csv(StringIO(raw), header=None)
        # Columns: Symbol, Date, Open, High, Low, Close, Volume (for daily)
        # Intraday adds Time column: Symbol, Date, Time, Open, High, Low, Close, Volume
        df.columns = [c.lower() for c in df.columns.astype(str)]
        return self._parse_barchart_df(df)

    def _barchart_api_v2(
        self,
        symbol: str,
        start: str,
        end: str,
        bar_type: str,
        bar_interval: str,
        api_key: str,
    ) -> list[BarData]:
        """Use the official Barchart Market Data API v2 (requires API key)."""
        import requests

        url = "https://marketdata.websol.barchart.com/getHistory.json"
        params = {
            "apikey":    api_key,
            "symbol":    symbol,
            "type":      bar_type,
            "startDate": start,
            "endDate":   end,
            "order":     "asc",
        }
        if bar_type == "minutes":
            params["interval"] = bar_interval

        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status", {}).get("code") != 200:
            msg = data.get("status", {}).get("message", "unknown error")
            raise ValueError(f"Barchart API error: {msg}")

        results = data.get("results") or []
        bars: list[BarData] = []
        for row in results:
            # tradingDay format: "2024-01-02T09:30:00-05:00" or "2024-01-02"
            raw_ts = row.get("tradingDay") or row.get("timestamp") or ""
            ts = pd.to_datetime(raw_ts, utc=True).tz_localize(None)
            bars.append(BarData(
                timestamp=ts.to_pydatetime(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row.get("volume", 0)),
                contract=self.contract,
            ))
        return bars

    def _parse_barchart_df(self, df: pd.DataFrame) -> list[BarData]:
        """Parse the CSV format returned by the free Barchart endpoint."""
        bars: list[BarData] = []
        cols = list(df.columns)

        # Determine column positions by count
        if len(cols) >= 8:
            # Intraday: symbol, date, time, open, high, low, close, volume
            date_col, time_col = cols[1], cols[2]
            o_col, h_col, l_col, c_col, v_col = cols[3], cols[4], cols[5], cols[6], cols[7]
            df["_dt"] = pd.to_datetime(
                df[date_col].astype(str) + " " + df[time_col].astype(str),
                errors="coerce",
            )
        else:
            # Daily: symbol, date, open, high, low, close, volume
            date_col = cols[1]
            o_col, h_col, l_col, c_col = cols[2], cols[3], cols[4], cols[5]
            v_col = cols[6] if len(cols) > 6 else None
            df["_dt"] = pd.to_datetime(df[date_col].astype(str), errors="coerce")

        df = df.dropna(subset=["_dt"])
        for _, row in df.iterrows():
            bars.append(BarData(
                timestamp=row["_dt"].to_pydatetime(),
                open=float(row[o_col]),
                high=float(row[h_col]),
                low=float(row[l_col]),
                close=float(row[c_col]),
                volume=int(row[v_col]) if v_col and pd.notna(row[v_col]) else 0,
                contract=self.contract,
            ))
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
