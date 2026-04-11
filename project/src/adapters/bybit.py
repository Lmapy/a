from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.adapters.base import ExchangeAdapter
from src.adapters.http_client import CachedHttpClient
from src.utils.time_utils import from_unix_ms, to_unix_ms

LOGGER = logging.getLogger(__name__)


class BybitAdapter(ExchangeAdapter):
    exchange_name = "bybit"
    base_url = "https://api.bybit.com"

    def __init__(self, cache_dir: Path, max_retries: int = 5, backoff_sec: float = 1.5) -> None:
        self.http = CachedHttpClient(cache_dir=cache_dir / "http", max_retries=max_retries, backoff_sec=backoff_sec)

    def _paginate(self, endpoint: str, params: dict[str, str | int]) -> list[dict]:
        cursor = None
        rows: list[dict] = []
        while True:
            q = dict(params)
            if cursor:
                q["cursor"] = cursor
            payload = self.http.get_json(f"{self.base_url}{endpoint}", q)
            result = payload.get("result", {})
            rows.extend(result.get("list", []))
            cursor = result.get("nextPageCursor")
            if not cursor:
                break
        return rows

    def fetch_ohlcv(self, symbol: str, start: datetime, end: datetime, timeframe: str = "1m") -> pd.DataFrame:
        interval_map = {"1m": "1", "5m": "5", "15m": "15", "1h": "60"}
        rows = self._paginate(
            "/v5/market/kline",
            {
                "category": "linear",
                "symbol": symbol,
                "interval": interval_map[timeframe],
                "start": to_unix_ms(start),
                "end": to_unix_ms(end),
                "limit": 1000,
            },
        )
        if not rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        df["timestamp"] = df["timestamp"].astype("int64").map(from_unix_ms)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    def fetch_funding(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        rows = self._paginate(
            "/v5/market/funding/history",
            {
                "category": "linear",
                "symbol": symbol,
                "startTime": to_unix_ms(start),
                "endTime": to_unix_ms(end),
                "limit": 200,
            },
        )
        if not rows:
            return pd.DataFrame(columns=["timestamp", "funding_rate"])
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_numeric(df["fundingRateTimestamp"], errors="coerce").astype("Int64").map(lambda x: from_unix_ms(int(x)))
        df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
        return df[["timestamp", "funding_rate"]]

    def fetch_open_interest(self, symbol: str, start: datetime, end: datetime, timeframe: str = "1m") -> pd.DataFrame:
        rows = self._paginate(
            "/v5/market/open-interest",
            {
                "category": "linear",
                "symbol": symbol,
                "intervalTime": "5min",
                "startTime": to_unix_ms(start),
                "endTime": to_unix_ms(end),
                "limit": 200,
            },
        )
        if not rows:
            LOGGER.warning("Bybit open interest empty for %s", symbol)
            return pd.DataFrame(columns=["timestamp", "open_interest"])
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("Int64").map(lambda x: from_unix_ms(int(x)))
        df["open_interest"] = pd.to_numeric(df["openInterest"], errors="coerce")
        return df[["timestamp", "open_interest"]]

    def fetch_mark_index(self, symbol: str, start: datetime, end: datetime, timeframe: str = "1m") -> pd.DataFrame:
        interval_map = {"1m": "1", "5m": "5", "15m": "15", "1h": "60"}
        mark_rows = self._paginate(
            "/v5/market/mark-price-kline",
            {
                "category": "linear",
                "symbol": symbol,
                "interval": interval_map[timeframe],
                "start": to_unix_ms(start),
                "end": to_unix_ms(end),
                "limit": 1000,
            },
        )
        idx_rows = self._paginate(
            "/v5/market/index-price-kline",
            {
                "category": "linear",
                "symbol": symbol,
                "interval": interval_map[timeframe],
                "start": to_unix_ms(start),
                "end": to_unix_ms(end),
                "limit": 1000,
            },
        )
        mark = pd.DataFrame(mark_rows, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        idx = pd.DataFrame(idx_rows, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        if mark.empty and idx.empty:
            return pd.DataFrame(columns=["timestamp", "mark_price", "index_price"])
        for frame, col in [(mark, "mark_price"), (idx, "index_price")]:
            if not frame.empty:
                frame["timestamp"] = frame["timestamp"].astype("int64").map(from_unix_ms)
                frame[col] = pd.to_numeric(frame["close"], errors="coerce")
        out = pd.DataFrame()
        if not mark.empty:
            out = mark[["timestamp", "mark_price"]]
        if not idx.empty:
            out = out.merge(idx[["timestamp", "index_price"]], on="timestamp", how="outer") if not out.empty else idx[["timestamp", "index_price"]]
        return out
