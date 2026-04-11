from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.adapters.base import ExchangeAdapter
from src.adapters.http_client import CachedHttpClient
from src.utils.time_utils import from_unix_ms, to_unix_ms

LOGGER = logging.getLogger(__name__)


class HyperliquidAdapter(ExchangeAdapter):
    exchange_name = "hyperliquid"
    info_url = "https://api.hyperliquid.xyz/info"

    def __init__(self, cache_dir: Path, max_retries: int = 5, backoff_sec: float = 1.5) -> None:
        self.http = CachedHttpClient(cache_dir=cache_dir / "http", max_retries=max_retries, backoff_sec=backoff_sec)

    def _post_json(self, payload: dict) -> dict:
        import requests

        for i in range(self.http.max_retries):
            try:
                r = requests.post(self.info_url, json=payload, timeout=30)
                if r.status_code >= 500:
                    raise requests.HTTPError(f"server err {r.status_code}")
                r.raise_for_status()
                return r.json()
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError):
                LOGGER.warning("Retry %s for hyperliquid payload=%s", i + 1, payload)
        raise RuntimeError("hyperliquid request failed")

    def fetch_ohlcv(self, symbol: str, start: datetime, end: datetime, timeframe: str = "1m") -> pd.DataFrame:
        interval_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h"}
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": symbol,
                "interval": interval_map[timeframe],
                "startTime": to_unix_ms(start),
                "endTime": to_unix_ms(end),
            },
        }
        rows = self._post_json(payload)
        if not isinstance(rows, list) or not rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_numeric(df["t"], errors="coerce").astype("Int64").map(lambda x: from_unix_ms(int(x)))
        df["open"] = pd.to_numeric(df["o"], errors="coerce")
        df["high"] = pd.to_numeric(df["h"], errors="coerce")
        df["low"] = pd.to_numeric(df["l"], errors="coerce")
        df["close"] = pd.to_numeric(df["c"], errors="coerce")
        df["volume"] = pd.to_numeric(df["v"], errors="coerce")
        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    def fetch_funding(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        payload = {"type": "fundingHistory", "coin": symbol, "startTime": to_unix_ms(start), "endTime": to_unix_ms(end)}
        rows = self._post_json(payload)
        if not isinstance(rows, list) or not rows:
            LOGGER.warning("Hyperliquid funding empty for %s", symbol)
            return pd.DataFrame(columns=["timestamp", "funding_rate"])
        df = pd.DataFrame(rows)
        ts_col = "time" if "time" in df.columns else "timestamp"
        rate_col = "fundingRate" if "fundingRate" in df.columns else "funding"
        df["timestamp"] = pd.to_numeric(df[ts_col], errors="coerce").astype("Int64").map(lambda x: from_unix_ms(int(x)))
        df["funding_rate"] = pd.to_numeric(df[rate_col], errors="coerce")
        return df[["timestamp", "funding_rate"]]

    def fetch_open_interest(self, symbol: str, start: datetime, end: datetime, timeframe: str = "1m") -> pd.DataFrame:
        LOGGER.warning("Hyperliquid public open interest history not wired to stable endpoint; storing nulls for %s", symbol)
        return pd.DataFrame(columns=["timestamp", "open_interest"])

    def fetch_mark_index(self, symbol: str, start: datetime, end: datetime, timeframe: str = "1m") -> pd.DataFrame:
        return pd.DataFrame(columns=["timestamp", "mark_price", "index_price"])
