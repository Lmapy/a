from __future__ import annotations

from pathlib import Path

from src.adapters.base import ExchangeAdapter
from src.adapters.bybit import BybitAdapter
from src.adapters.hyperliquid import HyperliquidAdapter


def build_adapter(exchange: str, cache_root: Path, max_retries: int, backoff_sec: float) -> ExchangeAdapter:
    if exchange == "bybit":
        return BybitAdapter(cache_dir=cache_root / "bybit", max_retries=max_retries, backoff_sec=backoff_sec)
    if exchange == "hyperliquid":
        return HyperliquidAdapter(cache_dir=cache_root / "hyperliquid", max_retries=max_retries, backoff_sec=backoff_sec)
    raise ValueError(f"unsupported exchange: {exchange}")
