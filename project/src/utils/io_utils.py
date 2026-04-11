from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def stable_cache_key(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)


def load_parquet_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def dedupe_sort(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    return df.sort_values(key_cols).drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)


def verify_continuity(df: pd.DataFrame, timeframe: str, ts_col: str = "timestamp") -> list[pd.Timestamp]:
    freq_map = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h"}
    freq = freq_map.get(timeframe)
    if freq is None or df.empty:
        return []
    ts = pd.to_datetime(df[ts_col], utc=True).sort_values()
    full = pd.date_range(ts.iloc[0], ts.iloc[-1], freq=freq, tz="UTC")
    missing = full.difference(ts)
    if len(missing):
        LOGGER.warning("Detected %s missing candles for timeframe=%s", len(missing), timeframe)
    return list(missing)
