"""Resample tick data to OHLCV candles at various timeframes with session tagging."""

import pandas as pd
import numpy as np
from pathlib import Path

from config import DATA_DIR, SESSIONS


def tick_to_ohlcv(ticks: pd.DataFrame, timeframe: str = "5min") -> pd.DataFrame:
    """Resample tick data (with 'time', 'ask', 'bid' columns) to OHLCV candles.

    Uses mid-price = (ask + bid) / 2.
    """
    df = ticks.copy()
    df["mid"] = (df["ask"] + df["bid"]) / 2.0

    if "time" in df.columns:
        df = df.set_index("time")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Total volume per tick
    vol_col = None
    if "ask_volume" in df.columns and "bid_volume" in df.columns:
        df["volume"] = df["ask_volume"] + df["bid_volume"]
        vol_col = "volume"
    elif "volume" in df.columns:
        vol_col = "volume"

    agg = {
        "mid": ["first", "max", "min", "last"],
    }
    if vol_col:
        agg[vol_col] = "sum"

    ohlcv = df.resample(timeframe).agg(agg)
    ohlcv.columns = ["open", "high", "low", "close"] + (["volume"] if vol_col else [])
    ohlcv = ohlcv.dropna(subset=["open"])

    if vol_col is None:
        ohlcv["volume"] = 0

    return ohlcv


def tag_sessions(df: pd.DataFrame, tz: str = "US/Central") -> pd.DataFrame:
    """Add a 'session' column: 'asian', 'london', 'ny', or 'closed'."""
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    ct = idx.tz_convert(tz)
    hours = ct.hour

    session = pd.Series("closed", index=df.index)

    # Asian: 18:00 - 02:00 CT (crosses midnight)
    asian_mask = (hours >= SESSIONS.asian_start) | (hours < SESSIONS.asian_end)
    session[asian_mask] = "asian"

    # London: 02:00 - 08:00 CT
    london_mask = (hours >= SESSIONS.london_start) & (hours < SESSIONS.london_end)
    session[london_mask] = "london"

    # NY: 08:00 - 17:00 CT
    ny_mask = (hours >= SESSIONS.ny_start) & (hours < SESSIONS.ny_end)
    session[ny_mask] = "ny"

    df["session"] = session.values
    return df


def filter_trading_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only candles during futures trading hours (filter out 5-6PM CT break)."""
    if df.index.tz is None:
        idx_ct = df.index.tz_localize("UTC").tz_convert("US/Central")
    else:
        idx_ct = df.index.tz_convert("US/Central")

    hours = idx_ct.hour
    # Remove the 1-hour daily break: 5 PM CT (hour 17)
    mask = hours != 17
    return df[mask]


def prepare_candles(ticks: pd.DataFrame, timeframe: str = "5min") -> pd.DataFrame:
    """Full pipeline: tick -> OHLCV -> filter hours -> tag sessions."""
    candles = tick_to_ohlcv(ticks, timeframe)
    candles = filter_trading_hours(candles)
    candles = tag_sessions(candles)
    return candles


def build_multi_timeframe(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build candles for all needed timeframes.

    Accepts either tick data (with ask/bid columns) or pre-built OHLCV data.
    """
    is_ohlcv = (
        data.attrs.get("_is_ohlcv", False)
        or all(c in data.columns for c in ["open", "high", "low", "close"])
    )

    if is_ohlcv:
        return _build_from_ohlcv(data)
    else:
        return _build_from_ticks(data)


def _build_from_ticks(ticks: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build candles from tick data."""
    timeframes = {"1min": "1min", "5min": "5min", "15min": "15min", "1H": "1h", "4H": "4h"}
    result = {}
    for label, tf in timeframes.items():
        df = prepare_candles(ticks, tf)
        result[label] = df
        print(f"[RESAMPLE] {label}: {len(df):,} candles")
    return result


def _build_from_ohlcv(candles: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build higher timeframe candles from existing OHLCV data."""
    if not isinstance(candles.index, pd.DatetimeIndex):
        raise ValueError("OHLCV data must have DatetimeIndex")

    # Detect input timeframe from index frequency
    if len(candles) > 1:
        median_diff = pd.Series(candles.index).diff().median()
        input_minutes = median_diff.total_seconds() / 60
    else:
        input_minutes = 5

    print(f"[RESAMPLE] Input data interval: ~{input_minutes:.0f} min")

    # Only build timeframes >= input resolution
    all_tfs = {"5min": "5min", "15min": "15min", "1H": "1h", "4H": "4h"}
    tf_minutes = {"5min": 5, "15min": 15, "1H": 60, "4H": 240}
    result = {}

    for label, tf in all_tfs.items():
        if tf_minutes[label] < input_minutes:
            continue

        if tf_minutes[label] == input_minutes:
            df = candles.copy()
        else:
            df = candles.resample(tf).agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna(subset=["open"])

        df = filter_trading_hours(df)
        df = tag_sessions(df)
        result[label] = df
        print(f"[RESAMPLE] {label}: {len(df):,} candles")

    return result


def save_candles(candles_dict: dict[str, pd.DataFrame]):
    """Save resampled candles to parquet."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for label, df in candles_dict.items():
        path = DATA_DIR / f"candles_{label}.parquet"
        df.to_parquet(path)
        print(f"[SAVE] {path}")


def load_candles(timeframe: str = "5min") -> pd.DataFrame:
    """Load cached candles."""
    path = DATA_DIR / f"candles_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No cached candles at {path}. Run data pipeline first.")
    return pd.read_parquet(path)
