"""Download real gold (XAUUSD) market data from multiple sources.

Primary: ejtraderLabs GitHub repo - real 15-min XAUUSD data (2012-2022)
Secondary: Dukascopy XAUUSD tick data via tick-vault
Tertiary: yfinance GC=F
"""

import asyncio
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from config import DATA_DIR


GITHUB_BASE = "https://raw.githubusercontent.com/ejtraderLabs/historical-data/main/XAUUSD"
GITHUB_FILES = {
    "15min": "XAUUSDm15.csv",
    "1H": "XAUUSDh1.csv",
    "4H": "XAUUSDh4.csv",
    "1D": "XAUUSDd1.csv",
}


def _ensure_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_github_data(timeframe: str = "15min") -> pd.DataFrame:
    """Download real XAUUSD data from ejtraderLabs GitHub repository.

    This is real historical market data (not synthetic):
    - Source: MetaTrader broker data compiled by ejtraderLabs
    - Coverage: 2012-05-15 to 2022-03-04
    - Columns: Date, open, high, low, close, tick_volume
    - Prices are in cents (divide by 100 for dollars)
    """
    _ensure_dir()

    fname = GITHUB_FILES.get(timeframe)
    if fname is None:
        raise ValueError(f"Unknown timeframe: {timeframe}. Available: {list(GITHUB_FILES.keys())}")

    cache_csv = DATA_DIR / f"XAUUSD_{timeframe}.csv"
    cache_parquet = DATA_DIR / f"XAUUSD_{timeframe}.parquet"

    # Use parquet cache if available
    if cache_parquet.exists():
        print(f"[DATA] Loading cached {timeframe} data from {cache_parquet}")
        return pd.read_parquet(cache_parquet)

    # Download CSV if not cached
    if not cache_csv.exists():
        url = f"{GITHUB_BASE}/{fname}"
        print(f"[DATA] Downloading real XAUUSD {timeframe} data from GitHub...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=120)
        data = resp.read()
        cache_csv.write_bytes(data)
        print(f"[DATA] Saved {len(data):,} bytes to {cache_csv}")
    else:
        print(f"[DATA] Loading cached CSV from {cache_csv}")

    # Parse CSV
    df = pd.read_csv(cache_csv, parse_dates=["Date"])
    df = df.set_index("Date")
    df.index = pd.to_datetime(df.index, utc=True)

    # Convert from cents to dollars
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col] / 100.0

    df = df.rename(columns={"tick_volume": "volume"})

    # Save as parquet for fast reload
    df.to_parquet(cache_parquet)
    print(f"[DATA] Processed {len(df):,} candles ({df.index.min()} to {df.index.max()})")

    return df


async def download_dukascopy(
    symbol: str = "XAUUSD",
    start: str = "2024-01-01",
    end: str = "2025-12-31",
) -> pd.DataFrame:
    """Download tick data from Dukascopy (requires network access to Dukascopy servers)."""
    from tick_vault import download_range, read_tick_data

    _ensure_dir()
    cache_path = DATA_DIR / f"{symbol}_ticks.parquet"

    if cache_path.exists():
        print(f"[DATA] Loading cached tick data from {cache_path}")
        return pd.read_parquet(cache_path)

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    print(f"[DATA] Downloading {symbol} tick data from {start} to {end}...")
    await download_range(symbol=symbol, start=start_dt, end=end_dt)

    print("[DATA] Reading downloaded tick data...")
    df = read_tick_data(symbol=symbol, start=start_dt, end=end_dt, strict=False)

    if df.empty:
        raise RuntimeError(f"No tick data downloaded for {symbol}")

    df.to_parquet(cache_path)
    print(f"[DATA] Saved {len(df):,} ticks to {cache_path}")
    return df


def load_data() -> pd.DataFrame:
    """Load XAUUSD data, using the best available source.

    Returns OHLCV DataFrame with dollar prices.
    """
    _ensure_dir()

    # Check for Dukascopy tick data first
    tick_cache = DATA_DIR / "XAUUSD_ticks.parquet"
    if tick_cache.exists():
        print("[DATA] Using cached Dukascopy tick data")
        return pd.read_parquet(tick_cache)

    # Use GitHub real data (15-min candles)
    return download_github_data("15min")


def load_all_timeframes() -> dict[str, pd.DataFrame]:
    """Load all available timeframes from GitHub data."""
    result = {}
    for tf in ["15min", "1H", "4H"]:
        try:
            result[tf] = download_github_data(tf)
        except Exception as e:
            print(f"[DATA] Warning: could not load {tf}: {e}")
    return result


if __name__ == "__main__":
    data = load_data()
    print(f"\nData shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print(f"Price range: ${data['close'].min():.2f} to ${data['close'].max():.2f}")
    print(data.head())
    print(data.tail())
