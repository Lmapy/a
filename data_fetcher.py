"""
Gold (XAU/USD) 1-minute candle data fetcher.
Uses yfinance for GC=F (gold futures) as proxy for XAU/USD.
yfinance limits 1m data to ~7 days per request, so we chunk requests.
For longer history, we fall back to hourly/daily and resample where needed.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

GOLD_TICKER = "GC=F"


def fetch_gold_1m_recent(days_back=7):
    """Fetch most recent 1m data (yfinance limit ~7 days)."""
    ticker = yf.Ticker(GOLD_TICKER)
    df = ticker.history(period=f"{days_back}d", interval="1m")
    return df


def fetch_gold_intraday_chunks(total_days=1825):
    """
    Fetch 1m gold data in 7-day chunks going back total_days.
    yfinance only allows ~60 days of 1m data total, and 7 days per request.
    Beyond 60 days, we use 2m/5m/15m/1h intervals.
    """
    all_data = []

    # 1m data: last 7 days (yfinance hard limit for free tier is ~7 days for 1m)
    print("Fetching 1m data (last 7 days)...")
    try:
        df_1m = yf.download(GOLD_TICKER, period="7d", interval="1m", progress=False)
        if len(df_1m) > 0:
            df_1m.columns = [c[0] if isinstance(c, tuple) else c for c in df_1m.columns]
            all_data.append(("1m_recent", df_1m))
            print(f"  Got {len(df_1m)} 1m bars")
    except Exception as e:
        print(f"  1m fetch failed: {e}")

    # 5m data: last 60 days
    print("Fetching 5m data (last 60 days)...")
    try:
        df_5m = yf.download(GOLD_TICKER, period="60d", interval="5m", progress=False)
        if len(df_5m) > 0:
            df_5m.columns = [c[0] if isinstance(c, tuple) else c for c in df_5m.columns]
            all_data.append(("5m_60d", df_5m))
            print(f"  Got {len(df_5m)} 5m bars")
    except Exception as e:
        print(f"  5m fetch failed: {e}")

    # 15m data: last 60 days (backup)
    print("Fetching 15m data (last 60 days)...")
    try:
        df_15m = yf.download(GOLD_TICKER, period="60d", interval="15m", progress=False)
        if len(df_15m) > 0:
            df_15m.columns = [c[0] if isinstance(c, tuple) else c for c in df_15m.columns]
            all_data.append(("15m_60d", df_15m))
            print(f"  Got {len(df_15m)} 15m bars")
    except Exception as e:
        print(f"  15m fetch failed: {e}")

    # 1h data: last 2 years
    print("Fetching 1h data (last 730 days)...")
    try:
        df_1h = yf.download(GOLD_TICKER, period="730d", interval="1h", progress=False)
        if len(df_1h) > 0:
            df_1h.columns = [c[0] if isinstance(c, tuple) else c for c in df_1h.columns]
            all_data.append(("1h_2y", df_1h))
            print(f"  Got {len(df_1h)} 1h bars")
    except Exception as e:
        print(f"  1h fetch failed: {e}")

    # Daily data: full 5 years
    print("Fetching daily data (5 years)...")
    try:
        df_daily = yf.download(GOLD_TICKER, period="5y", interval="1d", progress=False)
        if len(df_daily) > 0:
            df_daily.columns = [c[0] if isinstance(c, tuple) else c for c in df_daily.columns]
            all_data.append(("1d_5y", df_daily))
            print(f"  Got {len(df_daily)} daily bars")
    except Exception as e:
        print(f"  Daily fetch failed: {e}")

    return all_data


def save_all_data(all_data):
    """Save all fetched data to CSV files."""
    for name, df in all_data:
        path = os.path.join(DATA_DIR, f"gold_{name}.csv")
        df.to_csv(path)
        print(f"Saved {name}: {len(df)} rows -> {path}")


def load_data(name):
    """Load a saved dataset."""
    path = os.path.join(DATA_DIR, f"gold_{name}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df
    return None


def get_best_available_data():
    """
    Load the best available data for backtesting.
    Prefers higher frequency data, falls back to lower.
    Returns a dict of all available timeframes.
    """
    datasets = {}
    for name in ["1m_recent", "5m_60d", "15m_60d", "1h_2y", "1d_5y"]:
        df = load_data(name)
        if df is not None and len(df) > 0:
            datasets[name] = df
    return datasets


if __name__ == "__main__":
    print("=" * 60)
    print("GOLD DATA FETCHER - XAU/USD (GC=F)")
    print("=" * 60)

    all_data = fetch_gold_intraday_chunks()
    save_all_data(all_data)

    print("\n--- Summary ---")
    for name, df in all_data:
        print(f"{name}: {len(df)} bars, {df.index[0]} to {df.index[-1]}")
