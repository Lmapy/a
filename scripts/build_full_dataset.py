#!/usr/bin/env python3
"""Build full multi-year 5-minute datasets for ES and NQ from real 1-minute data.

Data sources (from FutureSharks/financial-data repo):
  - Oanda SPX500_USD: 2005-2020 (S&P 500 CFD, tracks ES futures)
  - Oanda NAS100_USD: 2005-2020 (NASDAQ 100 CFD, tracks NQ futures)
  - Histdata SPXUSD: 2010-2018 (S&P 500 index, 1-min bars)

Outputs 5-minute OHLCV CSVs filtered to US RTH (9:30-16:00 ET).
"""

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = "/tmp/financial-data/pyfinancialdata/data"


def load_oanda_instrument(instrument: str, years: range | None = None) -> pd.DataFrame:
    """Load all monthly Oanda CSVs for an instrument."""
    base = f"{DATA_ROOT}/currencies/oanda/{instrument}"
    if not os.path.exists(base):
        raise FileNotFoundError(f"No data at {base}")

    all_files = sorted(glob.glob(f"{base}/*/*.csv"))
    if years:
        all_files = [f for f in all_files if any(f"/{y}/" in f for y in years)]

    print(f"  Loading {len(all_files)} monthly files for {instrument}...")
    frames = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            frames.append(df)
        except Exception as e:
            print(f"  Warning: skipped {f}: {e}")

    raw = pd.concat(frames, ignore_index=True)
    raw["time"] = pd.to_datetime(raw["time"])
    raw = raw.set_index("time").sort_index()
    raw = raw[["open", "high", "low", "close", "volume"]].copy()
    return raw


def load_histdata_spx(years: range | None = None) -> pd.DataFrame:
    """Load histdata S&P 500 1-minute CSVs."""
    base = f"{DATA_ROOT}/stocks/histdata/SPXUSD"
    files = sorted(glob.glob(f"{base}/*.csv"))
    if years:
        files = [f for f in files if any(str(y) in f for y in years)]

    print(f"  Loading {len(files)} yearly files from histdata...")
    frames = []
    for f in files:
        df = pd.read_csv(f, sep=";", header=None,
                         names=["datetime", "open", "high", "low", "close", "volume"])
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)
    raw["datetime"] = pd.to_datetime(raw["datetime"], format="%Y%m%d %H%M%S")
    raw = raw.set_index("datetime").sort_index()
    raw = raw[["open", "high", "low", "close", "volume"]].copy()
    return raw


def resample_to_5min(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-min bars to 5-min bars."""
    bars = df.resample("5min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    return bars


def filter_rth(df: pd.DataFrame, source: str = "oanda") -> pd.DataFrame:
    """Filter to regular trading hours (9:30 - 16:00 ET).

    Oanda data is in UTC — volume analysis confirms peaks at 13:00-15:00 UTC
    and 19:00 UTC (= 9-11 AM ET and 3 PM ET). So RTH = 13:30-20:00 UTC.

    Histdata data appears to already be in ET.
    """
    if source == "oanda":
        # Convert UTC to US/Eastern, filter RTH, then drop timezone
        df = df.copy()
        df.index = df.index.tz_localize("UTC").tz_convert("US/Eastern")
        df = df.between_time("09:30", "16:00")
        df.index = df.index.tz_localize(None)
    else:
        # Histdata is already in ET
        df = df.between_time("09:30", "16:00")
    return df


def generate_volume_if_zero(df: pd.DataFrame) -> pd.DataFrame:
    """If volume is all zeros, generate realistic volume from price action."""
    if (df["volume"] == 0).mean() > 0.8:  # >80% zero volume
        bar_range = df["high"] - df["low"]
        median_range = bar_range.median()
        if median_range > 0:
            base_vol = (bar_range / median_range * 20000).clip(1000)
        else:
            base_vol = pd.Series(20000, index=df.index)

        # Add U-shape volume profile per day (fully vectorized)
        result = df.copy()
        day_key = df.index.date
        day_series = pd.Series(day_key, index=df.index)
        bar_idx = day_series.groupby(day_key).cumcount()
        day_size = day_series.groupby(day_key).transform("size")
        t = bar_idx / (day_size - 1).clip(lower=1)
        u_shape = np.clip(2.0 - 3.0 * t.values * (1 - t.values), 0.5, None)
        result["volume"] = (base_vol.values * u_shape).astype(int)
        return result
    return df


def build_dataset(name: str, raw_1min: pd.DataFrame, output_path: Path,
                  source: str = "oanda") -> dict:
    """Resample, filter, fix volume, and save."""
    print(f"  Raw 1-min bars: {len(raw_1min):,}")
    print(f"  Date range: {raw_1min.index[0]} to {raw_1min.index[-1]}")

    bars_5m = resample_to_5min(raw_1min)
    print(f"  5-min bars (all hours): {len(bars_5m):,}")

    bars_rth = filter_rth(bars_5m, source=source)
    print(f"  5-min bars (RTH only): {len(bars_rth):,}")

    bars_rth = generate_volume_if_zero(bars_rth)

    # Remove any days with fewer than 10 bars (likely holidays/half days)
    daily_counts = bars_rth.groupby(bars_rth.index.date).size()
    valid_days = daily_counts[daily_counts >= 10].index
    valid_set = set(valid_days)
    bars_rth = bars_rth[[d in valid_set for d in bars_rth.index.date]]

    n_days = len(set(bars_rth.index.date))
    print(f"  Trading days: {n_days}")
    print(f"  Price range: {bars_rth['close'].min():.2f} - {bars_rth['close'].max():.2f}")

    bars_rth.to_csv(output_path)
    print(f"  Saved to {output_path}\n")

    return {
        "name": name,
        "bars": len(bars_rth),
        "days": n_days,
        "date_range": f"{bars_rth.index[0]} to {bars_rth.index[-1]}",
        "price_range": f"{bars_rth['close'].min():.2f} - {bars_rth['close'].max():.2f}",
    }


def main():
    output_dir = Path("/home/user/a/data")
    output_dir.mkdir(exist_ok=True)

    summaries = []

    # 1. ES (S&P 500) from Oanda — 2005-2020
    print("=" * 60)
    print("BUILDING ES (S&P 500) DATASET — Oanda 2005-2020")
    print("=" * 60)
    es_oanda = load_oanda_instrument("SPX500_USD")
    s = build_dataset("ES_oanda", es_oanda, output_dir / "ES_5min.csv", source="oanda")
    summaries.append(s)

    # 2. NQ (NASDAQ 100) from Oanda — 2005-2020
    print("=" * 60)
    print("BUILDING NQ (NASDAQ 100) DATASET — Oanda 2005-2020")
    print("=" * 60)
    nq_oanda = load_oanda_instrument("NAS100_USD")
    s = build_dataset("NQ_oanda", nq_oanda, output_dir / "NQ_5min.csv", source="oanda")
    summaries.append(s)

    # 3. GC (Gold) from Oanda — 2006-2020
    print("=" * 60)
    print("BUILDING GC (Gold) DATASET — Oanda 2006-2020")
    print("=" * 60)
    gc_oanda = load_oanda_instrument("XAU_USD")
    s = build_dataset("GC_oanda", gc_oanda, output_dir / "GC_5min.csv", source="oanda")
    summaries.append(s)

    # 4. ES from histdata (higher quality) — 2010-2018
    print("=" * 60)
    print("BUILDING ES (S&P 500) DATASET — Histdata 2010-2018")
    print("=" * 60)
    es_hist = load_histdata_spx()
    s = build_dataset("ES_histdata", es_hist, output_dir / "ES_histdata_5min.csv", source="histdata")
    summaries.append(s)

    # Summary
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    for s in summaries:
        print(f"  {s['name']}: {s['bars']:,} bars, {s['days']} days, {s['date_range']}")
        print(f"    Prices: {s['price_range']}")

    total_bars = sum(s["bars"] for s in summaries)
    print(f"\n  TOTAL: {total_bars:,} 5-minute bars across all datasets")


if __name__ == "__main__":
    main()
