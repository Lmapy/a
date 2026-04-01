#!/usr/bin/env python3
"""Download real futures data from Yahoo Finance for backtesting."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.feed import YFinanceDataFeed


def main():
    feed = YFinanceDataFeed()
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)

    instruments = ["ES", "NQ"]
    timeframes = [
        ("5min", "5m intraday"),
        ("1d", "daily"),
    ]

    for instrument in instruments:
        for timeframe, label in timeframes:
            print(f"Downloading {instrument} {label} data...")
            try:
                df = feed.get_bars(instrument, timeframe)
                filename = f"{instrument}_{timeframe}.csv"
                filepath = output_dir / filename
                df.to_csv(filepath)
                print(f"  -> Saved {len(df)} bars to {filepath}")
            except Exception as e:
                print(f"  -> Error: {e}")

    print("\nDone! Data saved to ./data/")


if __name__ == "__main__":
    main()
