#!/usr/bin/env python3
"""Run a backtest on real or CSV futures data."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.orchestrator import TradingOrchestrator
from src.data.feed import CSVDataFeed, YFinanceDataFeed


def main():
    parser = argparse.ArgumentParser(description="Run prop firm challenge backtest")
    parser.add_argument("--instrument", default="ES", help="Futures instrument (ES, NQ, MES, MNQ)")
    parser.add_argument("--timeframe", default="5min", help="Bar timeframe")
    parser.add_argument("--data-dir", default=None, help="Directory with CSV data files")
    parser.add_argument("--config-dir", default="config", help="Config directory")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load data
    if args.data_dir:
        feed = CSVDataFeed(args.data_dir)
    else:
        print(f"Downloading {args.instrument} {args.timeframe} data from Yahoo Finance...")
        feed = YFinanceDataFeed()

    try:
        data = feed.get_bars(args.instrument, args.timeframe)
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Try running: python scripts/download_data.py first")
        sys.exit(1)

    print(f"Loaded {len(data)} bars for {args.instrument} ({args.timeframe})")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print()

    # Run full analysis
    orchestrator = TradingOrchestrator(args.config_dir)
    orchestrator.run_full_analysis(data, args.instrument, args.output_dir)


if __name__ == "__main__":
    main()
