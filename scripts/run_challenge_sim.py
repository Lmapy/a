#!/usr/bin/env python3
"""Run multiple challenge simulations to estimate pass rate empirically."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.challenge_sim import ChallengeSim
from src.core.orchestrator import TradingOrchestrator
from src.data.feed import CSVDataFeed, YFinanceDataFeed


def main():
    parser = argparse.ArgumentParser(description="Run prop firm challenge simulations")
    parser.add_argument("--instrument", default="ES", help="Futures instrument")
    parser.add_argument("--timeframe", default="5min", help="Bar timeframe")
    parser.add_argument("--data-dir", default=None, help="Directory with CSV data files")
    parser.add_argument("--config-dir", default="config", help="Config directory")
    parser.add_argument("--n-runs", type=int, default=5, help="Number of challenge attempts to simulate")
    parser.add_argument("--window-days", type=int, default=30, help="Trading days per challenge window")
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
        print(f"Downloading {args.instrument} {args.timeframe} data...")
        feed = YFinanceDataFeed()

    try:
        data = feed.get_bars(args.instrument, args.timeframe)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    print(f"Loaded {len(data)} bars for {args.instrument}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print()

    # Run challenge simulations
    orchestrator = TradingOrchestrator(args.config_dir)
    sim = ChallengeSim(orchestrator.config, orchestrator.strategies)
    results = sim.run_multiple(data, args.instrument, n_runs=args.n_runs, window_days=args.window_days)

    # Print summary
    print(ChallengeSim.summarize(results))

    # Print per-attempt details
    for i, result in enumerate(results):
        status = "PASSED" if result.passed else "FAILED"
        print(f"  Attempt {i+1}: {status} in {result.days_taken} days, P&L=${result.profit:+,.2f}")


if __name__ == "__main__":
    main()
