#!/usr/bin/env python3
"""Run Monte Carlo simulation to estimate challenge pass rate."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.engine import BacktestEngine
from src.backtest.monte_carlo import MonteCarloSimulator
from src.core.orchestrator import TradingOrchestrator
from src.data.feed import CSVDataFeed, YFinanceDataFeed


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo pass rate estimation")
    parser.add_argument("--instrument", default="ES", help="Futures instrument")
    parser.add_argument("--timeframe", default="5min", help="Bar timeframe")
    parser.add_argument("--data-dir", default=None, help="Directory with CSV data files")
    parser.add_argument("--config-dir", default="config", help="Config directory")
    parser.add_argument("--n-trials", type=int, default=10000, help="Number of MC trials")
    parser.add_argument("--max-days", type=int, default=60, help="Max days per trial")
    parser.add_argument("--block-size", type=int, default=3, help="Block bootstrap size")
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
    print()

    # Step 1: Run backtest to get daily P&Ls
    print("Running backtest to generate daily P&L distribution...")
    orchestrator = TradingOrchestrator(args.config_dir)
    engine = BacktestEngine(orchestrator.config, orchestrator.strategies)
    bt_result = engine.run(data, args.instrument)

    daily_pnls = bt_result.daily_pnls
    if not daily_pnls or len(daily_pnls) < 5:
        print(f"Not enough daily P&L data ({len(daily_pnls)} days). Need at least 5.")
        print("Try using more historical data or a different instrument.")
        sys.exit(1)

    print(f"Generated {len(daily_pnls)} daily P&Ls from backtest")
    print(f"  Mean daily P&L: ${sum(daily_pnls)/len(daily_pnls):+,.2f}")
    print(f"  Total P&L: ${sum(daily_pnls):+,.2f}")
    print()

    # Step 2: Run Monte Carlo
    mc = MonteCarloSimulator(
        profit_target=orchestrator.config.firm.profit_target,
        drawdown_amount=orchestrator.config.firm.trailing_drawdown.initial_amount,
        n_trials=args.n_trials,
        max_days=args.max_days,
        block_size=args.block_size,
    )
    result = mc.run(daily_pnls)
    print(MonteCarloSimulator.format_result(result))


if __name__ == "__main__":
    main()
