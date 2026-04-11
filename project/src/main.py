from __future__ import annotations

import argparse
from datetime import datetime, timezone

from src.adapters.pipeline import download_exchange_data
from src.backtest.pipeline import run_portfolio_backtest, run_strategy_backtest
from src.features.engineer import run_feature_pipeline
from src.utils.config import load_config
from src.utils.logging_utils import configure_logging


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def cli() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Crypto research pipeline")
    parser.add_argument("--config", default="config/settings.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download")
    d.add_argument("--exchange", required=True, choices=["bybit", "hyperliquid"])
    d.add_argument("--symbols", nargs="+", required=True)
    d.add_argument("--start", required=True)
    d.add_argument("--end", required=True)

    sub.add_parser("features")

    b = sub.add_parser("backtest")
    b.add_argument("--strategy", required=True, choices=["breakout", "sweep", "vwap_reversion"])

    sub.add_parser("portfolio")

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.command == "download":
        download_exchange_data(cfg, args.exchange, args.symbols, _dt(args.start), _dt(args.end))
    elif args.command == "features":
        run_feature_pipeline(cfg)
    elif args.command == "backtest":
        run_strategy_backtest(cfg, args.strategy)
    elif args.command == "portfolio":
        run_portfolio_backtest(cfg)


if __name__ == "__main__":
    cli()
