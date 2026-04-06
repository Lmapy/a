#!/usr/bin/env python3
"""Entry point for the Hyperliquid Multi-Asset Trading Bot.

Usage:
    python main.py              # Run the bot in live mode
    python main.py --dry-run    # Analyze markets without trading
    python main.py --once       # Run a single rebalance cycle
"""

import argparse
import sys

from hyperliquid_bot.bot import HyperliquidBot
from hyperliquid_bot.config import BotConfig


def main():
    parser = argparse.ArgumentParser(description="Hyperliquid Multi-Asset Trading Bot")
    parser.add_argument("--dry-run", action="store_true", help="Analyze markets without executing trades")
    parser.add_argument("--once", action="store_true", help="Run a single rebalance cycle then exit")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    config = BotConfig()
    config.log_level = args.log_level

    try:
        bot = HyperliquidBot(config)

        if args.dry_run:
            bot.dry_run()
        elif args.once:
            bot.run_once()
        else:
            bot.run()
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        print("Check your .env file. See .env.example for required variables.", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutdown requested.")
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
