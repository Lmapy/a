"""Main bot orchestrator: initialization, scheduling, and lifecycle."""

import logging
import signal
import sys
import time
from typing import Any

import eth_account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

from .config import BotConfig
from .market_data import CoinGlassData, HyperliquidData
from .portfolio import PortfolioManager
from .risk_manager import RiskManager
from .strategy import StrategyEngine

logger = logging.getLogger(__name__)


class HyperliquidBot:
    """Main trading bot that orchestrates all components."""

    def __init__(self, config: BotConfig):
        self.config = config
        self.running = False
        self._setup_logging()
        self._validate_config()
        self._init_components()

    def _setup_logging(self):
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("bot.log", mode="a"),
            ],
        )

    def _validate_config(self):
        errors = self.config.validate()
        if errors:
            for err in errors:
                logger.error(f"Config error: {err}")
            raise ValueError(f"Configuration errors: {'; '.join(errors)}")

    def _init_components(self):
        """Initialize all bot components."""
        logger.info("Initializing bot components...")

        # Determine API URL
        base_url = self.config.hyperliquid.base_url
        is_mainnet = self.config.hyperliquid.mainnet
        network = "MAINNET" if is_mainnet else "TESTNET"
        logger.info(f"Connecting to Hyperliquid {network}: {base_url}")

        # Create wallet from private key
        self.account = eth_account.Account.from_key(self.config.hyperliquid.private_key)
        logger.info(f"Wallet address: {self.account.address}")

        # Verify wallet matches configured address
        configured_addr = self.config.hyperliquid.wallet_address.lower()
        if configured_addr and self.account.address.lower() != configured_addr:
            raise ValueError(
                f"Private key address {self.account.address} doesn't match "
                f"configured address {self.config.hyperliquid.wallet_address}"
            )

        # Initialize Hyperliquid SDK components
        self.info = Info(base_url, skip_ws=True)
        self.exchange = Exchange(
            self.account,
            base_url,
            vault_address=self.config.hyperliquid.vault_address,
        )

        # Initialize bot modules
        self.market_data = HyperliquidData(self.config)
        self.coinglass = CoinGlassData(self.config)
        self.risk_manager = RiskManager(self.config.risk)
        self.strategy = StrategyEngine(self.config.strategy, self.market_data, self.coinglass)
        self.portfolio = PortfolioManager(
            self.config, self.exchange, self.market_data, self.strategy, self.risk_manager
        )

        logger.info("All components initialized")
        if self.coinglass.enabled:
            logger.info("CoinGlass liquidity data: ENABLED")
        else:
            logger.info("CoinGlass liquidity data: DISABLED (no API key)")

    def _print_startup_info(self):
        """Print configuration summary at startup."""
        logger.info("=" * 60)
        logger.info("HYPERLIQUID MULTI-ASSET TRADING BOT")
        logger.info("=" * 60)
        logger.info(f"Strategy mode: {self.config.strategy.mode}")
        logger.info(f"Max positions: {self.config.risk.max_positions}")
        logger.info(f"Max leverage: {self.config.risk.max_leverage}x")
        logger.info(f"Max drawdown: {self.config.risk.max_drawdown_pct}%")
        logger.info(f"Position risk: {self.config.risk.max_position_risk_pct}% per trade")
        logger.info(f"Portfolio risk: {self.config.risk.max_portfolio_risk_pct}%")
        logger.info(f"Min volume 24h: ${self.config.risk.min_volume_24h:,.0f}")
        logger.info(f"Rebalance interval: {self.config.strategy.rebalance_interval}s")
        logger.info(f"Stop loss: {self.config.risk.stop_loss_pct}%")
        logger.info(f"Take profit: {self.config.risk.take_profit_pct}%")
        logger.info("=" * 60)

    def _initial_scan(self):
        """Do an initial market scan to verify connectivity and data."""
        logger.info("Running initial market scan...")
        assets = self.market_data.get_all_assets()
        if not assets:
            raise RuntimeError("Failed to fetch any assets from Hyperliquid")

        tradeable = self.market_data.get_tradeable_assets()
        logger.info(f"Found {len(assets)} total assets, {len(tradeable)} tradeable")

        # Check account
        user_state = self.market_data.get_user_state(self.config.hyperliquid.wallet_address)
        margin = user_state.get("marginSummary", {})
        equity = float(margin.get("accountValue", 0))
        logger.info(f"Account equity: ${equity:.2f}")

        if equity <= 0:
            logger.warning("Account has no equity! Bot will monitor but cannot trade.")

        return equity

    def run(self):
        """Main bot loop."""
        self._print_startup_info()

        # Setup graceful shutdown
        def shutdown_handler(signum, frame):
            logger.info("Shutdown signal received, stopping bot...")
            self.running = False

        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)

        # Initial connectivity check
        equity = self._initial_scan()

        self.running = True
        cycle = 0
        interval = self.config.strategy.rebalance_interval

        logger.info(f"Bot started. Rebalancing every {interval}s. Press Ctrl+C to stop.")

        while self.running:
            cycle += 1
            cycle_start = time.time()

            try:
                logger.info(f"\n--- Cycle {cycle} ---")
                self.portfolio.rebalance()

                # Print performance summary periodically
                if cycle % 10 == 0:
                    perf = self.portfolio.get_performance_summary()
                    logger.info(f"Performance: {perf}")

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in cycle {cycle}: {e}", exc_info=True)

            # Sleep until next cycle
            elapsed = time.time() - cycle_start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0 and self.running:
                logger.debug(f"Sleeping {sleep_time:.0f}s until next cycle")
                # Sleep in small increments for responsiveness to shutdown
                end_time = time.time() + sleep_time
                while time.time() < end_time and self.running:
                    time.sleep(min(1.0, end_time - time.time()))

        logger.info("Bot stopped.")
        self._print_final_summary()

    def _print_final_summary(self):
        """Print final performance summary."""
        perf = self.portfolio.get_performance_summary()
        logger.info("=" * 60)
        logger.info("FINAL SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total trades: {perf.get('total_trades', 0)}")
        logger.info(f"Opens: {perf.get('opens', 0)}")
        logger.info(f"Closes: {perf.get('closes', 0)}")
        logger.info(f"Symbols traded: {perf.get('symbols_traded', [])}")
        logger.info(f"Peak equity: ${perf.get('peak_equity', 0):.2f}")
        logger.info(f"Final drawdown: {perf.get('current_drawdown', 0):.2f}%")
        logger.info("=" * 60)

    def run_once(self):
        """Run a single rebalance cycle (useful for testing)."""
        self._print_startup_info()
        self._initial_scan()
        self.portfolio.rebalance()
        perf = self.portfolio.get_performance_summary()
        logger.info(f"Single cycle complete. Performance: {perf}")
        return perf

    def dry_run(self):
        """Run analysis without executing trades (read-only mode)."""
        self._print_startup_info()
        equity = self._initial_scan()

        logger.info("\n=== DRY RUN MODE (no trades will be executed) ===\n")

        tradeable = self.market_data.get_tradeable_assets()
        signals = self.strategy.rank_assets(tradeable)

        logger.info(f"\nTop signals ({len(signals)} actionable):")
        logger.info("-" * 80)
        for i, sig in enumerate(signals[:20]):
            direction = "LONG " if sig.score > 0 else "SHORT"
            asset = self.market_data._asset_cache.get(sig.symbol)
            vol = asset.volume_24h if asset else 0
            logger.info(
                f"  {i+1:2d}. {sig.symbol:10s} | {direction} | "
                f"score={sig.score:+.3f} | conf={sig.confidence:.2f} | "
                f"lev={sig.suggested_leverage}x | "
                f"SL=${sig.stop_loss:.2f} TP=${sig.take_profit:.2f} | "
                f"vol=${vol:,.0f}"
            )

            # Show component breakdown for top 5
            if i < 5:
                for comp_name, comp_data in sig.components.items():
                    comp_score = comp_data.get("score", 0) if isinstance(comp_data, dict) else comp_data
                    logger.info(f"       {comp_name}: {comp_score:+.3f}")

        logger.info("-" * 80)

        # Simulate position sizing
        portfolio = self.portfolio.get_portfolio_state()
        logger.info(f"\nSimulated positions (equity=${equity:.2f}):")
        for sig in signals[:self.config.risk.max_positions]:
            asset = self.market_data._asset_cache.get(sig.symbol)
            if not asset:
                continue
            ob = self.market_data.get_orderbook(sig.symbol)
            sizing = self.risk_manager.calculate_position_size(
                sig, asset, equity, 0, ob
            )
            if sizing:
                logger.info(
                    f"  {sizing.symbol:10s} | {sizing.side:5s} | "
                    f"${sizing.size_usd:8.0f} ({sizing.size_coins:.4f}) | "
                    f"{sizing.leverage}x | margin=${sizing.margin_required:.0f} | "
                    f"R:R={sizing.risk_reward_ratio:.2f}"
                )

        return signals
