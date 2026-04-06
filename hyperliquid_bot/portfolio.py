"""Portfolio manager: position tracking, rebalancing, and execution."""

import logging
import time
from dataclasses import dataclass, field

from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

from .config import BotConfig
from .market_data import HyperliquidData
from .risk_manager import PortfolioRisk, PositionSizing, RiskManager
from .strategy import AssetSignal, Signal, StrategyEngine

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    timestamp: float
    symbol: str
    side: str
    size: float
    price: float
    leverage: int
    action: str  # "open", "close", "reduce"
    reason: str
    pnl: float = 0.0


class PortfolioManager:
    """Manages the full lifecycle of positions and portfolio rebalancing."""

    def __init__(
        self,
        config: BotConfig,
        exchange: Exchange,
        market_data: HyperliquidData,
        strategy: StrategyEngine,
        risk_manager: RiskManager,
    ):
        self.config = config
        self.exchange = exchange
        self.market_data = market_data
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.trade_history: list[TradeRecord] = []
        self.active_signals: dict[str, AssetSignal] = {}

    def get_portfolio_state(self) -> PortfolioRisk:
        """Fetch current portfolio state from Hyperliquid."""
        user_state = self.market_data.get_user_state(self.config.hyperliquid.wallet_address)
        return self.risk_manager.assess_portfolio_risk(user_state)

    def _execute_open(self, sizing: PositionSizing) -> bool:
        """Open a new position."""
        try:
            is_buy = sizing.side == "long"

            # Set leverage first
            self.exchange.update_leverage(sizing.leverage, sizing.symbol, is_cross=True)

            # Place market order
            result = self.exchange.market_open(
                sizing.symbol,
                is_buy,
                sizing.size_coins,
                None,  # slippage handled by default
            )

            if result.get("status") == "ok":
                statuses = result.get("response", {}).get("data", {}).get("statuses", [])
                if statuses and "filled" in statuses[0]:
                    fill = statuses[0]["filled"]
                    fill_price = float(fill.get("avgPx", sizing.stop_loss))
                    logger.info(
                        f"OPENED {sizing.side.upper()} {sizing.symbol}: "
                        f"{sizing.size_coins} @ ${fill_price:.4f} "
                        f"(${sizing.size_usd:.0f}, {sizing.leverage}x)"
                    )
                    self.trade_history.append(TradeRecord(
                        timestamp=time.time(),
                        symbol=sizing.symbol,
                        side=sizing.side,
                        size=sizing.size_coins,
                        price=fill_price,
                        leverage=sizing.leverage,
                        action="open",
                        reason=f"signal_score={self.active_signals.get(sizing.symbol, {})}"
                    ))

                    # Place stop loss as a trigger order
                    self._place_stop_loss(sizing)
                    self._place_take_profit(sizing)
                    return True
                elif statuses and "error" in statuses[0]:
                    logger.error(f"Order error for {sizing.symbol}: {statuses[0]['error']}")
                    return False

            logger.warning(f"Unexpected order result for {sizing.symbol}: {result}")
            return False

        except Exception as e:
            logger.error(f"Failed to open {sizing.side} {sizing.symbol}: {e}")
            return False

    def _place_stop_loss(self, sizing: PositionSizing):
        """Place a stop-loss trigger order."""
        try:
            is_buy = sizing.side == "short"  # Opposite side to close
            trigger_px = sizing.stop_loss

            order_result = self.exchange.order(
                sizing.symbol,
                is_buy,
                sizing.size_coins,
                trigger_px,
                {"trigger": {"triggerPx": trigger_px, "isMarket": True, "tpsl": "sl"}},
            )
            logger.info(f"Stop loss set for {sizing.symbol} @ ${trigger_px:.4f}")
        except Exception as e:
            logger.warning(f"Failed to set stop loss for {sizing.symbol}: {e}")

    def _place_take_profit(self, sizing: PositionSizing):
        """Place a take-profit trigger order."""
        try:
            is_buy = sizing.side == "short"
            trigger_px = sizing.take_profit

            order_result = self.exchange.order(
                sizing.symbol,
                is_buy,
                sizing.size_coins,
                trigger_px,
                {"trigger": {"triggerPx": trigger_px, "isMarket": True, "tpsl": "tp"}},
            )
            logger.info(f"Take profit set for {sizing.symbol} @ ${trigger_px:.4f}")
        except Exception as e:
            logger.warning(f"Failed to set take profit for {sizing.symbol}: {e}")

    def _execute_close(self, symbol: str, reason: str) -> bool:
        """Close an existing position."""
        try:
            result = self.exchange.market_close(symbol)
            if result.get("status") == "ok":
                logger.info(f"CLOSED {symbol}: {reason}")
                self.trade_history.append(TradeRecord(
                    timestamp=time.time(),
                    symbol=symbol,
                    side="close",
                    size=0,
                    price=0,
                    leverage=0,
                    action="close",
                    reason=reason,
                ))
                return True
            logger.warning(f"Close failed for {symbol}: {result}")
            return False
        except Exception as e:
            logger.error(f"Failed to close {symbol}: {e}")
            return False

    def manage_existing_positions(self, portfolio: PortfolioRisk, current_prices: dict[str, float]):
        """Review and manage all existing positions."""
        for symbol, pos_data in portfolio.position_risks.items():
            price = current_prices.get(symbol, 0)
            if price <= 0:
                continue

            should_close, reason = self.risk_manager.should_close_position(symbol, pos_data, price)
            if should_close:
                logger.info(f"Closing {symbol}: {reason}")
                self._execute_close(symbol, reason)

    def open_new_positions(self, signals: list[AssetSignal], portfolio: PortfolioRisk):
        """Open new positions based on ranked signals."""
        opened = 0
        max_new = min(
            self.config.risk.max_positions - portfolio.num_positions,
            5,  # Max 5 new positions per cycle
        )

        if max_new <= 0:
            return

        for signal in signals:
            if opened >= max_new:
                break

            if signal.signal == Signal.NEUTRAL:
                continue

            # Get asset info from cache
            asset = self.market_data._asset_cache.get(signal.symbol)
            if not asset:
                continue

            orderbook = self.market_data.get_orderbook(signal.symbol)

            sizing = self.risk_manager.calculate_position_size(
                signal=signal,
                asset=asset,
                account_equity=portfolio.total_equity,
                current_positions=portfolio.num_positions + opened,
                orderbook=orderbook,
            )

            if not sizing:
                continue

            valid, reason = self.risk_manager.validate_new_trade(sizing, portfolio)
            if not valid:
                logger.info(f"Trade rejected for {signal.symbol}: {reason}")
                continue

            self.active_signals[signal.symbol] = signal
            if self._execute_open(sizing):
                opened += 1

        if opened > 0:
            logger.info(f"Opened {opened} new positions this cycle")

    def rebalance(self):
        """Full rebalance cycle: review existing, score universe, open new."""
        logger.info("=" * 60)
        logger.info("Starting rebalance cycle")
        logger.info("=" * 60)

        # 1. Get current portfolio state
        portfolio = self.get_portfolio_state()
        logger.info(
            f"Portfolio: equity=${portfolio.total_equity:.2f}, "
            f"margin={portfolio.margin_usage_pct:.1f}%, "
            f"positions={portfolio.num_positions}, "
            f"drawdown={self.risk_manager.current_drawdown:.2f}%"
        )

        # 2. Emergency: if drawdown breached, close everything
        if portfolio.max_drawdown_hit:
            logger.warning("MAX DRAWDOWN BREACHED - closing all positions")
            for symbol in list(portfolio.position_risks.keys()):
                self._execute_close(symbol, "max_drawdown_emergency")
            return

        # 3. Get current prices
        current_prices = self.market_data.get_all_mids()

        # 4. Manage existing positions (stop loss / take profit / trailing)
        self.manage_existing_positions(portfolio, current_prices)

        # 5. Refresh portfolio state after closes
        portfolio = self.get_portfolio_state()

        # 6. Score the universe of tradeable assets
        tradeable = self.market_data.get_tradeable_assets()
        if not tradeable:
            logger.warning("No tradeable assets found")
            return

        signals = self.strategy.rank_assets(tradeable)

        # 7. Open new positions from top signals
        self.open_new_positions(signals, portfolio)

        # 8. Log summary
        portfolio = self.get_portfolio_state()
        logger.info(
            f"Post-rebalance: equity=${portfolio.total_equity:.2f}, "
            f"positions={portfolio.num_positions}, "
            f"unrealized_pnl=${portfolio.total_unrealized_pnl:.2f}"
        )
        logger.info(f"Total trades this session: {len(self.trade_history)}")

    def get_performance_summary(self) -> dict:
        """Get a summary of trading performance."""
        if not self.trade_history:
            return {"total_trades": 0}

        opens = [t for t in self.trade_history if t.action == "open"]
        closes = [t for t in self.trade_history if t.action == "close"]

        return {
            "total_trades": len(self.trade_history),
            "opens": len(opens),
            "closes": len(closes),
            "symbols_traded": list(set(t.symbol for t in self.trade_history)),
            "peak_equity": self.risk_manager.peak_equity,
            "current_drawdown": self.risk_manager.current_drawdown,
        }
