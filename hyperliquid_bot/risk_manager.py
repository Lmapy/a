"""Risk management: position sizing, drawdown limits, exposure control."""

import logging
from dataclasses import dataclass

import numpy as np

from .config import RiskConfig
from .market_data import AssetInfo, OrderBookSnapshot
from .strategy import AssetSignal, Signal

logger = logging.getLogger(__name__)


@dataclass
class PositionSizing:
    symbol: str
    side: str  # "long" or "short"
    size_usd: float
    size_coins: float
    leverage: int
    margin_required: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float


@dataclass
class PortfolioRisk:
    total_equity: float
    total_margin_used: float
    total_unrealized_pnl: float
    margin_usage_pct: float
    num_positions: int
    max_drawdown_hit: bool
    risk_budget_remaining: float
    position_risks: dict  # symbol -> risk metrics


class RiskManager:
    """Manages position sizing, risk limits, and portfolio exposure."""

    def __init__(self, config: RiskConfig):
        self.config = config
        self.peak_equity = 0.0
        self.current_drawdown = 0.0

    def update_equity_tracking(self, current_equity: float):
        """Track peak equity and current drawdown."""
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity * 100

    def is_drawdown_breached(self) -> bool:
        """Check if max drawdown limit has been hit."""
        return self.current_drawdown >= self.config.max_drawdown_pct

    def calculate_position_size(
        self,
        signal: AssetSignal,
        asset: AssetInfo,
        account_equity: float,
        current_positions: int,
        orderbook: OrderBookSnapshot | None = None,
    ) -> PositionSizing | None:
        """Calculate optimal position size based on risk parameters."""

        # Don't open if at max positions
        if current_positions >= self.config.max_positions:
            logger.info(f"Max positions ({self.config.max_positions}) reached, skipping {signal.symbol}")
            return None

        # Don't open if in drawdown
        if self.is_drawdown_breached():
            logger.warning(f"Drawdown limit breached ({self.current_drawdown:.1f}%), no new positions")
            return None

        # Don't trade neutral signals
        if signal.signal == Signal.NEUTRAL:
            return None

        side = "long" if signal.signal in (Signal.LONG, Signal.STRONG_LONG) else "short"

        # Base position size from risk budget
        risk_per_trade = account_equity * (self.config.max_position_risk_pct / 100)

        # Scale by signal confidence and strength
        confidence_scalar = 0.5 + 0.5 * signal.confidence
        strength_scalar = min(abs(signal.score) / 0.5, 1.5)
        adjusted_risk = risk_per_trade * confidence_scalar * strength_scalar

        # Determine leverage (capped by config and asset max)
        leverage = min(signal.suggested_leverage, self.config.max_leverage, asset.max_leverage)

        # Position size = risk amount * leverage
        size_usd = adjusted_risk * leverage

        # Cap individual position to % of equity
        max_position_usd = account_equity * (self.config.max_portfolio_risk_pct / 100) * leverage
        size_usd = min(size_usd, max_position_usd)

        # Liquidity check: don't exceed a fraction of orderbook depth
        if orderbook:
            relevant_depth = orderbook.ask_depth if side == "long" else orderbook.bid_depth
            max_from_liquidity = relevant_depth * 0.05  # Max 5% of visible depth
            if size_usd > max_from_liquidity and max_from_liquidity > 0:
                logger.info(f"Reducing {signal.symbol} size from ${size_usd:.0f} to ${max_from_liquidity:.0f} (liquidity)")
                size_usd = max_from_liquidity

        # Spread check: skip if spread too wide
        if orderbook and orderbook.spread_pct > self.config.max_spread_pct:
            logger.info(f"Skipping {signal.symbol}: spread {orderbook.spread_pct:.3f}% > {self.config.max_spread_pct}%")
            return None

        # Minimum size guard
        if size_usd < 10:
            return None

        size_coins = size_usd / asset.mark_price if asset.mark_price > 0 else 0
        margin_required = size_usd / leverage

        # Risk/reward ratio
        if side == "long" and signal.stop_loss > 0 and signal.take_profit > 0:
            risk_dist = asset.mark_price - signal.stop_loss
            reward_dist = signal.take_profit - asset.mark_price
            rr = reward_dist / risk_dist if risk_dist > 0 else 0
        elif side == "short" and signal.stop_loss > 0 and signal.take_profit > 0:
            risk_dist = signal.stop_loss - asset.mark_price
            reward_dist = asset.mark_price - signal.take_profit
            rr = reward_dist / risk_dist if risk_dist > 0 else 0
        else:
            rr = 2.0  # Default assumption

        # Skip trades with poor risk/reward
        if rr < 1.5:
            logger.info(f"Skipping {signal.symbol}: R:R {rr:.2f} < 1.5")
            return None

        return PositionSizing(
            symbol=signal.symbol,
            side=side,
            size_usd=size_usd,
            size_coins=round(size_coins, asset.sz_decimals),
            leverage=leverage,
            margin_required=margin_required,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            risk_reward_ratio=rr,
        )

    def assess_portfolio_risk(self, user_state: dict) -> PortfolioRisk:
        """Assess current portfolio risk from Hyperliquid user state."""
        margin_summary = user_state.get("marginSummary", {})
        equity = float(margin_summary.get("accountValue", 0))
        margin_used = float(margin_summary.get("totalMarginUsed", 0))
        total_pnl = float(margin_summary.get("totalUnrealizedPnl", 0))

        self.update_equity_tracking(equity)

        positions = user_state.get("assetPositions", [])
        position_risks = {}

        for pos_wrapper in positions:
            pos = pos_wrapper.get("position", {})
            symbol = pos.get("coin", "")
            entry_px = float(pos.get("entryPx", 0))
            size = float(pos.get("szi", 0))
            unrealized = float(pos.get("unrealizedPnl", 0))
            margin = float(pos.get("marginUsed", 0))
            leverage_val = float(pos.get("leverage", {}).get("value", 1))

            if abs(size) > 0:
                position_risks[symbol] = {
                    "size": size,
                    "entry_price": entry_px,
                    "unrealized_pnl": unrealized,
                    "margin_used": margin,
                    "leverage": leverage_val,
                    "pnl_pct": (unrealized / margin * 100) if margin > 0 else 0,
                }

        margin_pct = (margin_used / equity * 100) if equity > 0 else 0

        return PortfolioRisk(
            total_equity=equity,
            total_margin_used=margin_used,
            total_unrealized_pnl=total_pnl,
            margin_usage_pct=margin_pct,
            num_positions=len(position_risks),
            max_drawdown_hit=self.is_drawdown_breached(),
            risk_budget_remaining=max(0, equity * (self.config.max_portfolio_risk_pct / 100) - margin_used),
            position_risks=position_risks,
        )

    def should_close_position(self, symbol: str, pos_data: dict, current_price: float) -> tuple[bool, str]:
        """Determine if a position should be closed based on risk rules."""
        entry_px = pos_data["entry_price"]
        size = pos_data["size"]
        pnl_pct = pos_data["pnl_pct"]
        is_long = size > 0

        # Stop loss check
        if is_long:
            loss_pct = (entry_px - current_price) / entry_px * 100
        else:
            loss_pct = (current_price - entry_px) / entry_px * 100

        if loss_pct >= self.config.stop_loss_pct:
            return True, f"stop_loss ({loss_pct:.2f}% loss)"

        # Take profit check
        if is_long:
            gain_pct = (current_price - entry_px) / entry_px * 100
        else:
            gain_pct = (entry_px - current_price) / entry_px * 100

        if gain_pct >= self.config.take_profit_pct:
            return True, f"take_profit ({gain_pct:.2f}% gain)"

        # Drawdown emergency: close all if portfolio drawdown critical
        if self.current_drawdown >= self.config.max_drawdown_pct * 0.9:
            return True, f"emergency_drawdown ({self.current_drawdown:.1f}%)"

        return False, ""

    def validate_new_trade(
        self,
        sizing: PositionSizing,
        portfolio_risk: PortfolioRisk,
    ) -> tuple[bool, str]:
        """Final validation before executing a trade."""

        # Check margin budget
        if sizing.margin_required > portfolio_risk.risk_budget_remaining:
            return False, f"Insufficient risk budget: need ${sizing.margin_required:.0f}, have ${portfolio_risk.risk_budget_remaining:.0f}"

        # Check total margin usage won't exceed safe level
        projected_margin_pct = (
            (portfolio_risk.total_margin_used + sizing.margin_required)
            / portfolio_risk.total_equity * 100
            if portfolio_risk.total_equity > 0 else 100
        )
        if projected_margin_pct > 80:
            return False, f"Projected margin usage {projected_margin_pct:.0f}% would exceed 80% safety limit"

        # Check if already in this asset
        if sizing.symbol in portfolio_risk.position_risks:
            return False, f"Already have position in {sizing.symbol}"

        # Check position count
        if portfolio_risk.num_positions >= self.config.max_positions:
            return False, f"At max positions ({self.config.max_positions})"

        return True, "ok"
