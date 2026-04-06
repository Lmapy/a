"""Configuration management for the Hyperliquid trading bot."""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class HyperliquidConfig:
    private_key: str = ""
    wallet_address: str = ""
    mainnet: bool = False
    vault_address: str | None = None

    def __post_init__(self):
        self.private_key = self.private_key or os.getenv("HL_PRIVATE_KEY", "")
        self.wallet_address = self.wallet_address or os.getenv("HL_WALLET_ADDRESS", "")
        self.mainnet = os.getenv("HL_MAINNET", "false").lower() == "true"
        self.vault_address = os.getenv("HL_VAULT_ADDRESS") or None

    @property
    def base_url(self) -> str:
        if self.mainnet:
            return "https://api.hyperliquid.xyz"
        return "https://api.hyperliquid-testnet.xyz"


@dataclass
class RiskConfig:
    max_portfolio_risk_pct: float = 2.0
    max_position_risk_pct: float = 0.5
    max_drawdown_pct: float = 10.0
    max_leverage: int = 3
    max_positions: int = 15
    min_volume_24h: float = 500_000.0
    max_spread_pct: float = 0.3
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 4.0
    trailing_stop_pct: float = 1.5

    def __post_init__(self):
        self.max_portfolio_risk_pct = float(os.getenv("MAX_PORTFOLIO_RISK_PCT", self.max_portfolio_risk_pct))
        self.max_position_risk_pct = float(os.getenv("MAX_POSITION_RISK_PCT", self.max_position_risk_pct))
        self.max_drawdown_pct = float(os.getenv("MAX_DRAWDOWN_PCT", self.max_drawdown_pct))
        self.max_leverage = int(os.getenv("MAX_LEVERAGE", self.max_leverage))
        self.max_positions = int(os.getenv("MAX_POSITIONS", self.max_positions))
        self.min_volume_24h = float(os.getenv("MIN_VOLUME_24H", self.min_volume_24h))


@dataclass
class StrategyConfig:
    rebalance_interval: int = 300
    mode: str = "hybrid"  # "momentum", "mean_reversion", "liquidity", "hybrid"
    momentum_weight: float = 0.35
    mean_reversion_weight: float = 0.30
    liquidity_weight: float = 0.20
    funding_weight: float = 0.15
    ema_fast: int = 12
    ema_slow: int = 26
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    volume_ma_period: int = 20
    min_score_threshold: float = 0.3

    def __post_init__(self):
        self.rebalance_interval = int(os.getenv("REBALANCE_INTERVAL_SECONDS", self.rebalance_interval))
        self.mode = os.getenv("STRATEGY_MODE", self.mode)


@dataclass
class CoinGlassConfig:
    api_key: str = ""
    base_url: str = "https://open-api-v3.coinglass.com/api"

    def __post_init__(self):
        self.api_key = self.api_key or os.getenv("COINGLASS_API_KEY", "")

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


@dataclass
class BotConfig:
    hyperliquid: HyperliquidConfig = field(default_factory=HyperliquidConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    coinglass: CoinGlassConfig = field(default_factory=CoinGlassConfig)
    log_level: str = "INFO"

    def validate(self) -> list[str]:
        errors = []
        if not self.hyperliquid.private_key:
            errors.append("HL_PRIVATE_KEY is required")
        if not self.hyperliquid.wallet_address:
            errors.append("HL_WALLET_ADDRESS is required")
        if self.risk.max_leverage < 1 or self.risk.max_leverage > 50:
            errors.append("MAX_LEVERAGE must be between 1 and 50")
        if self.strategy.mode not in ("momentum", "mean_reversion", "liquidity", "hybrid"):
            errors.append(f"Invalid STRATEGY_MODE: {self.strategy.mode}")
        return errors
