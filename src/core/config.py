from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DrawdownConfig:
    type: str = "eod"
    initial_amount: float = 2000.0
    trails_up: bool = True


@dataclass
class TradingHoursConfig:
    start: str = "08:30"
    end: str = "15:00"
    flat_by_close: bool = True


@dataclass
class FirmConfig:
    firm_name: str = "Topstep"
    account_name: str = "50k_combine"
    account_size: float = 50000.0
    profit_target: float = 3000.0
    trailing_drawdown: DrawdownConfig = field(default_factory=DrawdownConfig)
    daily_loss_limit: float | None = None
    max_contracts: dict[str, int] = field(default_factory=lambda: {"ES": 2, "NQ": 2, "MES": 10, "MNQ": 10})
    min_trading_days: int = 0
    max_calendar_days: int | None = None
    allowed_instruments: list[str] = field(default_factory=lambda: ["ES", "NQ", "MES", "MNQ"])
    trading_hours: TradingHoursConfig = field(default_factory=TradingHoursConfig)
    min_tick_profit: int = 0  # minimum ticks profit per trade (MFF requires 4)


@dataclass
class RiskConfig:
    max_daily_loss_pct: float = 50.0
    scale_up_threshold: float = 1000.0
    scale_down_threshold: float = 500.0
    min_drawdown_buffer: float = 300.0
    end_of_day_flatten_minutes: int = 5
    max_risk_per_trade_pct: float = 1.0
    max_consecutive_losses: int = 3
    profit_protection_threshold: float = 500.0
    acceleration_profit_pct: float = 50.0
    acceleration_multiplier: float = 1.3


@dataclass
class StrategyParamsConfig:
    strategy_name: str = ""
    enabled: bool = True
    instruments: list[str] = field(default_factory=lambda: ["ES", "NQ"])
    timeframe: str = "5min"
    parameters: dict = field(default_factory=dict)
    regime_filter: dict = field(default_factory=dict)
    risk: dict = field(default_factory=dict)


@dataclass
class MetaConfig:
    regime_lookback_bars: int = 100
    regime_recalc_interval: int = 10
    strategy_cooldown_bars: int = 5


@dataclass
class GlobalConfig:
    firm: FirmConfig = field(default_factory=FirmConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    meta: MetaConfig = field(default_factory=MetaConfig)
    strategies: dict[str, StrategyParamsConfig] = field(default_factory=dict)


def load_firm_config(path: str | Path) -> FirmConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    dd = data.get("trailing_drawdown", {})
    th = data.get("trading_hours", {})
    return FirmConfig(
        firm_name=data.get("firm_name", "Unknown"),
        account_name=data.get("account_name", ""),
        account_size=data.get("account_size", 50000),
        profit_target=data.get("profit_target", 3000),
        trailing_drawdown=DrawdownConfig(**dd),
        daily_loss_limit=data.get("daily_loss_limit"),
        max_contracts=data.get("max_contracts", {}),
        min_trading_days=data.get("min_trading_days", 0),
        max_calendar_days=data.get("max_calendar_days"),
        allowed_instruments=data.get("allowed_instruments", []),
        trading_hours=TradingHoursConfig(**th),
        min_tick_profit=data.get("min_tick_profit", 0),
    )


def load_strategy_config(path: str | Path) -> StrategyParamsConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return StrategyParamsConfig(
        strategy_name=data.get("strategy_name", ""),
        enabled=data.get("enabled", True),
        instruments=data.get("instruments", ["ES", "NQ"]),
        timeframe=data.get("timeframe", "5min"),
        parameters=data.get("parameters", {}),
        regime_filter=data.get("regime_filter", {}),
        risk=data.get("risk", {}),
    )


def load_global_config(config_dir: str | Path) -> GlobalConfig:
    config_dir = Path(config_dir)
    firm_dir = config_dir / "firms"
    strat_dir = config_dir / "strategies"
    global_file = config_dir / "global.yaml"

    # Load global risk/meta settings
    risk_cfg = RiskConfig()
    meta_cfg = MetaConfig()
    if global_file.exists():
        with open(global_file) as f:
            gdata = yaml.safe_load(f) or {}
        risk_data = gdata.get("risk", {})
        meta_data = gdata.get("meta", {})
        risk_cfg = RiskConfig(**{k: v for k, v in risk_data.items() if hasattr(RiskConfig, k)})
        meta_cfg = MetaConfig(**{k: v for k, v in meta_data.items() if hasattr(MetaConfig, k)})

    # Load first firm config found (or default)
    firm_cfg = FirmConfig()
    if firm_dir.exists():
        for fp in sorted(firm_dir.glob("*.yaml")):
            firm_cfg = load_firm_config(fp)
            break

    # Load all strategy configs
    strategies: dict[str, StrategyParamsConfig] = {}
    if strat_dir.exists():
        for sp in sorted(strat_dir.glob("*.yaml")):
            scfg = load_strategy_config(sp)
            if scfg.enabled:
                strategies[scfg.strategy_name] = scfg

    return GlobalConfig(firm=firm_cfg, risk=risk_cfg, meta=meta_cfg, strategies=strategies)
