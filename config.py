"""Central configuration for the Gold Price Action Prop Firm Strategy."""

from dataclasses import dataclass, field
from pathlib import Path


# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data" / "cache"
RESULTS_DIR = Path(__file__).parent / "results"


# ── Contract Specifications ────────────────────────────────────────────────────
@dataclass(frozen=True)
class ContractSpec:
    name: str
    tick_size: float      # minimum price increment in dollars
    tick_value: float     # dollar value per tick per contract
    oz_per_contract: int  # troy ounces per contract

GC = ContractSpec(name="GC", tick_size=0.10, tick_value=10.0, oz_per_contract=100)
MGC = ContractSpec(name="MGC", tick_size=0.10, tick_value=1.0, oz_per_contract=10)

# Default contract to trade
CONTRACT = MGC


# ── Prop Firm Rules ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PropFirmRules:
    name: str
    starting_balance: float
    profit_target: float
    max_trailing_drawdown: float   # EOD trailing drawdown
    daily_loss_limit: float | None  # None = no daily limit (MFF)
    consistency_pct: float          # best day <= this % of profit target
    max_contracts: int              # max mini contracts
    max_micros: int                 # max micro contracts
    max_trading_days: int

TOPSTEP_50K = PropFirmRules(
    name="TopStep 50K",
    starting_balance=50_000.0,
    profit_target=3_000.0,
    max_trailing_drawdown=2_000.0,
    daily_loss_limit=1_000.0,
    consistency_pct=0.50,
    max_contracts=5,
    max_micros=50,
    max_trading_days=22,
)

MFF_50K = PropFirmRules(
    name="MyFundedFutures 50K",
    starting_balance=50_000.0,
    profit_target=3_000.0,
    max_trailing_drawdown=2_000.0,
    daily_loss_limit=None,
    consistency_pct=0.50,
    max_contracts=5,
    max_micros=50,
    max_trading_days=22,
)

# Default rule set (TopStep is stricter, so if we pass TopStep we pass MFF)
RULES = TOPSTEP_50K


# ── Strategy Parameters ────────────────────────────────────────────────────────
@dataclass
class StrategyParams:
    # Swing detection
    swing_lookback: int = 3           # candles on each side to confirm swing
    htf_timeframe: str = "15min"      # higher-timeframe for structure (15min)
    entry_timeframe: str = "15min"    # entry timeframe

    # Fair Value Gap
    fvg_min_size: float = 1.00        # minimum FVG gap in dollars

    # Order Blocks
    ob_max_age_candles: int = 50      # OB expires after N candles

    # Liquidity
    equal_level_tolerance: float = 1.00   # dollars for equal highs/lows
    sweep_lookback_candles: int = 30      # how far back to look for sweeps

    # Signal confluence
    min_confluence_score: int = 3     # minimum score out of 5 to enter

    # Risk
    risk_per_trade: float = 300.0     # max dollar risk per trade
    reward_risk_ratio: float = 2.0    # R:R target
    sl_buffer: float = 1.50           # SL buffer beyond zone in dollars

    # Daily gates (buffers inside prop firm limits)
    daily_loss_gate: float = 600.0    # stop trading if daily loss >= this
    daily_profit_gate: float = 1_200.0  # stop trading if daily profit >= this

    # Execution
    slippage_ticks: int = 1           # ticks of slippage per trade
    commission_per_side: float = 0.62 # commission per side per MGC contract
    max_trades_per_day: int = 3       # allow up to 3 trades per day

PARAMS = StrategyParams()


# ── Session Times (US/Central timezone hours) ──────────────────────────────────
@dataclass(frozen=True)
class SessionTimes:
    # Hours in US/Central (CT)
    asian_start: int = 18    # 6:00 PM CT (previous calendar day)
    asian_end: int = 2       # 2:00 AM CT
    london_start: int = 2    # 2:00 AM CT
    london_end: int = 8      # 8:00 AM CT
    ny_start: int = 8        # 8:00 AM CT
    ny_end: int = 17         # 5:00 PM CT

SESSIONS = SessionTimes()


# ── Data Download Settings ─────────────────────────────────────────────────────
DOWNLOAD_SYMBOL = "XAUUSD"
DOWNLOAD_START = "2024-01-01"
DOWNLOAD_END = "2025-12-31"
