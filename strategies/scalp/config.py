"""CBR Gold Scalp configuration.

Single source of truth for every knob the strategy exposes. The
config is loaded from YAML, validated, and passed by reference into
every module. Round-trips through YAML so a backtest is fully
reproducible from `results/cbr_gold_scalp/config_used.json`.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import yaml


BiasMode = Literal["OFF", "PREVIOUS_1H_CANDLE_DIRECTION",
                    "EMA_SLOPE", "WICK_BODY_BIAS"]
DXYMode = Literal["OFF", "PREVIOUS_CANDLE_DIRECTION",
                   "EMA_SLOPE", "CLOSE_VS_EMA"]
TriggerMode = Literal["SWEEP_ONLY", "REBALANCE_ONLY",
                       "SWEEP_OR_REBALANCE", "SWEEP_AND_REBALANCE"]
ExpansionMeasureMode = Literal["CLOSE_TO_CLOSE", "HIGH_LOW", "BODY_ONLY"]
EntryMode = Literal["LIMIT_50_RETRACE", "MARKET_ON_MSB_CLOSE",
                     "LIMIT_618_RETRACE", "LIMIT_CUSTOM_RETRACE"]
StopMode = Literal["RECENT_SWING", "EXPANSION_EXTREME", "ATR", "CUSTOM_TICKS"]
TargetMode = Literal["FIXED_R_MULTIPLE", "EXPANSION_MIDPOINT",
                      "PREVIOUS_1H_EQUILIBRIUM",
                      "OPPOSING_EXPANSION_EXTREME",
                      "SESSION_HIGH_LOW"]
SizeMode = Literal["FIXED_QTY", "FIXED_RISK_CURRENCY", "PERCENT_EQUITY"]
StructureBreakMode = Literal["CLOSE_THROUGH", "WICK_THROUGH"]
AmbiguousResolution = Literal["CONSERVATIVE", "OPTIMISTIC", "SKIP_TRADE"]


@dataclass
class SessionConfig:
    timezone: str = "Australia/Melbourne"
    asia_session_start: str = "09:00"
    asia_session_end: str = "12:00"
    execution_window_start: str = "10:00"
    execution_window_end: str = "11:00"
    allow_setup_outside_execution: bool = True
    allow_entry_outside_execution: bool = False


@dataclass
class HTFBiasConfig:
    bias_mode: BiasMode = "PREVIOUS_1H_CANDLE_DIRECTION"
    ema_length: int = 20
    wick_body_threshold: float = 0.55     # large wick if wick/range >= this


@dataclass
class DXYConfig:
    dxy_mode: DXYMode = "OFF"
    dxy_symbol: str = "DXY"               # not in default dataset
    dxy_timeframe: str = "H1"
    ema_length: int = 20
    require_inverse_confirmation: bool = True


@dataclass
class ExpansionConfig:
    expansion_lookback_bars: int = 20
    min_directional_candle_percent: float = 0.65
    min_net_move_atr_multiple: float = 1.0
    atr_length: int = 14
    expansion_measurement_mode: ExpansionMeasureMode = "CLOSE_TO_CLOSE"
    minimum_total_range_atr_multiple: float = 1.0
    quality_score_threshold: float | None = None    # None = log only, no filter


@dataclass
class TriggerConfig:
    trigger_mode: TriggerMode = "SWEEP_OR_REBALANCE"
    rebalance_tolerance_ticks: int = 2
    rebalance_tolerance_atr_fraction: float = 0.25
    require_midpoint_touch: bool = True
    max_bars_after_expansion: int = 30
    max_bars_between_trigger_and_msb: int = 20


@dataclass
class StructureConfig:
    pivot_left: int = 2
    pivot_right: int = 2
    structure_break_mode: StructureBreakMode = "CLOSE_THROUGH"
    require_structure_shift_after_sweep_or_rebalance: bool = True


@dataclass
class EntryConfig:
    entry_mode: EntryMode = "LIMIT_50_RETRACE"
    custom_retrace: float = 0.5
    entry_expiry_bars: int = 10
    min_entry_distance_ticks: int = 0
    max_entry_distance_ticks: int = 99_999
    one_trade_per_session: bool = True
    max_trades_per_day: int = 2


@dataclass
class StopTargetConfig:
    stop_mode: StopMode = "RECENT_SWING"
    target_mode: TargetMode = "FIXED_R_MULTIPLE"
    risk_reward: float = 1.5
    stop_buffer_ticks: int = 1
    atr_stop_multiple: float = 1.0
    custom_stop_ticks: int = 200


@dataclass
class RiskConfig:
    position_size_mode: SizeMode = "FIXED_QTY"
    fixed_qty: float = 1.0
    fixed_risk_currency: float = 250.0
    risk_percent: float = 0.5
    initial_equity: float = 50_000.0
    tick_size: float = 0.10               # XAUUSD tick on Dukascopy spot
    tick_value: float = 0.10              # 1 unit / 1 contract assumption
    break_even_after_r: float | None = None
    # ---- partial-exit (Batch K2) -------------------------------------
    # Enable a "TP1 + runner" exit: sell `tp1_percent` of the position
    # at `tp1_r` R, optionally move the stop to break-even on the
    # remainder, optionally trail the runner with a fixed-R chandelier.
    partial_tp_enabled: bool = False
    tp1_r: float = 1.0                   # first target in R-multiples
    tp1_percent: float = 0.5             # share of the position closed at TP1
    move_stop_to_be_after_tp1: bool = True
    runner_trail_r: float | None = None  # trailing stop distance in R, runner only
    runner_target_r: float = 2.0         # final target for the runner (R)
    trail_after_tp1: bool = False        # legacy flag; same as runner_trail_r != None


@dataclass
class ATRRegimeConfig:
    """Optional ATR-regime filter (Batch K2).

    Skips setups when the current ATR (price-units) is outside the
    configured percentile band over a rolling lookback. Use to dodge
    very low-volatility chop and very high-volatility news spikes.
    """
    enabled: bool = False
    atr_length: int = 14
    rolling_window: int = 720            # 720 m1 bars = 12h rolling window
    min_percentile: float = 0.20         # reject if ATR rank below this
    max_percentile: float = 0.95         # reject if ATR rank above this


@dataclass
class NewsFilterConfig:
    """Optional CSV-based news filter (Batch K2).

    Skips any setup whose timestamp is within `window_minutes_before`
    or `window_minutes_after` of an event. CSV columns:
        time   — ISO 8601 in UTC
        impact — one of: low / med / high (strings)
        symbol — optional; if provided, only events touching the
                  configured symbol are blocked.
    """
    enabled: bool = False
    csv_path: str | None = None
    window_minutes_before: int = 5
    window_minutes_after: int = 30
    block_impacts: list[str] = field(default_factory=lambda: ["high"])


@dataclass
class EngineConfig:
    same_bar_entry_exit: bool = False           # if entry and stop hit same bar
    ambiguous_bar_resolution: AmbiguousResolution = "CONSERVATIVE"
    market_entry_fill_convention: Literal["close", "next_open"] = "next_open"


@dataclass
class CBRGoldScalpConfig:
    """Top-level config. Compose all the subsections."""
    name: str = "cbr_gold_scalp"
    symbol: str = "XAUUSD"
    primary_timeframe: str = "M1"
    bias_timeframe: str = "H1"

    session: SessionConfig = field(default_factory=SessionConfig)
    htf_bias: HTFBiasConfig = field(default_factory=HTFBiasConfig)
    dxy: DXYConfig = field(default_factory=DXYConfig)
    expansion: ExpansionConfig = field(default_factory=ExpansionConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    entry: EntryConfig = field(default_factory=EntryConfig)
    stop_target: StopTargetConfig = field(default_factory=StopTargetConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    atr_regime: ATRRegimeConfig = field(default_factory=ATRRegimeConfig)
    news: NewsFilterConfig = field(default_factory=NewsFilterConfig)

    # data window
    start_date: str | None = None         # ISO; None = full available
    end_date: str | None = None
    output_dir: str = "results/cbr_gold_scalp"

    # ---- IO -----------------------------------------------------------

    def to_json(self) -> dict:
        """Plain-dict; JSON / YAML round-trip."""
        return asdict(self)

    def to_json_str(self) -> str:
        return json.dumps(self.to_json(), indent=2, default=str)

    def write_used(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_json_str(), encoding="utf-8")

    @classmethod
    def from_json(cls, payload: dict) -> "CBRGoldScalpConfig":
        return cls(
            name=payload.get("name", "cbr_gold_scalp"),
            symbol=payload.get("symbol", "XAUUSD"),
            primary_timeframe=payload.get("primary_timeframe", "M1"),
            bias_timeframe=payload.get("bias_timeframe", "H1"),
            session=SessionConfig(**payload.get("session", {})),
            htf_bias=HTFBiasConfig(**payload.get("htf_bias", {})),
            dxy=DXYConfig(**payload.get("dxy", {})),
            expansion=ExpansionConfig(**payload.get("expansion", {})),
            trigger=TriggerConfig(**payload.get("trigger", {})),
            structure=StructureConfig(**payload.get("structure", {})),
            entry=EntryConfig(**payload.get("entry", {})),
            stop_target=StopTargetConfig(**payload.get("stop_target", {})),
            risk=RiskConfig(**payload.get("risk", {})),
            engine=EngineConfig(**payload.get("engine", {})),
            atr_regime=ATRRegimeConfig(**payload.get("atr_regime", {})),
            news=NewsFilterConfig(**payload.get("news", {})),
            start_date=payload.get("start_date"),
            end_date=payload.get("end_date"),
            output_dir=payload.get("output_dir", "results/cbr_gold_scalp"),
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "CBRGoldScalpConfig":
        text = Path(path).read_text(encoding="utf-8")
        payload = yaml.safe_load(text) or {}
        return cls.from_json(payload)

    # ---- validation ---------------------------------------------------

    def validate(self) -> list[str]:
        """Return a list of issues; empty = OK. Caller decides whether
        to abort or warn."""
        issues: list[str] = []
        if self.expansion.expansion_lookback_bars < 5:
            issues.append("expansion_lookback_bars < 5 is too short")
        if not (0.5 <= self.expansion.min_directional_candle_percent <= 1.0):
            issues.append("min_directional_candle_percent must be in [0.5, 1.0]")
        if self.stop_target.risk_reward <= 0:
            issues.append("risk_reward must be > 0")
        if self.entry.entry_expiry_bars < 1:
            issues.append("entry_expiry_bars must be >= 1")
        if self.structure.pivot_left < 1 or self.structure.pivot_right < 1:
            issues.append("pivot_left/right must be >= 1")
        if self.risk.tick_size <= 0:
            issues.append("tick_size must be > 0")
        return issues
