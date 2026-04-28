"""Shared dataclasses and type aliases for v2.

These are the only types that cross module boundaries. Keep this file
small -- if you find yourself adding helpers here, they probably belong
in core/<topic>.py instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

Direction = Literal[-1, 1]
Side = Literal["long", "short"]
EntryName = str
RegimeName = str


@dataclass
class Bar:
    time: pd.Timestamp   # UTC, tz-aware
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    spread: float = 0.0


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: Direction
    entry: float
    exit: float
    cost: float                  # round-trip cost in price units
    pnl: float                   # price units
    ret: float                   # pnl / entry
    # trade-level analytics (filled by analytics.trade_metrics)
    mae: float = 0.0             # max adverse excursion (price units, signed favourable=+)
    mfe: float = 0.0             # max favourable excursion (price units)
    time_to_tp_min: float | None = None
    time_to_sl_min: float | None = None
    near_miss_tp: bool = False   # came within 1 tick of TP without hitting it
    fill_kind: str = "market"    # market | limit | missed
    slippage: float = 0.0        # price units, signed against trade
    spread_paid: float = 0.0     # price units (one leg)
    h4_bucket: pd.Timestamp | None = None  # parent H4 bar this trade belongs to
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class Spec:
    """A v2 strategy spec. Schema is intentionally explicit."""
    id: str
    bias_timeframe: str = "H4"
    setup_timeframe: str = "H4"
    entry_timeframe: str = "M15"
    signal: dict = field(default_factory=lambda: {"type": "prev_color"})
    filters: list[dict] = field(default_factory=list)
    entry: dict = field(default_factory=lambda: {"type": "touch_entry"})
    stop: dict = field(default_factory=lambda: {"type": "prev_h4_open"})
    exit: dict = field(default_factory=lambda: {"type": "h4_close"})
    risk: dict = field(default_factory=lambda: {"per_trade_pct": 0.5})
    cost_bps: float = 1.5
    seed: int | None = None

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "bias_timeframe": self.bias_timeframe,
            "setup_timeframe": self.setup_timeframe,
            "entry_timeframe": self.entry_timeframe,
            "signal": self.signal,
            "filters": self.filters,
            "entry": self.entry,
            "stop": self.stop,
            "exit": self.exit,
            "risk": self.risk,
            "cost_bps": self.cost_bps,
            "seed": self.seed,
        }


@dataclass
class FoldResult:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    trades: int
    win_rate: float
    total_return: float
    sharpe_ann: float
    profit_factor: float
    max_drawdown: float


@dataclass
class StrategyReport:
    spec_id: str
    spec: dict
    folds: list[FoldResult]
    holdout_trades: list[Trade]
    statistical: dict
    execution: dict
    prop: dict
    certified: bool
    failures: list[str]
