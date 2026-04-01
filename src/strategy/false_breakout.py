"""Failed breakout fade strategy.

Detects false breakouts of the opening range and trades the reversal.

Setup:
  1. Wait for opening range to establish (first N minutes).
  2. Track breakout attempts above OR high or below OR low.
  3. If price breaks out but closes back inside the range within
     a few bars, this is a "false breakout" / "failed break."
  4. Enter opposite direction on the close-back-inside bar.
  5. Stop above the false breakout extreme (the wick high/low).
  6. Target: opposite end of the opening range (mean reversion),
     or a fixed R:R if that doesn't meet minimum.

This complements the breakout strategy — when breakout goes wrong,
this strategy profits from the reversal.
"""
from __future__ import annotations

import pandas as pd

from src.data.models import (
    Bar,
    INSTRUMENT_SPECS,
    MarketRegime,
    Position,
    Signal,
    SignalDirection,
)
from src.strategy.base import BaseStrategy


class FalseBreakoutStrategy(BaseStrategy):
    """Trade failed opening range breakouts (false breakout fades)."""

    name = "false_breakout"
    allowed_regimes = [
        "high_volatility", "strong_trend_up", "strong_trend_down",
        "weak_trend", "ranging", "low_volatility",
    ]
    blocked_regimes: list[str] = []

    def __init__(self, params: dict | None = None):
        p = params or {}
        self.opening_range_minutes: int = p.get("opening_range_minutes", 60)
        self.max_bars_for_fade: int = p.get("max_bars_for_fade", 24)  # 2 hours after OR
        self.min_rr: float = p.get("min_rr", 2.0)  # minimum R:R for entry
        self.default_rr: float = p.get("default_rr", 3.0)  # fallback R:R
        self.stop_buffer_ticks: int = p.get("stop_buffer_ticks", 4)
        self.max_trades_per_day: int = p.get("max_trades_per_day", 2)
        self.min_range_points: float = p.get("min_range_points", 0.5)
        self.max_range_points: float = p.get("max_range_points", 20.0)
        # How many bars after breakout to wait for failure (close back inside)
        self.max_fail_bars: int = p.get("max_fail_bars", 3)
        # Session open config
        self.session_open_hour: int = p.get("session_open_hour", 9)
        self.session_open_minute: int = p.get("session_open_minute", 30)
        self.instrument_session_open: dict[str, tuple[int, int]] = p.get(
            "instrument_session_open", {
                "GC": (18, 0), "MGC": (18, 0), "CL": (18, 0),
            }
        )

        # State
        self._or_high: dict[str, float] = {}
        self._or_low: dict[str, float] = {}
        self._or_established: dict[str, bool] = {}
        self._bar_count: dict[str, int] = {}
        self._bars_after_or: dict[str, int] = {}
        self._session_started: dict[str, bool] = {}
        self._traded_today: dict[str, int] = {}
        self._current_date: dict[str, object] = {}
        # Track breakout state
        self._break_high_active: dict[str, bool] = {}  # True if price has broken above OR
        self._break_low_active: dict[str, bool] = {}   # True if price has broken below OR
        self._break_high_extreme: dict[str, float] = {}  # highest price during upside break
        self._break_low_extreme: dict[str, float] = {}   # lowest price during downside break
        self._break_high_bars: dict[str, int] = {}  # bars since upside break started
        self._break_low_bars: dict[str, int] = {}   # bars since downside break started
        self._faded_high: dict[str, bool] = {}  # already faded a high break today
        self._faded_low: dict[str, bool] = {}   # already faded a low break today

    def reset(self) -> None:
        self.reset_daily()

    def reset_daily(self) -> None:
        self._or_high.clear()
        self._or_low.clear()
        self._or_established.clear()
        self._bar_count.clear()
        self._bars_after_or.clear()
        self._session_started.clear()
        self._traded_today.clear()
        self._break_high_active.clear()
        self._break_low_active.clear()
        self._break_high_extreme.clear()
        self._break_low_extreme.clear()
        self._break_high_bars.clear()
        self._break_low_bars.clear()
        self._faded_high.clear()
        self._faded_low.clear()

    def _update_state(self, bar: Bar) -> None:
        """Track OR and breakout state."""
        inst = bar.instrument

        # Daily reset
        bar_date = bar.timestamp.date() if hasattr(bar.timestamp, "date") else None
        if bar_date and bar_date != self._current_date.get(inst):
            if self._current_date.get(inst) is not None:
                self.reset_daily()
            self._current_date[inst] = bar_date

        # Session open detection (same as breakout strategy)
        bar_time = bar.timestamp.time() if hasattr(bar.timestamp, "time") else None
        if bar_time is not None and not self._session_started.get(inst, False):
            from datetime import time as dt_time
            if inst in self.instrument_session_open:
                h, m = self.instrument_session_open[inst]
            else:
                h, m = self.session_open_hour, self.session_open_minute
            session_open = dt_time(h, m)

            if h >= 17:
                if bar_time < session_open and bar_time >= dt_time(16, 0):
                    return
                self._session_started[inst] = True
            else:
                if bar_time < session_open:
                    return
                self._session_started[inst] = True

        self._bar_count[inst] = self._bar_count.get(inst, 0) + 1

        # Build opening range
        if not self._or_established.get(inst, False):
            self._or_high[inst] = max(self._or_high.get(inst, bar.high), bar.high)
            self._or_low[inst] = min(self._or_low.get(inst, bar.low), bar.low)

            bars_needed = self.opening_range_minutes // 5
            if self._bar_count[inst] >= bars_needed:
                self._or_established[inst] = True
                self._bars_after_or[inst] = 0
            return

        self._bars_after_or[inst] = self._bars_after_or.get(inst, 0) + 1

        or_high = self._or_high[inst]
        or_low = self._or_low[inst]

        # Track upside breakout state
        if bar.high > or_high:
            if not self._break_high_active.get(inst, False):
                # New breakout above OR
                self._break_high_active[inst] = True
                self._break_high_extreme[inst] = bar.high
                self._break_high_bars[inst] = 0
            else:
                # Update extreme
                self._break_high_extreme[inst] = max(
                    self._break_high_extreme.get(inst, bar.high), bar.high
                )
                self._break_high_bars[inst] = self._break_high_bars.get(inst, 0) + 1

        # Track downside breakout state
        if bar.low < or_low:
            if not self._break_low_active.get(inst, False):
                self._break_low_active[inst] = True
                self._break_low_extreme[inst] = bar.low
                self._break_low_bars[inst] = 0
            else:
                self._break_low_extreme[inst] = min(
                    self._break_low_extreme.get(inst, bar.low), bar.low
                )
                self._break_low_bars[inst] = self._break_low_bars.get(inst, 0) + 1

        # Expire stale breakout tracking
        if self._break_high_bars.get(inst, 0) > self.max_fail_bars:
            self._break_high_active[inst] = False
        if self._break_low_bars.get(inst, 0) > self.max_fail_bars:
            self._break_low_active[inst] = False

    def on_bar(self, bar: Bar, indicators: pd.Series, regime: MarketRegime) -> Signal | None:
        self._update_state(bar)

        if not self.is_regime_allowed(regime):
            return None

        inst = bar.instrument
        if not self._or_established.get(inst, False):
            return None

        # Only look for fades within time window
        if self._bars_after_or.get(inst, 0) > self.max_bars_for_fade:
            return None

        if self._traded_today.get(inst, 0) >= self.max_trades_per_day:
            return None

        or_high = self._or_high[inst]
        or_low = self._or_low[inst]
        or_range = or_high - or_low

        if or_range < self.min_range_points or or_range > self.max_range_points:
            return None

        spec = INSTRUMENT_SPECS.get(inst, {})
        tick_size = spec.get("tick_size", 0.25)
        buffer = self.stop_buffer_ticks * tick_size

        # Check for FAILED upside breakout → SHORT
        # Conditions: price broke above OR high, but current bar closes back below it
        if (self._break_high_active.get(inst, False)
                and not self._faded_high.get(inst, False)
                and bar.close < or_high):

            extreme = self._break_high_extreme.get(inst, or_high)
            stop = extreme + buffer
            risk = stop - bar.close

            if risk > 0:
                # Target: opposite end of OR (or_low), but ensure min R:R
                target_structural = or_low
                reward_structural = bar.close - target_structural

                if reward_structural / risk >= self.min_rr:
                    tp = target_structural
                elif risk * self.default_rr > 0:
                    tp = bar.close - risk * self.default_rr
                    if (bar.close - tp) / risk < self.min_rr:
                        tp = None  # skip

                if tp is not None:
                    reward = bar.close - tp
                    self._faded_high[inst] = True
                    self._break_high_active[inst] = False
                    self._traded_today[inst] = self._traded_today.get(inst, 0) + 1

                    return Signal(
                        direction=SignalDirection.SHORT,
                        instrument=inst,
                        entry_price=bar.close,
                        stop_loss=stop,
                        take_profit=tp,
                        confidence=1.0,
                        strategy_name=self.name,
                        metadata={
                            "setup_type": "failed_upside_break",
                            "or_high": or_high, "or_low": or_low,
                            "false_break_extreme": extreme,
                            "risk": risk, "reward": reward,
                        },
                    )

        # Check for FAILED downside breakout → LONG
        if (self._break_low_active.get(inst, False)
                and not self._faded_low.get(inst, False)
                and bar.close > or_low):

            extreme = self._break_low_extreme.get(inst, or_low)
            stop = extreme - buffer
            risk = bar.close - stop

            if risk > 0:
                target_structural = or_high
                reward_structural = target_structural - bar.close

                tp = None
                if reward_structural / risk >= self.min_rr:
                    tp = target_structural
                elif risk * self.default_rr > 0:
                    tp = bar.close + risk * self.default_rr
                    if (tp - bar.close) / risk < self.min_rr:
                        tp = None

                if tp is not None:
                    reward = tp - bar.close
                    self._faded_low[inst] = True
                    self._break_low_active[inst] = False
                    self._traded_today[inst] = self._traded_today.get(inst, 0) + 1

                    return Signal(
                        direction=SignalDirection.LONG,
                        instrument=inst,
                        entry_price=bar.close,
                        stop_loss=stop,
                        take_profit=tp,
                        confidence=1.0,
                        strategy_name=self.name,
                        metadata={
                            "setup_type": "failed_downside_break",
                            "or_high": or_high, "or_low": or_low,
                            "false_break_extreme": extreme,
                            "risk": risk, "reward": reward,
                        },
                    )

        return None

    def track_bar(self, bar: Bar, indicators: pd.Series, regime: MarketRegime) -> None:
        self._update_state(bar)

    def should_exit(self, position: Position, bar: Bar, indicators: pd.Series) -> Signal | None:
        if position.stop_loss is not None:
            if position.direction == SignalDirection.LONG and bar.low <= position.stop_loss:
                return self._exit_signal(position, position.stop_loss, "stop_loss")
            if position.direction == SignalDirection.SHORT and bar.high >= position.stop_loss:
                return self._exit_signal(position, position.stop_loss, "stop_loss")

        if position.take_profit is not None:
            if position.direction == SignalDirection.LONG and bar.high >= position.take_profit:
                return self._exit_signal(position, position.take_profit, "take_profit")
            if position.direction == SignalDirection.SHORT and bar.low <= position.take_profit:
                return self._exit_signal(position, position.take_profit, "take_profit")

        return None

    @staticmethod
    def _exit_signal(position: Position, price: float, reason: str) -> Signal:
        exit_dir = SignalDirection.SHORT if position.direction == SignalDirection.LONG else SignalDirection.LONG
        return Signal(
            direction=exit_dir,
            instrument=position.instrument,
            entry_price=price,
            stop_loss=0, take_profit=0, confidence=1.0,
            strategy_name="false_breakout",
            metadata={"exit_reason": reason},
        )
