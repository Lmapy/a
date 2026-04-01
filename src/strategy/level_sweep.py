from __future__ import annotations

import pandas as pd

from src.data.models import Bar, INSTRUMENT_SPECS, MarketRegime, Position, Signal, SignalDirection
from src.strategy.base import BaseStrategy


class LevelSweepStrategy(BaseStrategy):
    """Trade sweeps of key price levels (PDH/PDL, swing highs/lows).

    Concept: When price sweeps (briefly breaks) a key level and reverses,
    it's trapping breakout traders. We fade the sweep for a mean-reversion
    back into the range.

    Entry conditions:
        1. Price sweeps PDH/PDL or a 4H/1H swing high/low
        2. The sweep candle closes back inside the level (failed breakout)
        3. Confirmation: rejection wick is >= 50% of candle range

    Stop: Beyond the sweep extreme + buffer
    Take profit: Midpoint of prior day range (PDH/PDL midpoint) or 2R
    """

    name = "level_sweep"
    allowed_regimes = ["ranging", "weak_trend", "low_volatility", "high_volatility",
                       "strong_trend_up", "strong_trend_down"]
    blocked_regimes = []

    def __init__(self, params: dict | None = None):
        p = params or {}
        self.sweep_buffer_ticks = p.get("sweep_buffer_ticks", 2)  # min ticks beyond level
        self.rejection_wick_pct = p.get("rejection_wick_pct", 0.50)  # min wick ratio
        self.stop_buffer_ticks = p.get("stop_buffer_ticks", 8)  # stop beyond sweep extreme
        self.take_profit_rr = p.get("take_profit_rr", 2.0)
        self.max_sweep_ticks = p.get("max_sweep_ticks", 40)  # reject if sweep too far
        self.use_pdh_pdl = p.get("use_pdh_pdl", True)
        self.use_swing_levels = p.get("use_swing_levels", False)  # PDH/PDL only by default
        self.swing_lookback_bars = p.get("swing_lookback_bars", 48)  # 4 hours of 5-min bars
        self.min_tick_profit = p.get("min_tick_profit", 0)

        # State
        self._prev_day_high: dict[str, float] = {}
        self._prev_day_low: dict[str, float] = {}
        self._today_high: dict[str, float] = {}
        self._today_low: dict[str, float] = {}
        self._bar_history: dict[str, list[Bar]] = {}
        self._sweep_traded: dict[str, bool] = {}
        self._current_date: dict[str, object] = {}

    def reset(self) -> None:
        """Reset all state for a new backtest run."""
        self._prev_day_high.clear()
        self._prev_day_low.clear()
        self._today_high.clear()
        self._today_low.clear()
        self._bar_history.clear()
        self._sweep_traded.clear()
        self._current_date.clear()

    def reset_daily(self) -> None:
        """Called at start of each trading day."""
        # Move today's range to previous day
        for inst in list(self._today_high.keys()):
            self._prev_day_high[inst] = self._today_high[inst]
            self._prev_day_low[inst] = self._today_low[inst]
        self._today_high.clear()
        self._today_low.clear()
        self._sweep_traded.clear()

    def _update_state(self, bar: Bar) -> None:
        """Track bar data and update levels (no signal generation)."""
        inst = bar.instrument

        # Track current date for daily boundary detection
        bar_date = bar.timestamp.date() if hasattr(bar.timestamp, 'date') else None
        if bar_date and bar_date != self._current_date.get(inst):
            if self._current_date.get(inst) is not None:
                self.reset_daily()
            self._current_date[inst] = bar_date

        # Update today's high/low
        self._today_high[inst] = max(self._today_high.get(inst, bar.high), bar.high)
        self._today_low[inst] = min(self._today_low.get(inst, bar.low), bar.low)

        # Maintain bar history for swing detection
        if inst not in self._bar_history:
            self._bar_history[inst] = []
        self._bar_history[inst].append(bar)
        if len(self._bar_history[inst]) > self.swing_lookback_bars * 2:
            self._bar_history[inst] = self._bar_history[inst][-self.swing_lookback_bars * 2:]

    def on_bar(self, bar: Bar, indicators: pd.Series, regime: MarketRegime) -> Signal | None:
        if not self.is_regime_allowed(regime):
            self._update_state(bar)
            return None

        self._update_state(bar)

        inst = bar.instrument
        spec = INSTRUMENT_SPECS.get(inst, {})
        tick_size = spec.get("tick_size", 0.25)

        # Max 2 sweep trades per day
        if self._sweep_traded.get(inst, 0) >= 2:
            return None

        # Collect key levels
        levels = self._get_key_levels(inst, tick_size)
        if not levels:
            return None

        # Check for sweep + rejection on current bar
        return self._check_sweep(bar, inst, levels, tick_size)

    def track_bar(self, bar: Bar, indicators: pd.Series, regime: MarketRegime) -> None:
        """Lightweight state tracking — just update levels without signal check."""
        self._update_state(bar)

    def _get_key_levels(self, inst: str, tick_size: float) -> list[tuple[str, float, str]]:
        """Returns list of (level_name, price, direction_to_trade_on_sweep).

        direction: 'high' means it's a resistance level (sweep above → sell),
                   'low' means it's a support level (sweep below → buy).
        """
        levels = []

        # PDH/PDL
        if self.use_pdh_pdl:
            if inst in self._prev_day_high:
                levels.append(("PDH", self._prev_day_high[inst], "high"))
            if inst in self._prev_day_low:
                levels.append(("PDL", self._prev_day_low[inst], "low"))

        # Swing highs/lows from bar history
        if self.use_swing_levels and inst in self._bar_history:
            bars = self._bar_history[inst]
            swing_levels = self._find_swing_levels(bars, tick_size)
            levels.extend(swing_levels)

        return levels

    def _find_swing_levels(self, bars: list[Bar], tick_size: float) -> list[tuple[str, float, str]]:
        """Find significant swing highs and lows from recent bar history."""
        if len(bars) < 10:
            return []

        levels = []
        lookback = min(self.swing_lookback_bars, len(bars) - 2)

        # Find swing highs (bars where high is higher than N surrounding bars)
        for i in range(5, lookback):
            bar = bars[-(i + 1)]
            is_swing_high = True
            is_swing_low = True

            for j in range(1, 4):  # check 3 bars on each side
                left_idx = -(i + 1 + j)
                right_idx = -(i + 1 - j)
                if abs(left_idx) > len(bars) or abs(right_idx) > len(bars) or right_idx >= 0:
                    is_swing_high = False
                    is_swing_low = False
                    break
                left_bar = bars[left_idx]
                right_bar = bars[right_idx]
                if left_bar.high >= bar.high or right_bar.high >= bar.high:
                    is_swing_high = False
                if left_bar.low <= bar.low or right_bar.low <= bar.low:
                    is_swing_low = False

            if is_swing_high:
                levels.append(("SwingH", bar.high, "high"))
            if is_swing_low:
                levels.append(("SwingL", bar.low, "low"))

        # Deduplicate close levels (within 5 ticks)
        unique = []
        for name, price, direction in levels:
            is_dup = False
            for _, up, ud in unique:
                if abs(price - up) < 5 * tick_size and direction == ud:
                    is_dup = True
                    break
            if not is_dup:
                unique.append((name, price, direction))

        return unique[:4]  # max 4 swing levels

    def _check_sweep(self, bar: Bar, inst: str, levels: list, tick_size: float) -> Signal | None:
        """Check if current bar sweeps a level and rejects."""
        buffer = self.sweep_buffer_ticks * tick_size
        max_sweep = self.max_sweep_ticks * tick_size
        candle_range = bar.high - bar.low
        if candle_range < tick_size:
            return None

        for level_name, level_price, level_type in levels:
            if level_type == "high":
                # Sweep above resistance: bar.high > level, but bar.close < level
                swept_above = bar.high > level_price + buffer
                closed_below = bar.close < level_price
                sweep_distance = bar.high - level_price

                if swept_above and closed_below and sweep_distance < max_sweep:
                    # Rejection wick check: upper wick must be significant
                    upper_wick = bar.high - max(bar.open, bar.close)
                    if upper_wick / candle_range >= self.rejection_wick_pct:
                        # SHORT signal: fade the sweep
                        stop = bar.high + self.stop_buffer_ticks * tick_size
                        risk = stop - bar.close
                        tp = bar.close - risk * self.take_profit_rr

                        # Enforce min tick profit
                        if self.min_tick_profit > 0:
                            min_tp = bar.close - self.min_tick_profit * tick_size
                            tp = min(tp, min_tp)

                        self._sweep_traded[inst] = self._sweep_traded.get(inst, 0) + 1
                        return Signal(
                            direction=SignalDirection.SHORT,
                            instrument=inst,
                            entry_price=bar.close,
                            stop_loss=stop,
                            take_profit=tp,
                            confidence=1.0,
                            strategy_name=self.name,
                            metadata={
                                "level_name": level_name,
                                "level_price": level_price,
                                "sweep_distance": sweep_distance,
                            },
                        )

            elif level_type == "low":
                # Sweep below support: bar.low < level, but bar.close > level
                swept_below = bar.low < level_price - buffer
                closed_above = bar.close > level_price
                sweep_distance = level_price - bar.low

                if swept_below and closed_above and sweep_distance < max_sweep:
                    # Rejection wick check: lower wick must be significant
                    lower_wick = min(bar.open, bar.close) - bar.low
                    if lower_wick / candle_range >= self.rejection_wick_pct:
                        # LONG signal: fade the sweep
                        stop = bar.low - self.stop_buffer_ticks * tick_size
                        risk = bar.close - stop
                        tp = bar.close + risk * self.take_profit_rr

                        # Enforce min tick profit
                        if self.min_tick_profit > 0:
                            min_tp = bar.close + self.min_tick_profit * tick_size
                            tp = max(tp, min_tp)

                        self._sweep_traded[inst] = self._sweep_traded.get(inst, 0) + 1
                        return Signal(
                            direction=SignalDirection.LONG,
                            instrument=inst,
                            entry_price=bar.close,
                            stop_loss=stop,
                            take_profit=tp,
                            confidence=1.0,
                            strategy_name=self.name,
                            metadata={
                                "level_name": level_name,
                                "level_price": level_price,
                                "sweep_distance": sweep_distance,
                            },
                        )

        return None

    def should_exit(self, position: Position, bar: Bar, indicators: pd.Series) -> Signal | None:
        # Check stop loss
        if position.stop_loss is not None:
            if position.direction == SignalDirection.LONG and bar.low <= position.stop_loss:
                return self._exit_signal(position, position.stop_loss, "stop_loss")
            if position.direction == SignalDirection.SHORT and bar.high >= position.stop_loss:
                return self._exit_signal(position, position.stop_loss, "stop_loss")

        # Check take profit
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
            stop_loss=0,
            take_profit=0,
            confidence=1.0,
            strategy_name="level_sweep",
            metadata={"exit_reason": reason},
        )
