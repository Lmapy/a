from __future__ import annotations

import pandas as pd

from src.data.models import Bar, MarketRegime, Position, Signal, SignalDirection
from src.strategy.base import BaseStrategy


class TrendFollowingStrategy(BaseStrategy):
    """Trade in the direction of the trend using EMA cross + pullback entry.

    Entry:
        - EMA fast > EMA slow (uptrend): wait for pullback bar (red) then
          bullish bar near fast EMA → go LONG
        - EMA fast < EMA slow (downtrend): wait for rally bar (green) then
          bearish bar near fast EMA → go SHORT
    Exit:
        - Hard stop at stop_loss_atr * ATR from entry
        - Take profit at rr_target * risk from entry
        - Trailing stop once in profit
    """

    name = "trend_following"
    allowed_regimes = ["strong_trend_up", "strong_trend_down", "weak_trend", "ranging", "high_volatility"]
    blocked_regimes = ["low_volatility"]

    def __init__(self, params: dict | None = None):
        p = params or {}
        self.ema_fast = p.get("ema_fast", 20)
        self.ema_slow = p.get("ema_slow", 50)
        self.adx_threshold = p.get("adx_threshold", 0)  # disabled by default
        self.stop_loss_atr = p.get("stop_loss_atr", 2.0)
        self.rr_target = p.get("rr_target", 3.0)
        self.trailing_stop_atr = p.get("trailing_stop_atr", 2.5)
        self._best_price: dict[str, float] = {}
        self._prev_bar: Bar | None = None

    def on_bar(self, bar: Bar, indicators: pd.Series, regime: MarketRegime) -> Signal | None:
        if not self.is_regime_allowed(regime):
            self._prev_bar = bar
            return None

        ema_fast = indicators.get(f"ema_{self.ema_fast}")
        ema_slow = indicators.get(f"ema_{self.ema_slow}")
        atr_val = indicators.get("atr_14")

        if any(v is None or (isinstance(v, float) and pd.isna(v))
               for v in [ema_fast, ema_slow, atr_val]):
            self._prev_bar = bar
            return None

        if atr_val <= 0:
            self._prev_bar = bar
            return None

        # Need previous bar for pattern check
        if self._prev_bar is None:
            self._prev_bar = bar
            return None

        prev = self._prev_bar
        signal = None

        # Uptrend: fast EMA above slow
        if ema_fast > ema_slow:
            # Pullback bar (previous was bearish) + current is bullish
            prev_bearish = prev.close < prev.open
            curr_bullish = bar.close > bar.open
            near_ema = bar.low <= ema_fast + atr_val * 0.5

            if prev_bearish and curr_bullish and near_ema:
                stop = bar.close - self.stop_loss_atr * atr_val
                risk = bar.close - stop
                tp = bar.close + risk * self.rr_target
                confidence = 1.0
                signal = Signal(
                    direction=SignalDirection.LONG,
                    instrument=bar.instrument,
                    entry_price=bar.close,
                    stop_loss=stop,
                    take_profit=tp,
                    confidence=confidence,
                    strategy_name=self.name,
                    metadata={"trend": "up"},
                )

        # Downtrend: fast EMA below slow
        elif ema_fast < ema_slow:
            prev_bullish = prev.close > prev.open
            curr_bearish = bar.close < bar.open
            near_ema = bar.high >= ema_fast - atr_val * 0.5

            if prev_bullish and curr_bearish and near_ema:
                stop = bar.close + self.stop_loss_atr * atr_val
                risk = stop - bar.close
                tp = bar.close - risk * self.rr_target
                confidence = 1.0
                signal = Signal(
                    direction=SignalDirection.SHORT,
                    instrument=bar.instrument,
                    entry_price=bar.close,
                    stop_loss=stop,
                    take_profit=tp,
                    confidence=confidence,
                    strategy_name=self.name,
                    metadata={"trend": "down"},
                )

        self._prev_bar = bar
        return signal

    def should_exit(self, position: Position, bar: Bar, indicators: pd.Series) -> Signal | None:
        atr_val = indicators.get("atr_14")
        if atr_val is None or pd.isna(atr_val) or atr_val <= 0:
            return None

        inst = position.instrument

        if position.direction == SignalDirection.LONG:
            self._best_price[inst] = max(self._best_price.get(inst, bar.high), bar.high)

            # Hard stop
            if position.stop_loss is not None and bar.low <= position.stop_loss:
                self._best_price.pop(inst, None)
                return self._exit_signal(position, position.stop_loss, "stop_loss")

            # Trailing stop (only after in profit)
            trailing_stop = self._best_price[inst] - self.trailing_stop_atr * atr_val
            if trailing_stop > position.entry_price and bar.low <= trailing_stop:
                self._best_price.pop(inst, None)
                return self._exit_signal(position, trailing_stop, "trailing_stop")

            # Take profit
            if position.take_profit is not None and bar.high >= position.take_profit:
                self._best_price.pop(inst, None)
                return self._exit_signal(position, position.take_profit, "take_profit")

        else:  # SHORT
            self._best_price[inst] = min(self._best_price.get(inst, bar.low), bar.low)

            if position.stop_loss is not None and bar.high >= position.stop_loss:
                self._best_price.pop(inst, None)
                return self._exit_signal(position, position.stop_loss, "stop_loss")

            trailing_stop = self._best_price[inst] + self.trailing_stop_atr * atr_val
            if trailing_stop < position.entry_price and bar.high >= trailing_stop:
                self._best_price.pop(inst, None)
                return self._exit_signal(position, trailing_stop, "trailing_stop")

            if position.take_profit is not None and bar.low <= position.take_profit:
                self._best_price.pop(inst, None)
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
            strategy_name="trend_following",
            metadata={"exit_reason": reason},
        )
