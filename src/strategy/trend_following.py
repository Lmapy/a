from __future__ import annotations

import pandas as pd

from src.data.models import Bar, MarketRegime, Position, Signal, SignalDirection
from src.strategy.base import BaseStrategy


class TrendFollowingStrategy(BaseStrategy):
    """Trade in the direction of the trend using EMA crossover and pullbacks.

    Entry:
        - EMA20 > EMA50 (uptrend): buy when price pulls back to within
          pullback_atr ATRs of EMA20 and closes back above it
        - EMA20 < EMA50 (downtrend): sell when price rallies near EMA20
          and closes back below it
        - ADX > threshold confirms trend strength
    Exit:
        - Trailing stop at trailing_stop_atr * ATR from best price
        - Hard stop at stop_loss_atr * ATR from entry
    """

    name = "trend_following"
    allowed_regimes = ["strong_trend_up", "strong_trend_down", "weak_trend"]
    blocked_regimes = ["ranging", "low_volatility"]

    def __init__(self, params: dict | None = None):
        p = params or {}
        self.ema_fast = p.get("ema_fast", 20)
        self.ema_slow = p.get("ema_slow", 50)
        self.adx_threshold = p.get("adx_threshold", 20)
        self.pullback_atr = p.get("pullback_atr", 0.75)
        self.stop_loss_atr = p.get("stop_loss_atr", 2.5)
        self.trailing_stop_atr = p.get("trailing_stop_atr", 2.5)
        self._best_price: dict[str, float] = {}

    def on_bar(self, bar: Bar, indicators: pd.Series, regime: MarketRegime) -> Signal | None:
        if not self.is_regime_allowed(regime):
            return None

        ema_fast = indicators.get("ema_20")
        ema_slow = indicators.get("ema_50")
        adx_val = indicators.get("adx_14")
        atr_val = indicators.get("atr_14")
        plus_di = indicators.get("plus_di")
        minus_di = indicators.get("minus_di")

        if any(v is None or (isinstance(v, float) and pd.isna(v))
               for v in [ema_fast, ema_slow, adx_val, atr_val]):
            return None

        if atr_val <= 0 or adx_val < self.adx_threshold:
            return None

        ema_spread = abs(ema_fast - ema_slow) / atr_val

        # Uptrend: fast EMA above slow, price near fast EMA, bouncing up
        if ema_fast > ema_slow:
            near_ema = abs(bar.low - ema_fast) < self.pullback_atr * atr_val
            bounced = bar.close > ema_fast
            di_confirms = plus_di is not None and minus_di is not None and plus_di > minus_di

            if near_ema and bounced and di_confirms:
                stop = bar.close - self.stop_loss_atr * atr_val
                tp = bar.close + self.stop_loss_atr * 2 * atr_val  # 2:1 R:R target
                confidence = min(adx_val / 50.0, 1.0) * min(ema_spread, 1.0)
                return Signal(
                    direction=SignalDirection.LONG,
                    instrument=bar.instrument,
                    entry_price=bar.close,
                    stop_loss=stop,
                    take_profit=tp,
                    confidence=confidence,
                    strategy_name=self.name,
                    metadata={"adx": adx_val, "trend": "up", "ema_spread": ema_spread},
                )

        # Downtrend: fast EMA below slow, price near fast EMA, rejected down
        if ema_fast < ema_slow:
            near_ema = abs(bar.high - ema_fast) < self.pullback_atr * atr_val
            rejected = bar.close < ema_fast
            di_confirms = plus_di is not None and minus_di is not None and minus_di > plus_di

            if near_ema and rejected and di_confirms:
                stop = bar.close + self.stop_loss_atr * atr_val
                tp = bar.close - self.stop_loss_atr * 2 * atr_val
                confidence = min(adx_val / 50.0, 1.0) * min(ema_spread, 1.0)
                return Signal(
                    direction=SignalDirection.SHORT,
                    instrument=bar.instrument,
                    entry_price=bar.close,
                    stop_loss=stop,
                    take_profit=tp,
                    confidence=confidence,
                    strategy_name=self.name,
                    metadata={"adx": adx_val, "trend": "down", "ema_spread": ema_spread},
                )

        return None

    def should_exit(self, position: Position, bar: Bar, indicators: pd.Series) -> Signal | None:
        atr_val = indicators.get("atr_14")
        if atr_val is None or pd.isna(atr_val) or atr_val <= 0:
            return None

        inst = position.instrument

        if position.direction == SignalDirection.LONG:
            self._best_price[inst] = max(self._best_price.get(inst, bar.high), bar.high)
            trailing_stop = self._best_price[inst] - self.trailing_stop_atr * atr_val

            # Hard stop
            if position.stop_loss is not None and bar.low <= position.stop_loss:
                self._best_price.pop(inst, None)
                return self._exit_signal(position, position.stop_loss, "stop_loss")

            # Trailing stop (only if it's tighter than the hard stop)
            if trailing_stop > (position.stop_loss or 0) and bar.low <= trailing_stop:
                self._best_price.pop(inst, None)
                return self._exit_signal(position, trailing_stop, "trailing_stop")

            # Take profit
            if position.take_profit is not None and bar.high >= position.take_profit:
                self._best_price.pop(inst, None)
                return self._exit_signal(position, position.take_profit, "take_profit")

        else:  # SHORT
            self._best_price[inst] = min(self._best_price.get(inst, bar.low), bar.low)
            trailing_stop = self._best_price[inst] + self.trailing_stop_atr * atr_val

            if position.stop_loss is not None and bar.high >= position.stop_loss:
                self._best_price.pop(inst, None)
                return self._exit_signal(position, position.stop_loss, "stop_loss")

            if trailing_stop < (position.stop_loss or float("inf")) and bar.high >= trailing_stop:
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
