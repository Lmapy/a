from __future__ import annotations

import pandas as pd

from src.data.models import Bar, MarketRegime, Position, Signal, SignalDirection
from src.strategy.base import BaseStrategy


class TrendFollowingStrategy(BaseStrategy):
    """Trade pullbacks to the fast EMA in the direction of the trend.

    Entry: EMA20 > EMA50 (uptrend) and price pulls back near EMA20, then bounces.
    Stop: below the pullback low (clamped to stop_loss_atr * ATR).
    Take profit: trailing stop at trailing_stop_atr * ATR from highest close.
    Filter: ADX > threshold, trending regime.
    """

    name = "trend_following"
    allowed_regimes = ["strong_trend_up", "strong_trend_down", "weak_trend"]
    blocked_regimes = ["ranging", "low_volatility"]

    def __init__(self, params: dict | None = None):
        p = params or {}
        self.ema_fast = p.get("ema_fast", 20)
        self.ema_slow = p.get("ema_slow", 50)
        self.adx_threshold = p.get("adx_threshold", 25)
        self.pullback_atr = p.get("pullback_atr", 0.5)
        self.stop_loss_atr = p.get("stop_loss_atr", 2.0)
        self.trailing_stop_atr = p.get("trailing_stop_atr", 2.0)
        self._highest_since_entry: dict[str, float] = {}
        self._lowest_since_entry: dict[str, float] = {}

    def on_bar(self, bar: Bar, indicators: pd.Series, regime: MarketRegime) -> Signal | None:
        if not self.is_regime_allowed(regime):
            return None

        ema_fast = indicators.get("ema_20")
        ema_slow = indicators.get("ema_50")
        adx_val = indicators.get("adx_14")
        atr_val = indicators.get("atr_14")

        if any(v is None or (isinstance(v, float) and pd.isna(v))
               for v in [ema_fast, ema_slow, adx_val, atr_val]):
            return None

        if atr_val <= 0 or adx_val < self.adx_threshold:
            return None

        # Uptrend: EMA fast > EMA slow, price pulled back near fast EMA
        if ema_fast > ema_slow:
            pullback_zone = ema_fast + self.pullback_atr * atr_val
            if bar.low <= pullback_zone and bar.close > ema_fast:
                stop = bar.close - self.stop_loss_atr * atr_val
                tp = bar.close + self.trailing_stop_atr * 2 * atr_val
                confidence = min(adx_val / 50.0, 1.0)
                return Signal(
                    direction=SignalDirection.LONG,
                    instrument=bar.instrument,
                    entry_price=bar.close,
                    stop_loss=stop,
                    take_profit=tp,
                    confidence=confidence,
                    strategy_name=self.name,
                    metadata={"adx": adx_val, "trend": "up"},
                )

        # Downtrend: EMA fast < EMA slow, price pulled back near fast EMA
        if ema_fast < ema_slow:
            pullback_zone = ema_fast - self.pullback_atr * atr_val
            if bar.high >= pullback_zone and bar.close < ema_fast:
                stop = bar.close + self.stop_loss_atr * atr_val
                tp = bar.close - self.trailing_stop_atr * 2 * atr_val
                confidence = min(adx_val / 50.0, 1.0)
                return Signal(
                    direction=SignalDirection.SHORT,
                    instrument=bar.instrument,
                    entry_price=bar.close,
                    stop_loss=stop,
                    take_profit=tp,
                    confidence=confidence,
                    strategy_name=self.name,
                    metadata={"adx": adx_val, "trend": "down"},
                )

        return None

    def should_exit(self, position: Position, bar: Bar, indicators: pd.Series) -> Signal | None:
        atr_val = indicators.get("atr_14")
        if atr_val is None or pd.isna(atr_val) or atr_val <= 0:
            return None

        inst = position.instrument

        # Track trailing stop reference price
        if position.direction == SignalDirection.LONG:
            self._highest_since_entry[inst] = max(
                self._highest_since_entry.get(inst, bar.high), bar.high
            )
            trailing_stop = self._highest_since_entry[inst] - self.trailing_stop_atr * atr_val

            # Check stop loss
            if position.stop_loss is not None and bar.low <= position.stop_loss:
                self._highest_since_entry.pop(inst, None)
                return self._exit_signal(position, position.stop_loss, "stop_loss")

            # Check trailing stop
            if bar.low <= trailing_stop:
                self._highest_since_entry.pop(inst, None)
                return self._exit_signal(position, trailing_stop, "trailing_stop")

        else:  # SHORT
            self._lowest_since_entry[inst] = min(
                self._lowest_since_entry.get(inst, bar.low), bar.low
            )
            trailing_stop = self._lowest_since_entry[inst] + self.trailing_stop_atr * atr_val

            if position.stop_loss is not None and bar.high >= position.stop_loss:
                self._lowest_since_entry.pop(inst, None)
                return self._exit_signal(position, position.stop_loss, "stop_loss")

            if bar.high >= trailing_stop:
                self._lowest_since_entry.pop(inst, None)
                return self._exit_signal(position, trailing_stop, "trailing_stop")

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
