from __future__ import annotations

import pandas as pd

from src.data.models import Bar, MarketRegime, Position, Signal, SignalDirection
from src.strategy.base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """Trade reversions to VWAP when price extends too far.

    Entry: price > entry_threshold_atr ATRs from VWAP → fade the move.
    Stop: entry +/- stop_loss_atr * ATR.
    Take profit: back toward VWAP (take_profit_atr * ATR from entry).
    Filter: only in ranging/low-volatility regimes with sufficient volume.
    """

    name = "mean_reversion"
    allowed_regimes = ["ranging", "low_volatility", "weak_trend"]
    blocked_regimes = ["strong_trend_up", "strong_trend_down"]

    def __init__(self, params: dict | None = None):
        p = params or {}
        self.entry_threshold_atr = p.get("entry_threshold_atr", 1.5)
        self.stop_loss_atr = p.get("stop_loss_atr", 2.0)
        self.take_profit_atr = p.get("take_profit_atr", 1.0)
        self.rsi_oversold = p.get("rsi_oversold", 30)
        self.rsi_overbought = p.get("rsi_overbought", 70)
        self.min_volume_percentile = p.get("min_volume_percentile", 30)

    def on_bar(self, bar: Bar, indicators: pd.Series, regime: MarketRegime) -> Signal | None:
        if not self.is_regime_allowed(regime):
            return None

        atr_val = indicators.get("atr_14")
        vwap_val = indicators.get("vwap")
        rsi_val = indicators.get("rsi_14")
        volume = bar.volume
        volume_sma = indicators.get("volume_sma")

        if any(v is None or (isinstance(v, float) and pd.isna(v))
               for v in [atr_val, vwap_val, rsi_val, volume_sma]):
            return None

        if atr_val <= 0:
            return None

        # Volume filter
        if volume_sma > 0 and volume < volume_sma * (self.min_volume_percentile / 100.0):
            return None

        distance_atr = (bar.close - vwap_val) / atr_val

        # Short signal: price too far above VWAP + RSI overbought
        if distance_atr > self.entry_threshold_atr and rsi_val > self.rsi_overbought:
            stop = bar.close + self.stop_loss_atr * atr_val
            tp = bar.close - self.take_profit_atr * atr_val
            confidence = min(abs(distance_atr) / (self.entry_threshold_atr * 2), 1.0)
            return Signal(
                direction=SignalDirection.SHORT,
                instrument=bar.instrument,
                entry_price=bar.close,
                stop_loss=stop,
                take_profit=tp,
                confidence=confidence,
                strategy_name=self.name,
                metadata={"distance_atr": distance_atr, "rsi": rsi_val},
            )

        # Long signal: price too far below VWAP + RSI oversold
        if distance_atr < -self.entry_threshold_atr and rsi_val < self.rsi_oversold:
            stop = bar.close - self.stop_loss_atr * atr_val
            tp = bar.close + self.take_profit_atr * atr_val
            confidence = min(abs(distance_atr) / (self.entry_threshold_atr * 2), 1.0)
            return Signal(
                direction=SignalDirection.LONG,
                instrument=bar.instrument,
                entry_price=bar.close,
                stop_loss=stop,
                take_profit=tp,
                confidence=confidence,
                strategy_name=self.name,
                metadata={"distance_atr": distance_atr, "rsi": rsi_val},
            )

        return None

    def should_exit(self, position: Position, bar: Bar, indicators: pd.Series) -> Signal | None:
        atr_val = indicators.get("atr_14")
        vwap_val = indicators.get("vwap")
        if atr_val is None or vwap_val is None:
            return None

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

        # Exit if price crosses back through VWAP (mean reverted)
        if position.direction == SignalDirection.LONG and bar.close >= vwap_val:
            return self._exit_signal(position, bar.close, "vwap_reversion")
        if position.direction == SignalDirection.SHORT and bar.close <= vwap_val:
            return self._exit_signal(position, bar.close, "vwap_reversion")

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
            strategy_name="mean_reversion",
            metadata={"exit_reason": reason},
        )
