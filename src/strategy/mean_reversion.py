from __future__ import annotations

import pandas as pd

from src.data.models import Bar, MarketRegime, Position, Signal, SignalDirection
from src.strategy.base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """Trade reversions to VWAP when price overextends.

    Entry logic (uses OR, not AND, for flexibility):
        - Price is > threshold ATRs from VWAP, OR
        - RSI is extreme AND price is moving away from VWAP
    Exit:
        - Take profit at a fraction of the distance back to VWAP
        - Stop loss at entry +/- stop_loss_atr * ATR
        - Time-based exit: close after max_hold_bars if neither hit
    """

    name = "mean_reversion"
    allowed_regimes = ["ranging", "low_volatility", "weak_trend", "high_volatility"]
    blocked_regimes = []

    def __init__(self, params: dict | None = None):
        p = params or {}
        self.entry_threshold_atr = p.get("entry_threshold_atr", 1.0)
        self.stop_loss_atr = p.get("stop_loss_atr", 2.5)
        self.take_profit_atr = p.get("take_profit_atr", 1.5)
        self.rsi_oversold = p.get("rsi_oversold", 35)
        self.rsi_overbought = p.get("rsi_overbought", 65)
        self.min_volume_percentile = p.get("min_volume_percentile", 10)
        self.max_hold_bars = p.get("max_hold_bars", 20)
        self._bars_in_trade: dict[str, int] = {}

    def on_bar(self, bar: Bar, indicators: pd.Series, regime: MarketRegime) -> Signal | None:
        if not self.is_regime_allowed(regime):
            return None

        atr_val = indicators.get("atr_14")
        vwap_val = indicators.get("vwap")
        rsi_val = indicators.get("rsi_14")
        bb_upper = indicators.get("bb_upper")
        bb_lower = indicators.get("bb_lower")

        if any(v is None or (isinstance(v, float) and pd.isna(v))
               for v in [atr_val, vwap_val, rsi_val]):
            return None

        if atr_val <= 0:
            return None

        distance_atr = (bar.close - vwap_val) / atr_val

        # SHORT: price extended above VWAP or hitting upper Bollinger Band
        short_vwap = distance_atr > self.entry_threshold_atr
        short_rsi = rsi_val is not None and rsi_val > self.rsi_overbought
        short_bb = bb_upper is not None and not pd.isna(bb_upper) and bar.close > bb_upper

        if short_vwap and (short_rsi or short_bb):
            stop = bar.close + self.stop_loss_atr * atr_val
            tp = bar.close - self.take_profit_atr * atr_val
            confidence = min(abs(distance_atr) / 3.0, 1.0)
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

        # LONG: price extended below VWAP or hitting lower Bollinger Band
        long_vwap = distance_atr < -self.entry_threshold_atr
        long_rsi = rsi_val is not None and rsi_val < self.rsi_oversold
        long_bb = bb_lower is not None and not pd.isna(bb_lower) and bar.close < bb_lower

        if long_vwap and (long_rsi or long_bb):
            stop = bar.close - self.stop_loss_atr * atr_val
            tp = bar.close + self.take_profit_atr * atr_val
            confidence = min(abs(distance_atr) / 3.0, 1.0)
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
        inst = position.instrument
        self._bars_in_trade[inst] = self._bars_in_trade.get(inst, 0) + 1

        # Check stop loss
        if position.stop_loss is not None:
            if position.direction == SignalDirection.LONG and bar.low <= position.stop_loss:
                self._bars_in_trade.pop(inst, None)
                return self._exit_signal(position, position.stop_loss, "stop_loss")
            if position.direction == SignalDirection.SHORT and bar.high >= position.stop_loss:
                self._bars_in_trade.pop(inst, None)
                return self._exit_signal(position, position.stop_loss, "stop_loss")

        # Check take profit
        if position.take_profit is not None:
            if position.direction == SignalDirection.LONG and bar.high >= position.take_profit:
                self._bars_in_trade.pop(inst, None)
                return self._exit_signal(position, position.take_profit, "take_profit")
            if position.direction == SignalDirection.SHORT and bar.low <= position.take_profit:
                self._bars_in_trade.pop(inst, None)
                return self._exit_signal(position, position.take_profit, "take_profit")

        # Time-based exit
        if self._bars_in_trade.get(inst, 0) >= self.max_hold_bars:
            self._bars_in_trade.pop(inst, None)
            return self._exit_signal(position, bar.close, "time_exit")

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
