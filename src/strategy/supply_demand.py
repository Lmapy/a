from __future__ import annotations

from collections import deque

import pandas as pd

from src.data.models import Bar, INSTRUMENT_SPECS, MarketRegime, Position, Signal, SignalDirection
from src.strategy.base import BaseStrategy


class SupplyDemandStrategy(BaseStrategy):
    """Trade bounces from supply and demand zones on 1H/4H timeframes.

    Concept (bernt-style supply/demand):
        1. Identify "impulsive" moves: candles with large body and strong follow-through
        2. The consolidation (base) before the impulse is the zone
        3. When price returns to that zone, expect institutional orders to defend it
        4. Enter on the first touch with a tight stop just beyond the zone

    Zone detection (using 5-min bars aggregated to 1H):
        - Build 1H candles from 5-min bars
        - Detect impulse candles: body > 1.5x ATR and close near extreme
        - The 1H candle BEFORE the impulse = the zone
        - Zone is defined by the high/low of that base candle

    Entry on 5-min chart:
        - Price enters the zone
        - Wait for rejection candle (wick into zone, close outside)
        - Enter on close of rejection candle

    Stop: Beyond zone boundary + buffer
    Target: 2x risk or opposing zone
    """

    name = "supply_demand"
    allowed_regimes = ["ranging", "weak_trend", "low_volatility", "high_volatility",
                       "strong_trend_up", "strong_trend_down"]
    blocked_regimes = []

    def __init__(self, params: dict | None = None):
        p = params or {}
        self.zone_atr_mult = p.get("zone_atr_mult", 2.0)  # impulse body must be > N x ATR
        self.zone_body_pct = p.get("zone_body_pct", 0.75)  # impulse body must be > N% of range
        self.stop_buffer_ticks = p.get("stop_buffer_ticks", 12)
        self.take_profit_rr = p.get("take_profit_rr", 2.0)
        self.max_zone_age_bars = p.get("max_zone_age_bars", 240)  # 20 hours worth of 5-min
        self.max_zones = p.get("max_zones", 6)
        self.hourly_bars_for_zone = p.get("hourly_bars_for_zone", 12)  # 12 5-min = 1 hour
        self.rejection_wick_pct = p.get("rejection_wick_pct", 0.55)
        self.min_tick_profit = p.get("min_tick_profit", 0)

        # State per instrument
        self._bar_buffer: dict[str, deque] = {}  # recent 5-min bars
        self._hourly_bars: dict[str, list] = {}  # aggregated hourly bars
        self._zones: dict[str, list] = {}  # active supply/demand zones
        self._zone_traded: dict[str, bool] = {}  # one trade per zone per day
        self._bar_count: dict[str, int] = {}
        self._atr_values: dict[str, deque] = {}  # rolling ATR for zones
        self._current_date: dict[str, object] = {}

    def reset_daily(self) -> None:
        """Reset daily trade tracking. Zones persist across days."""
        self._zone_traded.clear()

    def on_bar(self, bar: Bar, indicators: pd.Series, regime: MarketRegime) -> Signal | None:
        if not self.is_regime_allowed(regime):
            return None

        inst = bar.instrument
        spec = INSTRUMENT_SPECS.get(inst, {})
        tick_size = spec.get("tick_size", 0.25)

        # Track date changes
        bar_date = bar.timestamp.date() if hasattr(bar.timestamp, 'date') else None
        if bar_date and bar_date != self._current_date.get(inst):
            if self._current_date.get(inst) is not None:
                self.reset_daily()
            self._current_date[inst] = bar_date

        # Initialize buffers
        if inst not in self._bar_buffer:
            self._bar_buffer[inst] = deque(maxlen=self.hourly_bars_for_zone * 50)
            self._hourly_bars[inst] = []
            self._zones[inst] = []
            self._atr_values[inst] = deque(maxlen=20)

        self._bar_buffer[inst].append(bar)
        self._bar_count[inst] = self._bar_count.get(inst, 0) + 1

        # Aggregate to hourly bars every N 5-min bars
        if self._bar_count[inst] % self.hourly_bars_for_zone == 0:
            hourly = self._aggregate_hourly(inst)
            if hourly:
                self._hourly_bars[inst].append(hourly)
                self._update_atr(inst)
                self._detect_zones(inst, tick_size)

        # Expire old zones
        self._expire_zones(inst)

        # Check for entry signals at active zones
        if self._zone_traded.get(inst, 0) >= 2:
            return None

        return self._check_zone_entry(bar, inst, tick_size)

    def _aggregate_hourly(self, inst: str) -> dict | None:
        """Aggregate recent 5-min bars into one hourly bar."""
        recent = list(self._bar_buffer[inst])[-self.hourly_bars_for_zone:]
        if len(recent) < self.hourly_bars_for_zone:
            return None

        return {
            "open": recent[0].open,
            "high": max(b.high for b in recent),
            "low": min(b.low for b in recent),
            "close": recent[-1].close,
            "timestamp": recent[-1].timestamp,
            "bar_idx": self._bar_count[inst],
        }

    def _update_atr(self, inst: str) -> None:
        """Update rolling ATR from hourly bars."""
        bars = self._hourly_bars[inst]
        if len(bars) < 2:
            return
        last = bars[-1]
        prev = bars[-2]
        tr = max(
            last["high"] - last["low"],
            abs(last["high"] - prev["close"]),
            abs(last["low"] - prev["close"]),
        )
        self._atr_values[inst].append(tr)

    def _detect_zones(self, inst: str, tick_size: float) -> None:
        """Check if the latest hourly bar is an impulse move and create a zone."""
        bars = self._hourly_bars[inst]
        if len(bars) < 3:
            return
        if not self._atr_values[inst]:
            return

        current = bars[-1]
        base = bars[-2]  # candle before impulse = zone
        atr = sum(self._atr_values[inst]) / len(self._atr_values[inst])

        body = abs(current["close"] - current["open"])
        candle_range = current["high"] - current["low"]
        if candle_range < tick_size:
            return

        body_pct = body / candle_range

        # Check if this is an impulse candle
        if body < atr * self.zone_atr_mult:
            return
        if body_pct < self.zone_body_pct:
            return

        # Determine direction of impulse
        if current["close"] > current["open"]:
            # Bullish impulse → demand zone below (the base candle)
            zone = {
                "type": "demand",
                "high": base["high"],
                "low": base["low"],
                "created_bar": self._bar_count[inst],
                "touched": False,
            }
        else:
            # Bearish impulse → supply zone above (the base candle)
            zone = {
                "type": "supply",
                "high": base["high"],
                "low": base["low"],
                "created_bar": self._bar_count[inst],
                "touched": False,
            }

        # Don't add duplicate zones (within 10 ticks of existing)
        for existing in self._zones[inst]:
            if existing["type"] == zone["type"]:
                if abs(existing["high"] - zone["high"]) < 10 * tick_size:
                    return

        self._zones[inst].append(zone)

        # Limit total zones
        if len(self._zones[inst]) > self.max_zones:
            self._zones[inst] = self._zones[inst][-self.max_zones:]

    def _expire_zones(self, inst: str) -> None:
        """Remove zones that are too old."""
        if inst not in self._zones:
            return
        current_bar = self._bar_count.get(inst, 0)
        self._zones[inst] = [
            z for z in self._zones[inst]
            if current_bar - z["created_bar"] < self.max_zone_age_bars
        ]

    def _check_zone_entry(self, bar: Bar, inst: str, tick_size: float) -> Signal | None:
        """Check if current bar provides an entry at any active zone."""
        if inst not in self._zones:
            return None

        candle_range = bar.high - bar.low
        if candle_range < tick_size:
            return None

        for zone in self._zones[inst]:
            if zone["touched"]:
                continue

            if zone["type"] == "demand":
                # Price must dip into demand zone and reject (close above zone high)
                if bar.low <= zone["high"] and bar.close > zone["high"]:
                    # Check rejection wick
                    lower_wick = min(bar.open, bar.close) - bar.low
                    if lower_wick / candle_range >= self.rejection_wick_pct:
                        zone["touched"] = True
                        self._zone_traded[inst] = self._zone_traded.get(inst, 0) + 1

                        stop = zone["low"] - self.stop_buffer_ticks * tick_size
                        risk = bar.close - stop
                        tp = bar.close + risk * self.take_profit_rr

                        if self.min_tick_profit > 0:
                            min_tp = bar.close + self.min_tick_profit * tick_size
                            tp = max(tp, min_tp)

                        return Signal(
                            direction=SignalDirection.LONG,
                            instrument=inst,
                            entry_price=bar.close,
                            stop_loss=stop,
                            take_profit=tp,
                            confidence=1.0,
                            strategy_name=self.name,
                            metadata={
                                "zone_type": "demand",
                                "zone_high": zone["high"],
                                "zone_low": zone["low"],
                            },
                        )

            elif zone["type"] == "supply":
                # Price must push into supply zone and reject (close below zone low)
                if bar.high >= zone["low"] and bar.close < zone["low"]:
                    # Check rejection wick
                    upper_wick = bar.high - max(bar.open, bar.close)
                    if upper_wick / candle_range >= self.rejection_wick_pct:
                        zone["touched"] = True
                        self._zone_traded[inst] = self._zone_traded.get(inst, 0) + 1

                        stop = zone["high"] + self.stop_buffer_ticks * tick_size
                        risk = stop - bar.close
                        tp = bar.close - risk * self.take_profit_rr

                        if self.min_tick_profit > 0:
                            min_tp = bar.close - self.min_tick_profit * tick_size
                            tp = min(tp, min_tp)

                        return Signal(
                            direction=SignalDirection.SHORT,
                            instrument=inst,
                            entry_price=bar.close,
                            stop_loss=stop,
                            take_profit=tp,
                            confidence=1.0,
                            strategy_name=self.name,
                            metadata={
                                "zone_type": "supply",
                                "zone_high": zone["high"],
                                "zone_low": zone["low"],
                            },
                        )

        return None

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
            stop_loss=0,
            take_profit=0,
            confidence=1.0,
            strategy_name="supply_demand",
            metadata={"exit_reason": reason},
        )
